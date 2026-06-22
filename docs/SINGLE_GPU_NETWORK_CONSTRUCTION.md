# Building Spiking Neural Networks (Single GPU)

On a single GPU, SuperNeuroABM gives you **two ways** to build a spiking neural
network:

1. **Agent-by-agent** — `create_soma()` / `create_synapse()`: add neurons and
   connections one at a time. Incremental, returns each agent's id.
2. **Bulk lists** — `create_from_lists()`: describe every soma and synapse as a
   Python list and build the whole model in a single call.

Both produce an identical model (same `locations` connectivity); they differ
only in convenience. Pick agent-by-agent when you build incrementally or need
each id as you go; pick bulk lists when you already have the network as data.

> **This doc is single-GPU only.** For a partitioned network across multiple
> GPUs / nodes, construction is file-based (`load_post_owned()` /
> `load_from_adjacency()`) and has its own ownership contract — see
> [`PARTITION_LOADING.md`](PARTITION_LOADING.md). (For the distributed *run* and
> HPC launch, see [`DISTRIBUTED_SIMULATION.md`](DISTRIBUTED_SIMULATION.md).)

---

## Method 1: Agent-by-Agent (`create_soma` / `create_synapse`)

Fine-grained control: you create each neuron, hold onto its id, and wire
synapses by referencing those ids.

```python
from superneuroabm.model import NeuromorphicModel

model = NeuromorphicModel()

# --- Neurons (somas) ---
input_neuron  = model.create_soma(breed="lif_soma", config_name="config_0")
hidden_neuron = model.create_soma(breed="lif_soma", config_name="config_0")
output_neuron = model.create_soma(breed="lif_soma", config_name="config_0")

# --- Connections (synapses) ---
# External input -> input neuron. pre_soma_id=-1 means "external input".
input_synapse = model.create_synapse(
    breed="single_exp_synapse",
    pre_soma_id=-1,
    post_soma_id=input_neuron,
    config_name="config_0",
    overrides={"hyperparameters": {"weight": 500.0}},
)

# Input -> Hidden, with STDP learning enabled via a learning_rule.
model.create_synapse(
    breed="single_exp_synapse",
    pre_soma_id=input_neuron,
    post_soma_id=hidden_neuron,
    config_name="config_0",
    learning_rule="exp_pair_wise_stdp",      # omit for a fixed-weight synapse
    overrides={"hyperparameters": {"weight": 10.0}},
)

# Hidden -> Output (fixed weight).
model.create_synapse(
    breed="single_exp_synapse",
    pre_soma_id=hidden_neuron,
    post_soma_id=output_neuron,
    config_name="config_0",
    overrides={"hyperparameters": {"weight": 15.0}},
)

# --- Run ---
model.setup(use_gpu=True)                    # compile kernels, allocate memory
model.add_spike(synapse_id=input_synapse, tick=10, value=1.0)
model.simulate(ticks=100, update_data_ticks=1)

print(f"Input spikes:  {model.get_spike_times(soma_id=input_neuron)}")
print(f"Hidden spikes: {model.get_spike_times(soma_id=hidden_neuron)}")
print(f"Output spikes: {model.get_spike_times(soma_id=output_neuron)}")
```

**Network created:**
```
External Input --(input_synapse, w=500)--> Input
   Input --(STDP, w=10)--> Hidden --(w=15)--> Output
```

### Building larger networks programmatically

Because each `create_*` call is just a function call, you can generate structure
with ordinary Python loops:

```python
import numpy as np
from superneuroabm.model import NeuromorphicModel

model = NeuromorphicModel(enable_internal_states_tracking=False)  # save memory

# 100 input neurons, 10 output neurons.
input_layer  = [model.create_soma(breed="lif_soma", config_name="config_0")
                for _ in range(100)]
output_layer = [model.create_soma(breed="lif_soma", config_name="config_0")
                for _ in range(10)]

# All-to-all input -> output with random weights.
for pre in input_layer:
    for post in output_layer:
        model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=pre,
            post_soma_id=post,
            config_name="config_0",
            overrides={"hyperparameters": {"weight": float(np.random.uniform(5, 15))}},
        )

model.setup(use_gpu=True)
```

**Good for:** incremental construction, algorithmic patterns (grids, layers,
random graphs), mixing neuron breeds (`lif_soma`, `izh_soma`, `hg_lif_soma`)
on the fly.

**Less good for:** very large or pre-computed networks, where the per-call loop
is just boilerplate around data you already have — use Method 2.

---

## Method 2: Bulk Lists (`create_from_lists`)

When you already have your somas and synapses as data (single-GPU work, nothing
to serialize), `create_from_lists()` builds the **entire** model in one call. It
is the bulk counterpart to Method 1.

### Entry schema

- **Soma** dict: `{"id", "breed"?, "config"?, "overrides"?}`
- **Synapse** dict: `{"id", "pre", "post", "breed"?, "config"?, "overrides"?, "learning_rule"?, "learning_rule_config"?}`
- **You assign every `id`.** A synapse names its endpoints by soma id via
  `pre`/`post`; `pre = -1` marks an external-input synapse.
- Omitted `breed`/`config` fall back to the method defaults
  (`lif_soma`/`config_0` for somas, `single_exp_synapse`/`config_0` for synapses).
- `overrides` is grouped by property type: `"hyperparameters"`,
  `"internal_states"`, `"learning_hyperparameters"`, `"learning_internal_states"`.

`create_from_lists()` is a **one-shot, whole-model builder**: call it exactly
once on a fresh model. It is mutually exclusive with the incremental
`create_soma`/`create_synapse` — use one construction path per model.

### Basic example (same network as Method 1)

```python
from superneuroabm.model import NeuromorphicModel

model = NeuromorphicModel()

model.create_from_lists(
    somas=[
        {"id": 0},   # input
        {"id": 1},   # hidden
        {"id": 2},   # output
    ],
    synapses=[
        # External input -> input neuron (pre = -1)
        {"id": 100, "pre": -1, "post": 0,
         "overrides": {"hyperparameters": {"weight": 500.0}}},
        # Input -> Hidden, with STDP learning
        {"id": 101, "pre": 0, "post": 1,
         "learning_rule": "exp_pair_wise_stdp",
         "overrides": {"hyperparameters": {"weight": 10.0}}},
        # Hidden -> Output (fixed weight)
        {"id": 102, "pre": 1, "post": 2,
         "overrides": {"hyperparameters": {"weight": 15.0}}},
    ],
)

model.setup(use_gpu=True)
model.add_spike(synapse_id=100, tick=10, value=1.0)   # inject on external synapse
model.simulate(ticks=100, update_data_ticks=1)

print(f"Output spikes: {model.get_spike_times(soma_id=2)}")
```

This builds the **same model** as the Method 1 feedforward example — you just
describe it as data and submit it in one call.

### Generating the lists programmatically

The lists are plain data, so build them however is convenient (comprehensions, a
dataframe, your own graph object), then pass them in:

```python
import numpy as np
from superneuroabm.model import NeuromorphicModel

N_IN, N_OUT = 100, 10
somas = [{"id": i} for i in range(N_IN + N_OUT)]

synapses, sid = [], 1000
for pre in range(N_IN):
    for post in range(N_IN, N_IN + N_OUT):
        synapses.append({
            "id": sid, "pre": pre, "post": post,
            "overrides": {"hyperparameters": {"weight": float(np.random.uniform(5, 15))}},
        })
        sid += 1

model = NeuromorphicModel(enable_internal_states_tracking=False)
model.create_from_lists(somas=somas, synapses=synapses)
model.setup(use_gpu=True)
```

**Good for:** networks you already hold as data; building the whole model in one
call without an agent-by-agent loop; stable ids you control so synapses
reference somas unambiguously.

**Constraints:** single GPU only; one-shot (cannot be mixed with `create_*` or
modified after build); you are responsible for assigning unique ids.

---

## Choosing between the two

| | Method 1: Agent-by-agent | Method 2: Bulk lists |
|---|---|---|
| **Call style** | one `create_*` per agent | one `create_from_lists` call |
| **Ids** | auto-assigned, returned to you | you assign them |
| **Incremental / mix-and-modify** | yes | no (one-shot) |
| **Best when** | you build as you go, algorithmically | you already have the network as data |
| **Scope** | single GPU | single GPU |

For multi-GPU / multi-node runs, neither applies — use the file-based,
partition-aware loaders described in
[`PARTITION_LOADING.md`](PARTITION_LOADING.md).

---

## Configurations

Default parameter presets live in `superneuroabm/component_base_config.yaml`,
keyed by breed and config name.

- **Somas:** `lif_soma`, `izh_soma`, `hg_lif_soma` — each has a `config_0`.
- **Synapses:** `single_exp_synapse` (`config_0`), `weighted_synapse`
  (`config_0`, `config_1`).

Override any preset value per agent via `overrides` (Method 1 argument or the
`"overrides"` entry field in Method 2):

```python
overrides={
    "hyperparameters": {"weight": 14.0, "tau_fall": 10.0e-3},
    "internal_states": {"I_synapse": 0.0},
}
```

### Learning (STDP)

STDP is enabled per synapse with a **learning rule**, not a special synapse
config. Available rules (under `learning_rule:` in the YAML) include
`exp_pair_wise_stdp` and `three_bit_exp_pair_wise_stdp`, each with a `default`
config.

```python
# Method 1
model.create_synapse(..., learning_rule="exp_pair_wise_stdp",
                     learning_rule_config="default")

# Method 2
{"id": 101, "pre": 0, "post": 1, "learning_rule": "exp_pair_wise_stdp"}
```

Tune the rule with `overrides["learning_hyperparameters"]` (e.g. `a_exp_pre`,
`tau_pre_stdp`). A synapse with no `learning_rule` keeps a fixed weight.
