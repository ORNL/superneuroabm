# Preparing Your Data for SuperNeuroABM

This guide explains the **network definition file** that SuperNeuroABM reads, so you can turn
your own data — or a network you design by hand — into something the model can load. The
format is **SNN-native**: you describe *somas* (neurons) and *synapses* (connections)
directly. You do not need a graph or any graph library.

The single entry point is:

```python
from superneuroabm.model import NeuromorphicModel

model = NeuromorphicModel(enable_internal_state_tracking=False)
model.load_from_file("my_network.pkl")
model.setup()
model.simulate(ticks=100, update_data_ticks=1)
```

`load_from_file()` reads your file, computes every agent's parameters from breed/config
defaults (plus any overrides), wires up the connections, and bulk-builds the model. Your only
job is to produce a file in the structure below.

---

## 1. The mental model

A SuperNeuroABM network is just **two lists of agents**:

- **Somas** = neurons.
- **Synapses** = directed connections. Every synapse is itself an agent with its own ID,
  weight, delay, and (optional) learning rule. A synapse goes from a **pre** soma to a
  **post** soma; `pre = -1` means "external input" (the synapse you drive with `add_spike`).

---

## 2. Top-level structure

The file is a single `dict` with three keys (2 required, 1 optional):

```python
network = {
    "somas":        [ ... ],   # list of soma dicts (required)
    "synapses":     [ ... ],   # list of synapse dicts (required)
    "remote_ranks": { },       # {agent_id: rank} — only for multi-GPU; omit otherwise
}
```

For a normal single-GPU run, omit `remote_ranks` (or set it to `{}`).

---

## 3. Soma schema

```python
{
    "id":        42,            # REQUIRED. int. globally unique (see §5)
    "breed":     "lif_soma",    # optional. neuron model. default: "lif_soma"
    "config":    "config_0",    # optional. parameter set.  default: "config_0"
    "overrides": {},            # optional. per-neuron parameter tweaks (see §6)
    "metadata":  [],            # optional. list of string labels (see §7)
}
```

Only `id` is mandatory; everything else falls back to defaults.

---

## 4. Synapse schema

```python
{
    "id":                   100042,      # REQUIRED. int. unique, distinct from soma ids (§5)
    "pre":                  12,          # REQUIRED. pre-synaptic soma id, or -1 for external input
    "post":                 42,          # REQUIRED. post-synaptic soma id
    "breed":                "single_exp_synapse",  # optional. default: "single_exp_synapse"
    "config":               "config_0",  # optional. default: "config_0"
    "learning_rule":        None,         # optional. e.g. "exp_pair_wise_stdp" (None = static)
    "learning_rule_config": "default",    # optional. default: "default"
    "overrides":            {},           # optional (see §6)
    "metadata":             [],           # optional. list of string labels
}
```

`id`, `pre`, and `post` are required. **External input synapses** use `pre = -1`; these are
the synapses you later drive with `model.add_spike(id, tick, value)`. They are automatically
tagged with the `"input_synapse"` label, so you can find them via
`model.get_agents_with_label("input_synapse")`.

---

## 5. The ID rules (important)

- Somas **and** synapses share **one global integer ID space**.
- Every soma `id` and every synapse `id` must be **globally unique** — a synapse ID must
  never collide with a soma ID.
- Simplest safe scheme: number neurons `0..N-1`, then start synapse IDs at `N`
  (or `max_soma_id + 1`) and increment:

```python
syn_id = max(s["id"] for s in somas) + 1
for syn in synapses:
    syn["id"] = syn_id
    syn_id += 1
```

- `pre = -1` is a reserved sentinel for "external input", not a real neuron.

---

## 6. Overrides: customizing parameters per agent

Breeds/configs (§9) provide default parameter values. `overrides` changes them on a
**single** soma or synapse without defining a new config. It is **one dict, grouped by
category**:

```python
"overrides": {
    "hyperparameters":          {"weight": 25.0, "synaptic_delay": 2.0},  # synapse/soma params
    "internal_states":          {"v": -65.0},                             # synapse/soma state
    "learning_hyperparameters": {"a_exp_pre": 0.01},                      # learning-rule params
    "learning_internal_states": {"pre_trace": 0.1},                       # learning-rule state
}
```

- The four groups are **namespaced**, so synapse params and learning-rule params never
  collide even if they shared a name. Only list the values you want to change.
- Override keys must **exactly match** a parameter name of the chosen breed/config (an
  unknown key raises an error).
- The two `learning_*` groups are only valid when `learning_rule` is set. Somas only ever use
  `hyperparameters` and `internal_states`.

**Config-driven vs. runtime.** These four groups are the *only* things you set in the file.
Everything else (the synaptic-delay register, spike buffers, history buffers) is runtime
plumbing built by the engine. For example, you never set the delay register directly — set
`synaptic_delay` under `hyperparameters` and the engine sizes the register for you.

---

## 7. Metadata labels

`metadata` is a list of arbitrary string tags. After loading, retrieve all agents with a tag:

```python
topic_neurons = model.get_agents_with_label("topic")
test_papers   = model.get_agents_with_label("test")
```

Use it to mark roles your experiment cares about (`"topic"`, `"test"`, `"excitatory"`, …).
The label `"input_synapse"` is added automatically for `pre = -1` synapses.

> In multi-GPU runs, `get_agents_with_label` returns only the *local* subset (labels live on
> the rank that owns the agent). Drive global logic from your own side data.

---

## 8. `remote_ranks` (only for multi-GPU / MPI runs)

For a single GPU, omit `remote_ranks` (or set `{}`) and skip this section.

For distributed runs you write **one file per rank** (`partition_0.pkl`, …). Each rank's file
lists only its **local** somas, and `remote_ranks` maps every *off-rank* agent it references
to that agent's owning rank:

```python
"remote_ranks": {           # {agent_id: owner_rank}
    73:    1,   # a remote post-soma that a local synapse targets
    100073: 1,  # a remote synapse that a local post-soma reads across the partition boundary
}
```

It must include **both** remote post-somas that local synapses target **and** remote synapse
IDs that local somas read across a boundary.

---

## 9. Available breeds, configs, and learning rules

These ship in `superneuroabm/component_base_config.yaml`. Pass your own YAML via
`NeuromorphicModel(user_config=...)` to add more.

### Soma (neuron) breeds

| breed                   | configs            | key hyperparameters |
|-------------------------|--------------------|---------------------|
| `lif_soma`              | `config_0`         | `C, R, vthr, tref, vrest, vreset, I_in, scaling_factor` |
| `izh_soma`              | `config_0`, `config_1` | `k, vthr, C, a, b, vpeak, vrest, d, vreset, I_in` |
| `hg_lif_soma`           | `config_0`         | `T, tref, X, A, taus, taum, deltaa, I_in` |
| `lif_soma_adaptive_thr` | `config_0`         | `…, vthr_initial, delta_thr, tau_decay_thr` |

### Synapse breeds

| breed                 | configs            | key hyperparameters |
|-----------------------|--------------------|---------------------|
| `single_exp_synapse`  | `config_0`         | `weight, synaptic_delay, scale, tau_fall, tau_rise` |
| `weighted_synapse`    | `config_0`, `config_1` | `weight, synaptic_delay, scale` (`config_1` adds `tau_fall, tau_rise`) |

### Learning rules (set `learning_rule` on the synapse; `learning_rule_config = "default"`)

| `learning_rule`                 | description |
|---------------------------------|-------------|
| `exp_pair_wise_stdp`            | standard exponential pair-wise STDP |
| `three_bit_exp_pair_wise_stdp`  | 3-bit quantized STDP (`wmin, wmax, num_levels`) |
| `exp_pair_wise_stdp_bounded`    | bounded STDP (`wmin, wmax`) |
| `memristive_exp_pair_wise_stdp` | memristive STDP with write/read noise |

Leave `learning_rule = None` for a static (non-plastic) synapse. A learning rule's state
(`pre_trace, post_trace, dW`) is overridable via the `learning_internal_states` group.

---

## 10. File format

`load_from_file` auto-detects format from the extension. **`.pkl` (pickled dict) is the
canonical, recommended format** — save the dict exactly as shown above:

```python
import pickle
with open("my_network.pkl", "wb") as f:
    pickle.dump(network, f)
```

Legacy graph-centric files (`nodes`/`edges`/`source`/`target`) are **rejected with a clear
error** — regenerate them with an updated producer.

---

## 11. End-to-end conversion recipe

Turn a raw directed edge list into a loadable network:

```python
import pickle

# --- your raw data ---
raw_neurons = [0, 1, 2]                 # neuron ids
raw_edges   = [(0, 1), (0, 2), (1, 2)]  # (pre, post) pairs
external_inputs = [0]                   # neurons that receive an external spike

# --- somas ---
somas = [{"id": n, "breed": "lif_soma", "config": "config_0"} for n in raw_neurons]

# --- synapses (unique ids past the largest neuron id) ---
syn_id = max(raw_neurons) + 1
synapses = []
for pre, post in raw_edges:
    synapses.append({
        "id": syn_id, "pre": pre, "post": post,
        "breed": "single_exp_synapse", "config": "config_0",
        "overrides": {"hyperparameters": {"weight": 14.0}},
        "learning_rule": "exp_pair_wise_stdp",   # or None for static
    })
    syn_id += 1

# --- external input synapses (pre = -1) ---
input_synapse_ids = {}
for post in external_inputs:
    synapses.append({
        "id": syn_id, "pre": -1, "post": post,
        "breed": "single_exp_synapse", "config": "config_0",
        "overrides": {"hyperparameters": {"weight": 500.0}},
        "metadata": ["input_synapse"],
    })
    input_synapse_ids[post] = syn_id
    syn_id += 1

network = {"somas": somas, "synapses": synapses}   # remote_ranks omitted (single GPU)
with open("my_network.pkl", "wb") as f:
    pickle.dump(network, f)
```

Then load and run:

```python
model = NeuromorphicModel(enable_internal_state_tracking=False)
model.load_from_file("my_network.pkl")
model.setup()
model.add_spike(synapse_id=input_synapse_ids[0], tick=1, value=1.0)
model.simulate(ticks=50, update_data_ticks=1)
print(model.get_spike_times(soma_id=2))
```

> Keep any experiment side data (ground-truth labels, id↔name maps, which input synapse feeds
> which neuron) in a **separate** file — the network file is only about model structure. See
> `superneuroabm_sgnn/build_network_from_data.py` for a real example that emits a network
> `.pkl` plus an `aux.pkl` of labels.

---

## 12. Validation checklist

- [ ] Every soma has a unique `id`; every synapse has a unique `id`.
- [ ] No synapse `id` collides with any soma `id` (shared ID space).
- [ ] Every synapse has `id`, `pre`, and `post`.
- [ ] External input synapses use `pre = -1` (and, ideally, `metadata: ["input_synapse"]`).
- [ ] `breed`/`config` names exist in the config YAML; `overrides` keys match that
      breed/config's parameter names, grouped under `hyperparameters` / `internal_states` /
      `learning_hyperparameters` / `learning_internal_states`.
- [ ] `remote_ranks` omitted/`{}` for single-GPU; populated per-rank for multi-GPU (§8).
