# Custom Somas, Synapses, and Learning Rules

SuperNeuroABM ships with a handful of neuron models, synapse models, and STDP
variants, but none of them are privileged. Every built-in component is just a
CUDA device function plus a block of YAML — and so is yours. You write a
`@jit.rawkernel(device="cuda")` function, declare its parameters in a config
file, register it before `setup()`, and it runs at full speed alongside the
built-ins, in the same fused kernel, on one GPU or across many.

There is no plugin system, no subclassing, no C++ to write, and no fork of the
library to maintain. Custom components live in *your* files, next to your
experiment.

| You want to change… | Write a… | Register with |
|---|---|---|
| Neuron dynamics (membrane, threshold, reset, adaptation) | soma step function | `model.register_soma_type(...)` |
| Synaptic current dynamics (kernel shape, delay, device physics) | synapse step function | `model.register_synapse_type(...)` |
| How weights change (plasticity) | learning rule step function | `model.register_learning_rule(...)` |

All three are the same kind of object: a device function with one fixed
signature. A soma, a synapse and a learning rule differ only in *which* slots
of the shared property tensors they read and write, and *when* in the tick they
are scheduled.

**Related docs:** [`FUNCTIONALITY_GUIDE.md`](FUNCTIONALITY_GUIDE.md) (concepts and
built-ins) · [`SINGLE_GPU_NETWORK_CONSTRUCTION.md`](SINGLE_GPU_NETWORK_CONSTRUCTION.md)
(building networks) · [`CPU_GPU_DATA_FLOW.md`](CPU_GPU_DATA_FLOW.md) (when reads and
writes are valid).

---

## 1. The mental model

An agent (soma or synapse) is a **breed** plus five flat `float` vectors:

| Vector | Filled from YAML | Meaning |
|---|---|---|
| `hyperparameters` | `hyperparameters:` block | Constants of the model (τ, threshold, weight…) |
| `internal_states` | `internal_states:` block | Mutable state the kernel evolves (v, I_synapse…) |
| `learning_hyperparameters` | learning rule's `learning_hyperparameters:` | Plasticity constants |
| `learning_internal_states` | learning rule's `learning_internal_states:` | Plasticity state (traces, dW) |
| buffers (`*_buffer`) | — | Per-tick history, when tracking is enabled |

Inside the kernel these arrive as 2-D tensors indexed by `agent_index`, and
**the element order is the key order of your YAML block**. `hyperparameters[agent_index][2]`
is the third key you wrote in the config. That is the entire contract between
your config and your kernel: position.

Because the mapping is positional, the framework gives you a name-based API on
the host side (`get_hyperparameters` / `set_hyperparameters`) that resolves names
through the same YAML key order — it works for custom breeds automatically, with
no registration of names anywhere.

---

## 2. The step function signature

Every step function — soma, synapse, or learning rule — takes exactly these 16
parameters, in this order:

```python
import cupy as cp
from cupyx import jit


@jit.rawkernel(device="cuda")
def my_step_func(
    tick,                              # current tick (int-like scalar)
    agent_index,                       # this agent's local index == GPU thread
    dt,                                # global: timestep, seconds
    I_bias,                            # global: bias current
    agent_ids,                         # local index -> global agent id
    breeds,                            # local index -> breed index
    locations,                         # connectivity (see below)
    neuron_params,                     # == hyperparameters   (synapse_params for synapses)
    learning_params,                   # == learning_hyperparameters
    internal_states,
    learning_internal_states,
    synapse_history,                   # delay register
    input_spikes_tensor,               # externally injected spikes (synapses only)
    output_spikes_tensor,              # soma spike output ring buffer
    internal_states_buffer,            # per-tick history
    learning_internal_states_buffer,   # per-tick history
):
    ...
```

The parameter *names* are yours to choose (built-ins call slot 8
`neuron_params` in somas and `synapse_params` in synapses — same tensor), but
the *order* is fixed.

`locations` is the connectivity view, and it means different things by role:

- **soma**: `locations[agent_index]` is the list of incoming synapse indices.
- **synapse**: `locations[agent_index] == [pre_soma_index, post_soma_index]`,
  where `-1` in the pre slot means "external input, read `input_spikes_tensor`".

These are already local indices — SAGESim converts agent ids to indices before
launch, so there is never a search to do.

### What the framework does to your source

Your function is not called as written. At `setup()` SAGESim reads its source
(`inspect.getsource`), rewrites it, and fuses it into one big kernel
(`step_func_code.py`, written into the current working directory — useful to
read when debugging). Three rewrites matter to you:

1. **`locations` → CSR.** It is replaced by `neighbor_offsets` / `neighbor_values`
   and your access patterns are rewritten with it. Stick to the patterns the
   built-ins use: `for i in range(len(locations[agent_index]))` and
   `locations[agent_index][i]`.
2. **`_seed` and `logical_ids` are injected** into the signature. You never
   declare them.
3. **`rand_*` calls get the seed prepended.** Call
   `rand_normal(tick, agent_index, salt)` with three arguments in your source;
   codegen turns it into the seeded, rank-agnostic 4-argument form. `salt` is a
   unique small integer per call site.

Because the module is re-imported by *stem* in the generated file
(`from my_module import *`), pass `step_func_path` when registering and give
your files distinct basenames.

### What you may write inside a kernel

This is CUDA device code compiled by `cupyx.jit`, not Python:

- Scalar float/int math, `if` / `elif` / `else`, `for i in range(...)`, `while`.
- `cp.exp`, `cp.isnan`, and other elementwise CuPy scalar math.
- Calls to other `@jit.rawkernel(device="cuda")` helpers, e.g.
  `superneuroabm.step_functions.synapse.util.get_soma_spike` and
  `sagesim.math_utils` (`clamp`, `lerp`, `safe_divide`, `kronecker`,
  `rand_uniform_philox`, `rand_normal`, `rand_normal_bounded`).
- Indexing into the tensors you were passed.

Not available: Python objects, lists/dicts, NumPy, allocation, exceptions,
`print`, closures over globals that are not compile-time constants.

---

## 3. Pinned slots — the contracts you must honor

Most of the layout is yours to define. A few positions are read by *other*
agents' kernels, which cannot consult your config, so they are fixed. Two are
validated at model-build time; the rest are conventions that fail silently if
broken. Get these right and everything else is free.

| Contract | Who depends on it | Enforced? |
|---|---|---|
| Synapse `hyperparameters[0] = weight` | every synapse and learning-rule kernel | ✅ `ValueError` at `create_synapse()` |
| Synapse `hyperparameters[1] = synaptic_delay` | delay register sizing | ✅ `ValueError` at `create_synapse()` |
| `learning_hyperparameters[0] = stdp_type` | generated rule selector | ✅ `ValueError` at `create_synapse()` |
| Synapse `internal_states[0] = output current` | post-soma sums `internal_states[syn][0]` | ❌ silent |
| Soma writes `output_spikes_tensor[agent_index][tick % 2] = s` | synapses and STDP read spikes | ❌ silent |
| Buffer writes use `tick % len(buffer[agent_index])` | tracking-disabled mode uses length-1 buffers | ❌ out-of-bounds |

The soma spike ring has **two** slots and readers take `(tick - 1) % 2`, which
is where the built-in one-tick synaptic delay comes from.

So the minimum viable custom soma ends with:

```python
    internal_states[agent_index][0] = v
    output_spikes_tensor[agent_index][t_current % 2] = s

    buffer_idx = t_current % len(internal_states_buffer[agent_index])
    internal_states_buffer[agent_index][buffer_idx][0] = v
```

and the minimum viable custom synapse ends with:

```python
    internal_states[agent_index][0] = I_synapse   # what the post-soma will read
```

---

## 4. Recipe: a custom soma

**Step 1 — write the kernel** (`my_somas.py`, anywhere in your project):

```python
import cupy as cp
from cupyx import jit


@jit.rawkernel(device="cuda")
def exp_lif_soma_step_func(
    tick, agent_index, dt, I_bias, agent_ids, breeds, locations,
    neuron_params, learning_params, internal_states, learning_internal_states,
    synapse_history, input_spikes_tensor, output_spikes_tensor,
    internal_states_buffer, learning_internal_states_buffer,
):
    # Sum synaptic input: each incoming synapse publishes current at slot 0
    synapse_indices = locations[agent_index]
    I_synapse = 0.0
    for i in range(len(synapse_indices)):
        synapse_index = int(synapse_indices[i])
        if synapse_index >= 0 and not cp.isnan(synapse_indices[i]):
            I_synapse += internal_states[synapse_index][0]

    t_current = int(tick)

    # Hyperparameters — order matches the YAML block below
    tau_m = neuron_params[agent_index][0]
    vthr = neuron_params[agent_index][1]
    vrest = neuron_params[agent_index][2]
    vreset = neuron_params[agent_index][3]
    tref = neuron_params[agent_index][4]
    I_in = neuron_params[agent_index][5]
    scaling_factor = neuron_params[agent_index][6]

    v = internal_states[agent_index][0]
    tcount = internal_states[agent_index][1]
    tlast = internal_states[agent_index][2]

    # Exact exponential update instead of Euler
    decay = cp.exp(-dt / tau_m)
    I_total = I_synapse * scaling_factor + I_in

    in_refractory = (tlast > 0) and (dt * tcount <= tlast + tref)
    if not in_refractory:
        v = vrest + (v - vrest) * decay + I_total * dt / tau_m

    s = 1.0 * ((v >= vthr) and not in_refractory)
    tlast = tlast * (1 - s) + dt * tcount * s
    v = v * (1 - s) + vreset * s

    internal_states[agent_index][0] = v
    internal_states[agent_index][1] = tcount + 1
    internal_states[agent_index][2] = tlast

    output_spikes_tensor[agent_index][t_current % 2] = s

    buffer_idx = t_current % len(internal_states_buffer[agent_index])
    internal_states_buffer[agent_index][buffer_idx][0] = v
    internal_states_buffer[agent_index][buffer_idx][1] = tcount + 1
    internal_states_buffer[agent_index][buffer_idx][2] = tlast
```

**Step 2 — declare its parameters** in your config YAML, under the breed name
you will register. Key order *is* the index order the kernel reads:

```yaml
soma:
  exp_lif_soma:
    config_0:
      hyperparameters:
        tau_m: 20e-3
        vthr: -55.0
        vrest: -65.0
        vreset: -65.0
        tref: 2e-3
        I_in: 0.0
        scaling_factor: 1.0
      internal_states:
        v: -65.0
        tcount: 0.0
        tlast: 0.0
```

**Step 3 — register and use it:**

```python
from pathlib import Path
from superneuroabm.model import NeuromorphicModel
from my_somas import exp_lif_soma_step_func

model = NeuromorphicModel(user_config=Path("my_config.yaml"))

model.register_soma_type(                       # before setup()
    name="exp_lif_soma",
    step_func=exp_lif_soma_step_func,
    step_func_path=Path("my_somas.py"),
)

soma = model.create_soma(breed="exp_lif_soma", config_name="config_0")
```

Multiple configs per breed are just multiple YAML blocks — one kernel, many
parameterizations, chosen per agent at `create_soma()`. Custom and built-in
breeds mix freely in the same network.

---

## 5. Recipe: a custom synapse

Identical shape, with `register_synapse_type()` and a `synapse:` config block.
Two rules apply: `weight` and `synaptic_delay` must be the first two keys, and
the kernel must publish its output current to `internal_states[agent_index][0]`.

Reading the presynaptic spike is one helper call, and it transparently handles
the external-input case (`pre_soma_index == -1`):

```python
from superneuroabm.step_functions.synapse.util import get_soma_spike

    pre_soma_index = locations[agent_index][0]
    post_soma_index = locations[agent_index][1]

    spike = get_soma_spike(
        tick, agent_index, dt, I_bias, agent_ids,
        pre_soma_index, t_current,
        input_spikes_tensor, output_spikes_tensor,
    )

    # e.g. alpha-function (second-order) kernel
    h = h * (1 - dt / tau) + spike * weight * scale / tau
    I_synapse = I_synapse * (1 - dt / tau) + h * dt

    internal_states[agent_index][0] = I_synapse   # pinned: post-soma reads this
    internal_states[agent_index][1] = h
```

```yaml
synapse:
  alpha_synapse:
    config_0:
      hyperparameters:
        weight: 300.0          # must be key #0
        synaptic_delay: 1.0    # must be key #1
        scale: 1.0
        tau: 10e-3
      internal_states:
        I_synapse: 0.0         # read by the post-soma
        h: 0.0
```

```python
model.register_synapse_type(
    name="alpha_synapse",
    step_func=alpha_synapse_step_func,
    step_func_path=Path("my_synapses.py"),
)
```

---

## 6. Recipe: a custom learning rule

Plasticity is deliberately **decoupled from synapse type**: a synapse's breed
owns its current dynamics, and its learning rule owns weight change. Any
registered rule can be attached to any synapse breed.

Dispatch works through one integer. Each synapse carries `stdp_type` in
`learning_hyperparameters[0]`; at `setup()` the framework generates a selector
kernel (`superneuroabm/_generated/learning_rule_selector.py`) that is an
`if/elif` over every registered rule id, and attaches it to every synapse breed
one priority step after the synapse itself. `stdp_type == -1` means "no
learning" and costs a comparison.

**Step 1 — write the rule.** It reads the pre/post spikes, updates traces, and
writes `synapse_params[agent_index][0]`:

```python
import cupy as cp
from cupyx import jit
from sagesim.math_utils import clamp
from superneuroabm.step_functions.synapse.util import get_soma_spike


@jit.rawkernel(device="cuda")
def bounded_stdp(
    tick, agent_index, dt, I_bias, agent_ids, breeds, locations,
    synapse_params, learning_params, internal_states, learning_internal_states,
    synapse_history, input_spikes_tensor, output_spikes_tensor,
    internal_states_buffer, learning_internal_states_buffer,
):
    t_current = int(tick)

    weight = synapse_params[agent_index][0]

    # learning_params[...][0] is stdp_type — your parameters start at 1
    tau_pre = learning_params[agent_index][1]
    tau_post = learning_params[agent_index][2]
    a_pre = learning_params[agent_index][3]
    a_post = learning_params[agent_index][4]
    wmin = learning_params[agent_index][6]
    wmax = learning_params[agent_index][7]

    pre_trace = learning_internal_states[agent_index][0]
    post_trace = learning_internal_states[agent_index][1]

    pre_soma_index = locations[agent_index][0]
    post_soma_index = locations[agent_index][1]

    pre_spike = get_soma_spike(
        tick, agent_index, dt, I_bias, agent_ids,
        pre_soma_index, t_current, input_spikes_tensor, output_spikes_tensor,
    )
    post_spike = get_soma_spike(
        tick, agent_index, dt, I_bias, agent_ids,
        post_soma_index, t_current, input_spikes_tensor, output_spikes_tensor,
    )

    pre_trace = pre_trace * (1 - dt / tau_pre) + pre_spike * a_pre
    post_trace = post_trace * (1 - dt / tau_post) + post_spike * a_post

    # Multiplicative bounds: change shrinks as the weight nears its limit
    dW = (pre_trace * post_spike * (wmax - weight)
          - post_trace * pre_spike * (weight - wmin))

    synapse_params[agent_index][0] = clamp(weight + dW, wmin, wmax)

    learning_internal_states[agent_index][0] = pre_trace
    learning_internal_states[agent_index][1] = post_trace
    learning_internal_states[agent_index][2] = dW

    buffer_idx = t_current % len(learning_internal_states_buffer[agent_index])
    learning_internal_states_buffer[agent_index][buffer_idx][0] = pre_trace
    learning_internal_states_buffer[agent_index][buffer_idx][1] = post_trace
    learning_internal_states_buffer[agent_index][buffer_idx][2] = dW
```

**Step 2 — register it.** The id is assigned at runtime, not by you. The four
built-in rules hold 0–3, so the first custom rule is 4:

```python
RULE_ID = model.register_learning_rule(
    step_func=bounded_stdp,
    step_func_path=Path("my_rules.py"),
)
```

**Step 3 — declare its hyperparameters** in a `learning_rule:` block, with
`stdp_type` first:

```yaml
learning_rule:
  bounded_stdp:
    default:
      learning_hyperparameters:
        stdp_type: 0.0        # placeholder; overridden with the runtime id
        tau_pre_stdp: 10e-3
        tau_post_stdp: 10e-3
        a_exp_pre: 0.005
        a_exp_post: 0.005
        stdp_history_length: 100
        wmin: 0.0
        wmax: 24.0
      learning_internal_states:
        pre_trace: 0.0
        post_trace: 0.0
        dW: 0.0
```

**Step 4 — attach it at synapse creation.** Two independent choices, and this
is the part that trips people up:

- `learning_rule=` picks the **YAML block** that supplies the learning
  hyperparameters and their names.
- `stdp_type` picks the **kernel that actually runs**.

```python
syn = model.create_synapse(
    breed="single_exp_synapse",           # current dynamics
    pre_soma_id=pre, post_soma_id=post,
    config_name="config_0",
    learning_rule="bounded_stdp",         # which YAML block
    learning_rule_config="default",
    overrides={"learning_hyperparameters": {"stdp_type": float(RULE_ID)}},
)
```

Because ids are assigned in registration order, always write the id back with
an override rather than hard-coding it in YAML. That also lets you reuse a
tuned block from someone else's config while running your own kernel — the
digits tutorial does exactly this, borrowing `exp_pair_wise_stdp_bounded`'s
tuned constants for a custom rule without editing the shared config.

Verify the override landed before you spend a long run on it:

```python
print(model.get_learning_hyperparameters(syn)["stdp_type"])   # must be RULE_ID
```

Always pass `learning_rule=` even when overriding `stdp_type` — `eval()` /
`train()` freeze and restore plasticity by looking at synapses *created with* a
rule, and a synapse with `learning_rule=None` is invisible to them.

---

## 7. Configuration files

`NeuromorphicModel(user_config=...)` **replaces** the packaged
`component_base_config.yaml` — it is not merged. Your file must therefore
declare every breed and config you use, including built-ins you still rely on
(e.g. `single_exp_synapse`, or a stock STDP block). See
`examples/custom_components_config.yaml` for a file that mixes both.

Two consequences of how the file is loaded:

- **All values become floats.** Booleans and integers are numeric flags
  (`tref_allows_integration: 1`), and the kernel compares them as floats.
- **Vector width is global.** Each property vector is sized to the widest block
  of that type anywhere in the file, and shorter ones are zero-padded. A custom
  soma with 20 hyperparameters widens the tensor for every agent, so keep
  parameter lists tight in large models.

Reordering a block silently changes which parameter your kernel reads. The
pinned slots (§3) are validated, and the name-based API is there so host code
never has to hard-code positions:

```python
model.get_hyperparameters(soma)                     # {'tau_m': 0.02, 'vthr': -55.0, ...}
model.set_hyperparameters(soma, {"vthr": -52.0})    # partial update, by name
```

---

## 8. Execution order

Within a tick, breeds run by priority, in sequence:

| Priority | What runs |
|---|---|
| 0 | all somas (custom ones included) |
| 100 | all synapses |
| 101 | the generated learning-rule selector, per synapse |

Registration slots your component into the same schedule as the built-ins — a
custom soma is not "after" the built-in somas, it is *with* them. Because
properties are not double-buffered, a write in the soma phase is visible to the
synapse phase in the *same* tick; the only deliberate delay is the spike ring
buffer's one-tick read.

Across MPI ranks, exactly two properties cross the boundary: a soma's
`output_spikes_tensor` and a synapse's `internal_states`. Custom breeds inherit
that property set unchanged, so a custom component that needs to publish
something to another agent must route it through those — a soma through its
spike output, a synapse through its current slot.

---

## 9. Rules, limits, and common errors

**Registration must happen before `setup()`.** `setup()` is what generates the
selector and compiles the fused kernel; registering after it raises
`RuntimeError`.

**Names must be unique.** Breed names collide by name, learning rules by the
step function's `__name__` — so a rule named `exp_pair_wise_stdp_bounded` is
rejected.

| Symptom | Cause |
|---|---|
| `RuntimeError: Cannot register ... after setup()` | register before `model.setup()` |
| `ValueError: 'weight' must be key #0 ...` | synapse config key order; `weight`, then `synaptic_delay` |
| `KeyError` on breed or config name at `create_soma()` | `user_config` YAML has no block for that breed/config |
| Kernel runs, weights never change | `stdp_type` still `-1` or pointing at another rule — check `get_learning_hyperparameters` |
| Post-soma sees no input | custom synapse didn't write `internal_states[agent_index][0]` |
| Downstream sees no spikes | custom soma didn't write `output_spikes_tensor[agent_index][tick % 2]` |
| `ImportError` in generated `step_func_code.py` | `step_func_path` wrong, or two step-function files share a basename |
| Parameters read as garbage | YAML key order changed under a kernel that indexes positionally |
| `RuntimeError: ... GPU buffers hold unsynced state` | call `reset(retain_parameters=True)` before host-side writes (see [`CPU_GPU_DATA_FLOW.md`](CPU_GPU_DATA_FLOW.md)) |

**Replacing the defaults entirely.** Instead of registering additively, the
constructor takes whole registries — useful when you want *only* your models
present:

```python
model = NeuromorphicModel(
    soma_breed_info={"my_soma": (my_soma_step_func, Path("my_somas.py"))},
    synapse_breed_info={"my_syn": (my_syn_step_func, Path("my_synapses.py"))},
)
```

---

## 10. Debugging a new component

1. **Start with one synapse and two somas.** Inject a handful of spikes with
   `add_spike()`, run ~100 ticks, print `get_spike_times()`.
2. **Read the generated code.** `step_func_code.py` in the working directory is
   the fused kernel as compiled — your function appears in it post-rewrite, which
   settles any question about what CSR transformation or seed injection did.
   `superneuroabm/_generated/learning_rule_selector.py` shows the exact dispatch
   table, so you can confirm your rule got an id and a branch.
3. **Turn on tracking** (`enable_internal_states_tracking=True`) and inspect
   `get_internal_states_history(agent_id)` / `get_learning_internal_states_history(agent_id)`
   tick by tick. This is where a trace that decays wrong or a `dW` that stays
   zero shows itself immediately.
4. **Check parameters by name** before blaming the kernel:
   `get_hyperparameters(id)` and `get_learning_hyperparameters(id)` print exactly
   what the GPU will see, in your names.
5. **Scale up last.** Kernel bugs are cheapest to find on 3 agents.

---

## 11. Working examples in this repository

| Path | What it shows |
|---|---|
| `examples/user_step_functions/exp_lif_soma.py` | Custom soma, exponential-Euler LIF |
| `examples/user_step_functions/alpha_synapse.py` | Custom synapse, second-order alpha kernel |
| `examples/user_step_functions/bounded_stdp.py` | Custom rule, multiplicative weight bounds |
| `examples/user_step_functions/symmetric_stdp.py` | Custom rule, correlation-based |
| `examples/custom_components_config.yaml` | A user config declaring custom *and* built-in blocks |
| `examples/register_custom_network.py` | All of the above registered and run end to end |
| `tutorials/user_customized_stdp.py` | SuperNeuroMAT-style rule used by tutorial 01 |
| `tutorials/user_customized_lif.py` | Custom soma registered from a notebook; fixes the `-inf` `tlast` reset |
| `tutorials/01_superneuroabm_digits.ipynb` | Custom soma *and* custom rule, in a real task |
| `superneuroabm/step_functions/` | The built-ins — the best reference for kernel style |
| `tests/test_registration_api.py` | Expected behavior of the registration API, including errors |
