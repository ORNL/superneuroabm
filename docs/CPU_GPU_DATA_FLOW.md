# CPU/GPU Data Flow

How model state moves between the host and the device, which operation touches which copy,
and the one ordering rule that follows from it.

This is the *how it works* reference. `CPU_GPU_SYNC_DESIGN_NOTES.md` is the companion
*why it is designed this way and what should change* document.

---

## 1. There are two copies of every property

| | Host (CPU) | Device (GPU) |
|---|---|---|
| Where | `AgentFactory._property_name_2_agent_data_tensor` | `GPUBufferManager.property_tensors` |
| Shape | dict of ragged Python lists, one row per local agent | list of 2-D CuPy arrays, one per property |
| Type | Python floats (float64) | **float32**, NaN-padded to the widest row |
| Who writes it | `create_*`, `reset()`, property setters, `add_spike*` | step functions, every tick |

Two consequences that surprise people:

- **Round trips quantize.** A value written as `1e-9` reads back as `9.9999997e-10` once it
  has been through the device. Cosmetic, but do not assert exact equality on it.
- **Padding.** After a device→host sync, every row is the *global* max width for that
  property with a NaN tail, not the breed's natural width. Indexing named positions is
  fine; `len()` on a row is not meaningful.

There is also an aliasing subtlety. `_generate_agent_data_tensors()` hands out *references*
to the host lists (`sagesim/agent.py:350-354`), so an in-place host write is immediately
visible model-side. But `_sync_gpu_to_agent_factory()` **replaces** those list objects
(`sagesim/model.py:1078`), breaking the aliasing — which is why `NeuromorphicModel.reset()`
calls `_regenerate_data_tensors()` again afterwards.

## 2. What `is_initialized` means — and what it does not

`GPUBufferManager.is_initialized` (`sagesim/gpu_kernels.py:524-528`) answers exactly one
question: **do device buffers currently exist, built from the host?**

- `_build_gpu_buffers()` sets it `True` (`sagesim/model.py:1414`) after uploading.
- Property setters set it `False` (`:553`, `:590`) so the next tick rebuilds from the host.
- `reset()` frees the buffers and installs a fresh manager, so it starts `False`.

It does **not** answer "which copy is authoritative?" After a kernel run and after a fresh
upload it is `True` in both cases, but only the first has device-newer data. Conflating
those two facts is the root of the hazard in §4.

There is no flag today that tracks device-newer-ness. §5 records the design for one.

## 3. Operation-by-operation

| Operation | Host | Device | `is_initialized` | Afterwards you may assume |
|---|---|---|---|---|
| `create_soma` / `create_synapse` | writes | — | — | host is the only copy |
| `setup()` | may reset agents | fresh empty manager | → `False` | codegen + JIT done; buffers not built yet |
| first `simulate()` tick | read (upload source) | **built, then written by kernel** | → `True` | **device is authoritative** |
| later `simulate()` ticks | — | written by kernel | `True` | device is authoritative |
| `get_agent_property_value` | read *only if* buffers are down | read when buffers are live | unchanged | returns fresh values either way |
| `set_agent_property_value` | **writes** | — | → `False` | host authoritative; **see §4** |
| `add_spike` / `add_spike_list` | **writes** (via the setter) | — | → `False` | same as above |
| `eval()` / `train()` | **writes** (`_write_stdp_type`, direct) | — | → `False` | same as above |
| `reset(retain_parameters=…)` | **overwritten by sync**, then agents reset | freed | → `False` | **host authoritative and current** |
| `get_spike_times` | reads the gathered spike log | reads `spike_record` | unchanged | independent of the property path |

The single most important row: **`reset()` is the only operation that copies device→host.**
`Model.reset()` calls `_sync_gpu_to_agent_factory()` (`sagesim/model.py:1065`, invoked at
`:1092`) before freeing the buffers. Nothing else does. (`_download_local_data_to_cpu` at
`:1418` exists but has no callers.)

## 4. The ordering rule

Because reads take the device path while writes take the host path, the two disagree
exactly when the kernel has run and nothing has synced:

```
simulate()                    device: weights 0…0.64   host: weights 0.0  (as created)
set_agent_property_value(...) → writes the stale host row, marks device for rebuild
simulate()                    → uploads the host row: your write ✓, learned weights ✗
```

No exception, no warning — just weights silently back at their initial values. So:

> **`simulate()` → `reset(retain_parameters=True)` → `eval()` → parameter writes → `simulate()`**

`retain_parameters=True` is what makes this safe to do mid-experiment: `_reset_agents`
(`superneuroabm/model.py:647-690`) clears `internal_states`, `input_spikes_tensor`,
`synapse_delay_reg` and `learning_internal_states`, but deliberately leaves
`hyperparameters` and `learning_hyperparameters` alone. Learned weights and any parameter
overrides both survive. `retain_parameters=False` restores them from config instead, which
discards both.

**You get told, not bitten.** `set_hyperparameters` / `set_learning_hyperparameters`
(`superneuroabm/model.py`) refuse to run while buffers are live and name the fix in the
error. The raw `set_agent_property_value` remains unguarded as an escape hatch.
`tests/test_learning_mode.py::test_raw_setter_after_simulate_loses_learned_weights` pins
the hazard as an `expectedFailure`, so it will report *unexpected success* the moment §5
lands.

### Recognising the failure

Weights sitting at exactly their creation values after a run that should have trained;
an inference pass scoring at chance; a parameter you set having taken effect while
everything the kernel learned did not.

## 5. Planned fix: a `_device_dirty` flag

Not implemented. Recorded here so the later pass does not have to re-derive it.

Add a flag next to `is_initialized`, meaning **"the device holds values the host has not
seen"** — the fact `is_initialized` cannot express. It is the same shape as the existing
`_globals_dirty` (`sagesim/model.py:728, 736, 1652-1654`): set by the producer, checked at
the point of use.

| Event | `is_initialized` | `_device_dirty` | Why |
|---|---|---|---|
| `setup()` / `reset()` → fresh manager | `False` | `False` | nothing on the device yet |
| `_build_gpu_buffers()` (`:1414`) | → `True` | unchanged `False` | **upload** — device mirrors host |
| kernel write-back, end of `simulate()` (`:~1718`) | `True` | → **`True`** | step funcs mutated device tensors. **Only producer** |
| `_sync_gpu_to_agent_factory()` (`:1065`) | unchanged | → **`False`** | **download** — host mirrors device |
| `set_agent_property_value` (after the check) | → `False` | `False` | host write; the *device* is stale, which `is_initialized` tracks |
| `_write_stdp_type` (after the check) | → `False` | `False` | same, batched |
| `_regenerate_data_tensors()` | unchanged | unchanged | host-side rebuild, no device involvement |

Consumers call one guard before touching the host:

```python
def _ensure_host_current(self):
    if self._buffers_live() and self._gpu_buffers._device_dirty:
        self._sync_gpu_to_agent_factory()      # clears the flag
```

Call sites:
- **SAGESim** — `set_agent_property_value` (`:547`), `set_local_agent_property_value`
  (`:581`); and the host branches of the two getters (`:541`, `:573`) as belt-and-braces.
- **superneuroabm** — `_write_stdp_type` (`model.py:722-753`) and `_read_stdp_type`
  (`:695-720`) **bypass the setter** and poke
  `af._property_name_2_agent_data_tensor` directly. Fixing SAGESim alone would leave
  `eval()` order-dependent. Put the check **before** `_write_stdp_type`'s batch loop, not
  inside it, so `eval()` on a large model pays one sync and then a tight loop.
- **Not** `_reset_agents` (`:661`, `:763`) — safe by construction, since `reset()` calls
  `super().reset()` (which syncs) first.

Two mistakes that are easy to make:
- A **setter must not** set `_device_dirty`. It makes the *device* stale, not the host;
  marking it would make a later sync overwrite the user's write with old device data.
- **Buffer construction must not** set it either — that direction is an upload.

`is_initialized == False and _device_dirty == True` should be unreachable, since `reset()`
syncs before freeing and setters sync before invalidating. Worth an assert.

Cost: one device→host copy of all properties at the first host access after each
`simulate()` — the same copy `reset()` already performs, just triggered by need instead of
by convention. Everything else becomes free. Once this lands, the guard in
`set_hyperparameters` is deleted rather than reworked.

## 6. Known issue: `simulate()` writes the host store without invalidating the device

`superneuroabm/model.py:786-814` reaches into `af._property_name_2_agent_data_tensor`
directly and does two things:

- **(a)** when `enable_internal_states_tracking` is on, resizes `internal_states_buffer`
  and `learning_internal_states_buffer` to `ticks` rows (`:787-793`);
- **(b)** sorts each synapse's `input_spikes_tensor` in place by tick — in **both**
  branches, tracked or not (`:795-804`, `:806-814`).

Neither sets `is_initialized = False`. So on a second `simulate()` while buffers are still
live, the device keeps the old buffer shape and the unsorted spike tensor, and the
host-side work is silently discarded.

It is masked today because `add_spike` / `add_spike_list` go through the property setter
and invalidate the buffers, so the usual inject-then-simulate flow rebuilds anyway. The
exposed path is **back-to-back `simulate()` calls with no intervening property write**.

The fix needs *both* directions handled, which is why it waits for §5:

1. `_ensure_host_current()` **before** reading `data["internal_states"][idx]` to seed the
   buffer — after the first `simulate()` that host row is stale, so the buffer would
   otherwise be seeded with pre-simulation state.
2. `is_initialized = False` **after** the writes, so the device picks them up.

Related: `weighted_synapse.py:60-63` indexes `internal_states_buffer[agent_index][t_current]`
without the `% len(...)` guard every other step function uses, so with tracking disabled
(buffer length 1) it reads out of bounds for any tick > 0.
