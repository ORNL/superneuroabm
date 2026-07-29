# Spike Recording: Mask Build Cost and Staleness Window

Investigation notes for deferred work on `NeuromorphicModel._prepare_kernel_extras`
(`superneuroabm/model.py`). Nothing here has been implemented — this records what was measured and
what the fix looks like, so it can be picked up without redoing the analysis.

**Why it was deferred:** the change touches the hot path that the in-flight weak-scaling sweep is
measuring. It shifts `_simulation_time`, so it should land between sweeps, not during one.

---

## Background: how spike recording works

There is no per-soma-per-timestep spike trace. `output_spikes_tensor` is a 2-element ping-pong ring
buffer, written by every soma step func as `output_spikes_tensor[agent_index][t_current % 2] = s`
(e.g. `step_functions/soma/lif.py:90`). Counting spikes from it is not possible.

Instead, `_get_extra_kernel_config` (`model.py:1712`) injects post-step CUDA that appends to a GPU-side
atomic log, emitted only at soma priority 0:

```python
'_sv = a{prop_idx}[_real_idx][thread_local_tick % 2]',
'if _sv > 0.0 and spike_mask[_real_idx] > 0.0:',
'\t_slot = jit.atomic_add(spike_record_count, 0, 1)',
'\tspike_record[_slot * 2] = agent_ids[_real_idx]',
'\tspike_record[_slot * 2 + 1] = float(thread_local_tick)',
```

`spike_mask` is a float32 bitmask over agent slots, built host-side by `_prepare_kernel_extras`
(`model.py:1739-1753`) and cached in `self._spike_mask_gpu`. `set_recorded_somas(soma_ids)` selects
which somas it marks; if never called, `_recorded_soma_ids is None` means record-all.

> `set_recorded_somas([])` is **not** "record nothing special" — an empty list builds an all-zero mask
> and suppresses every soma. `None` is the record-all sentinel (`model.py:1742`).

---

## Finding 1 — the mask build is pathological at scale

The current build does `cp.zeros(...)` followed by **one scalar device write per soma**:

```python
mask = cp.zeros(buf.agent_capacity, dtype=cp.float32)
for sid in ids:
    idx = buf.agent_id_to_index.get(sid, -1)
    if 0 <= idx < num_local_agents:
        mask[idx] = 1.0          # <-- one H2D transfer each
```

Each `mask[idx] = 1.0` is a separate tiny host-to-device transfer. Measured on the project GPU:

| build | digits net (788 agents) | weak-scaling net (400k somas / 4.84M agents) |
|---|---|---|
| current (`cp.zeros` + per-soma scalar writes) | 190–746 µs | **4.58 s** (single GPU, 400k local) |
| proposed (host numpy + one H2D copy) | 27–32 µs | **0.03 s** |

Per rank at 80 workers (5k local somas, but the loop still walks all 400k global soma ids):
**73 ms → 11 ms**.

`scaling_analysis/weak_scaling.py` never calls `set_recorded_somas`, so it runs in record-all mode and
pays this for all 400k somas. The cost is charged to `_simulation_time`, not construction time —
`_prepare_kernel_extras` runs inside `super().simulate(...)`, which is the span `model.py:672-674`
brackets.

### Fix

Build on the host, copy once. Keep the `if self._spike_mask_gpu is None` guard exactly as-is:

```python
buf = self._gpu_buffers
mask = np.zeros(buf.agent_capacity, dtype=np.float32)   # host
ids = self._soma_ids if self._recorded_soma_ids is None else self._recorded_soma_ids
for sid in ids:
    idx = buf.agent_id_to_index.get(sid, -1)
    if 0 <= idx < num_local_agents:
        mask[idx] = 1.0
self._spike_mask_gpu = cp.asarray(mask)                 # one H2D copy
```

---

## Finding 2 — the lazy cache must stay

An earlier draft proposed dropping the cache and rebuilding every `simulate()` to make staleness
impossible. **That is wrong for multi-rank runs.**

`_prepare_kernel_extras` runs once per `worker_coroutine` call, and that differs by path:

- `num_workers == 1` → `worker_coroutine(ticks)` called **once**; all ticks are fused into a single
  kernel launch (`sagesim/model.py:1123`).
- `num_workers > 1` → called **once per sync chunk**, `ticks // update_data_ticks + 1` times
  (`sagesim/model.py:1127-1140`).

`weak_scaling.py` defaults to `--update-ticks 1`, so multi-node runs chunk **once per tick**. Removing
the cache would rebuild the mask on every tick; at `--ticks 1000` that is roughly 73 s of added
overhead per `simulate()` at current build cost.

> **Invariant any change must preserve: at most one mask build per `simulate()`, on the first chunk.
> Never per tick.**

### Timing shape

Only the first chunk of each `simulate()` gets cheaper. Every subsequent tick is a cache hit and is
unchanged. Nothing gets slower.

### Where the cache never helps

- **Single-GPU runs** — one `worker_coroutine` call, so the mask is built once and used once. This is
  why the single-GPU 400k case pays the full 4.58 s.
- **Per-sample inference loops** (e.g. `tutorials/02_superneuroabm_digits.ipynb`) — `reset()` per sample
  nulls the mask, so it is rebuilt every sample regardless.

---

## Finding 3 — latent staleness window

`set_agent_property_value` and `set_local_agent_property_value` (`sagesim/model.py:553,590`) set
`buf.is_initialized = False`, so the next `simulate()` re-runs `_build_gpu_buffers` and rebuilds
`agent_id_to_index` (`sagesim/model.py:1212`). Neither nulls `_spike_mask_gpu`.

```
simulate()                  # mask built against index map A
set_agent_property_value()  # buffers invalidated; mask NOT
simulate()                  # index map rebuilt; stale mask silently reused
```

**Currently latent, not active.** With no agents added or removed the rebuilt map is identical, so the
stale mask still happens to be correct. It becomes a real bug once agent count or capacity changes:
`ensure_agent_capacity` doubles capacity (`sagesim/gpu_kernels.py:370-411`), which would leave the mask
undersized and produce an out-of-bounds read at `spike_mask[_real_idx]` in the generated kernel
(`model.py:1720`).

### Fix

Override both setters in `NeuromorphicModel` so they also drop the mask:

```python
def set_agent_property_value(self, id, property_name, value) -> None:
    super().set_agent_property_value(id, property_name, value)
    self._spike_mask_gpu = None   # buffers will rebuild; mask must follow

def set_local_agent_property_value(self, id, property_name, value) -> None:
    ...same...
```

These are the only two sites that clear `is_initialized` outside `GPUBufferManager`'s own constructor
and `free()`. `reset()` and `setup()` already null the mask, so with these two the invalidation set is
complete and **`set_recorded_somas()` becomes safe to call at any point before `simulate()`**.

This does not violate the per-simulate invariant: `worker_coroutine` — the function whose body is the
per-chunk loop — contains no property-setter calls, so nothing can invalidate the mask between chunks.
Spike injection (`add_spike`, `add_spike_list`, `add_local_spike_list`) does route through
`set_local_agent_property_value` (`model.py:1704`) and would null the mask, but injection always
precedes `simulate()` and never runs between chunks.

---

## Landing checklist

1. Land **between** scaling sweeps and re-baseline the affected points. A single scaling series must
   not be half pre-change and half post-change.
2. Add a **build-count regression test**: counter around the mask-build branch, then
   `simulate(ticks=20, update_data_ticks=1)` must build **exactly once**, not 20 times. Repeat with
   spike injection beforehand to confirm injection does not push the count above one. This is the test
   that would have caught the per-tick-rebuild regression in the first draft.
3. Extend `tests/test_spike_mask.py` with the staleness case: `set_recorded_somas` → `simulate` →
   `set_agent_property_value` → `simulate`, asserting the same somas are still recorded; plus
   `set_recorded_somas` called after `setup()` and again after `reset()`.
4. Multi-rank guard: run `tests/test_lif_mixed_synapses_stdp_mpi.py`, or any 2-rank run with
   `update_data_ticks=1`, and confirm `_simulation_time` did not grow. Treat any increase as a blocker.

---

## Unrelated doc corrections noticed along the way

- `docs/FUNCTIONALITY_GUIDE.md` documents `internal_state` / `internal_learning_state` (singular); the
  code uses `internal_states` / `learning_internal_states`. It also lists synapse `internal_state` as
  `[I_synapse, I_synapse_supp, pre_trace, post_trace]`, but the configs put only `I_synapse` there, with
  the traces in `learning_internal_states`.
- Both `docs/FUNCTIONALITY_GUIDE.md` and `docs/DATA_FORMAT.md` use
  `enable_internal_state_tracking` (missing the `s`) and a stale `setup(use_gpu=True)`; `setup()` takes
  no arguments since commit `bf17a22`.
- Neither doc mentions `set_recorded_somas` or `get_all_spike_times`.
