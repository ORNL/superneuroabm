# CPU/GPU Property Sync: `reset()` as the Hidden Sync Point

Design notes on why changing a property on a trained model requires calling `reset()` first, why
that is confusing, and what could be done about it. **The sync layer itself is still unchanged.**
This records the analysis so it can be picked up without redoing it.

> **See also** [CPU_GPU_DATA_FLOW.md](CPU_GPU_DATA_FLOW.md) — the companion reference describing how
> the two copies actually behave today: what each operation touches, what `is_initialized` does and
> does not tell you, the full `_device_dirty` design chosen for the fix, and the related
> `simulate()` host-write issue. This document is the critique and the option space; that one is the
> ground truth to implement against.

**Status.** Option **(a)/(c)** below — a `_globals_dirty`-style staleness flag checked at the point
of use — is the chosen design, written up in detail in `CPU_GPU_DATA_FLOW.md` §5. It has not been
implemented. Two things landed in the meantime:

- **The layer split was settled**: coherence belongs in SAGESim (the property setters), naming
  belongs in superneuroabm. The fix must also cover `_write_stdp_type` / `_read_stdp_type`, which
  bypass the setter, or `eval()` stays order-dependent.
- **The hazard is now guarded rather than silent** at the level users touch:
  `set_hyperparameters` / `set_learning_hyperparameters` refuse to run while GPU buffers hold
  unsynced state and name the fix in the error. `set_agent_property_value` remains unguarded.
  `tests/test_learning_mode.py::test_raw_setter_after_simulate_loses_learned_weights` pins the
  underlying bug as an `expectedFailure`, so it reports *unexpected success* the moment the real
  fix lands — that is the trigger to revisit both documents and delete the guard.

**Why it is still deferred:** `tests/test_learning_mode.py:89-93` encodes the ordering rule as
*intended* behaviour, and the change spans two repos. It needs its own pass with its own test
updates rather than riding along with tutorial work.

---

## The rule users have to learn

To change a parameter on a trained model — say, switch the neurons to a different operating point
for inference — this works:

```python
model.simulate(ticks=N)                     # train
model.reset(retain_parameters=True)         # <-- REQUIRED, and not for the reason the name suggests
model.eval()
model.set_agent_property_value(soma_id, "hyperparameters", hp)
model.simulate(ticks=M)                     # infer
```

and this silently destroys every learned weight:

```python
model.simulate(ticks=N)
model.set_agent_property_value(soma_id, "hyperparameters", hp)   # no reset() first
model.simulate(ticks=M)
```

No exception, no warning. The second `simulate()` just runs with the pre-training weights. The only
symptom is bad numbers.

## Why

Every property exists in two places: the CPU-side `AgentFactory` lists
(`_property_name_2_agent_data_tensor`) and the GPU padded tensors (`GPUBufferManager.property_tensors`).
Ownership is implicit and flips with `_gpu_buffers.is_initialized`. The getter and setter then
disagree about which copy they touch:

```python
# sagesim/model.py:506 -- READS THE GPU when buffers are live
def get_agent_property_value(self, id, property_name):
    if self._is_setup and ... self._gpu_buffers.is_initialized:
        ...
        result = buf.property_tensors[prop_idx][buf_idx].get().tolist()   # device read
        ...
    # else: CPU path

# sagesim/model.py:547 -- ALWAYS WRITES THE CPU
def set_agent_property_value(self, id, property_name, value):
    self._agent_factory.set_agent_property_value(...)          # host lists
    if ... self._gpu_buffers.is_initialized:
        self._gpu_buffers.is_initialized = False               # mark stale; rebuild from CPU
        self._cached_all_args = None
```

After `simulate()`, STDP has updated weights *on the device*. The host lists still hold the values
from model construction. So:

- a **read** returns the trained weight (device path) — everything looks fine;
- a **write** lands on the stale host array and drops the "GPU is newer" fact on the floor;
- the next `simulate()` rebuilds the device buffers from that stale host array, so all learned
  weights revert and only the one write survives.

The single place that closes the gap is `_sync_gpu_to_agent_factory` (`sagesim/model.py:1065`),
which downloads every device tensor back into the host lists. It has exactly one caller —
`Model.reset()` (`sagesim/model.py:1092`), before it frees the buffers:

```python
def reset(self) -> None:
    self.tick = 0
    self._globals_dirty = True
    if hasattr(self, '_gpu_buffers') and self._gpu_buffers.is_initialized:
        self._sync_gpu_to_agent_factory()          # <-- the entire sync story
    self._regenerate_data_tensors()
    ...
```

There is a second downloader, `_download_local_data_to_cpu` (`sagesim/model.py:1418`), which is
dead code — grep across both packages finds no caller.

`superneuroabm/model.py:564-595` (`_write_stdp_type`) documents this contract correctly in a
comment, and `NeuromorphicModel.reset()` (`:692-724`) is careful to run `super().reset()` before
`_reset_agents()` for exactly this reason. The knowledge exists; it just is not expressed in the API.

## What is actually wrong with it

1. **The sync is a side effect of an unrelated method.** `reset()` promises "clear state". Nothing
   in its name, signature, or docstring says "…and this is your only opportunity to make host reads
   authoritative." A user who does not want to clear state has no supported way to sync.
2. **The getter/setter asymmetry is invisible.** The pair looks symmetric. One reads device, one
   writes host. Nothing in the signatures hints at it.
3. **It is a half-built write-back cache.** The design invalidates on write but never
   checks-before-read on the host path, which is the other half of the pattern. The codebase already
   implements the complete pattern elsewhere: `_globals_dirty` is set on every global write
   (`sagesim/model.py:728, 736, 1088`) and *checked at the point of use* before the kernel launch
   (`:1652-1654`). Properties have the invalidation half without the check half.
4. **Failure is silent and delayed.** The damage happens at the *next* `simulate()`, far from the
   offending line, and produces plausible-looking numbers rather than an error.
5. **`reset(retain_parameters=...)` bundles two unrelated operations** behind one boolean: clearing
   transient state (voltages, currents, spike logs, tick counter) and restoring parameters from
   config. Callers routinely want the first without the second, and the flag name describes neither
   well.

## How PyTorch avoids this entirely

PyTorch's relevant property is not a better sync — it is **no second copy to sync**.

- **One authoritative tensor with an explicit `.device`.** A `Parameter` lives on exactly one device.
  `module.to("cuda")` *moves* it (`Module._apply` swaps `param.data` for the device tensor); it does
  not create a shadow host copy that can drift. Reads and writes therefore always hit the same
  storage, and "host and device disagree" is not a representable state. Where PyTorch does copy —
  `t.cpu()`, `t.numpy()` — the result is a distinct object the user named, so any staleness is
  explicit and local.
- **Mode flips touch no data.** `model.eval()` / `model.train()` only set the recursive `self.training`
  flag that Dropout and BatchNorm read. There is no ordering constraint against anything else,
  because they move no bytes. This repo's `eval()` is *also* a pure mode flip
  (`superneuroabm/model.py:627`) — the ordering constraint on it comes entirely from the property
  layer underneath, which is precisely the leak.
- **Separate operations for separate concerns.** Clearing transients (`optimizer.zero_grad()`),
  snapshotting parameters (`state_dict()`), and restoring them (`load_state_dict()`) are three
  distinct calls. Nothing bundles them behind a boolean, and none of them is secretly a sync point.
- **The only thing named "synchronize" is about streams, not location.** `torch.cuda.synchronize()`
  orders kernel execution; it never moves parameters. Worth keeping the two ideas separate — the
  problem here is *data location*, not *stream ordering*.

Worth noting what PyTorch pays for this: no host-side mirror means every read is a device round-trip
(with an implicit stream sync), which is why `.item()` in a training loop is a known performance
footgun. This repo's host mirror exists partly to make the MPI-collective read path cheap. So
"delete one of the copies" is not automatically the right answer here — but the current design gets
the cost of two copies *and* the correctness hazard of none.

## Candidate fixes, cheapest first

**(a) Sync-before-write in the setter.** Make `set_agent_property_value` call
`_sync_gpu_to_agent_factory()` when `is_initialized` is true, before touching the host list. The
ordering rule disappears entirely; the failing example at the top of this doc becomes correct.
*Cost:* the first write after a `simulate()` triggers a full device→host download of every property.
Batched writes pay it once (the second write finds `is_initialized` already false), so the notebook
pattern of writing 640 synapses costs one download, not 640. *Blast radius:* small and local; the
existing correct call sites keep working unchanged, since a sync right after `reset()` already
synced is a no-op.

**(b) Write-through to the device.** When buffers are live, have the setter write
`buf.property_tensors[prop_idx][buf_idx]` directly and leave `is_initialized` alone — symmetric with
the getter, which already reads that exact location. No download at all, and no buffer rebuild.
*Cost:* one small H2D transfer per write, and the padding/`num_local_agents` bookkeeping has to be
right. *Watch out:* `_regenerate_data_tensors()` hands out references to the host lists
(`sagesim/agent.py:350-354`), so the host copy still needs updating too, or it becomes the stale one.
Probably the correct end state, and it composes with (a).

**(c) Finish the write-back cache.** Add a `_device_dirty` flag set when a kernel runs, checked on
every host read path (mirroring `_globals_dirty`). Most principled of the three, and it makes the
invariant explicit rather than implied by call ordering. *Cost:* every host access path has to honour
it, including the MPI-collective ones; more places to get wrong.

**(d) Split the API regardless of which of the above lands.** A public `sync_from_device()` that
does the download and nothing else, so the operation is nameable and does not have to be smuggled in
through `reset()`. Separately, split `reset(retain_parameters=...)` into two verbs — one that clears
transient state, one that restores parameters from config — since callers want them independently
and the boolean is not self-describing. This is a naming/ergonomics fix and is worth doing even if
the underlying sync stays as-is.

## Blast radius if any of this changes

- `tests/test_learning_mode.py:83-102` asserts the current ordering (`reset()` then `eval()`) and its
  comment explains the hazard as intended behaviour. Under (a) or (b) both orderings become correct,
  so the test should be rewritten to assert *that* rather than deleted.
- `tests/test_model_reset_stdp.py` pins `retain_parameters` semantics — unaffected by (a)/(b), needs
  updating under (d).
- `tutorials/02_superneuroabm_digits.ipynb` and `tutorials/03_masquelier_2008_stdp.ipynb` both do
  `reset(retain_parameters=True)` before reading or writing weights. Correct under every option
  above; only the explanatory comments would need softening.
- `superneuroabm/model.py:564-595` (`_write_stdp_type`) hand-rolls the invalidation dance and would
  simplify under (b).
- MPI: `get`/`set_agent_property_value` are collective, the `*_local_*` variants are not
  (`sagesim/model.py:556-596`). Any sync added to a setter must not introduce a collective call on
  the non-collective path.
