# Per-rank host-memory analysis: the 8-ranks-per-node simulate OOM

_Written 2026-07-23 while resolving the columnar-loader weak-scaling OOMs. This documents WHY
per-rank host memory is high at simulate, so the 8-ranks/node limit can be lifted later without
reducing the in-degree K. It is an analysis/roadmap, not a fixed spec._

## TL;DR

- On Frontier (512 GB/node, `DefMemPerNode=UNLIMITED`, no `--mem` on srun) the weak-scaling run
  packs the GPU count into `ceil(workers/8)` nodes, i.e. **8 ranks share one 512 GB node** for
  the 8/16/32-GPU cases.
- At `K=4000` (in-degree), each rank holds **~50 M synapse-agents** and its **simulate-time host
  peak is ~62–125 GB**. So 4 ranks/node fit (`4×peak < 512`) but **8 ranks/node OOM**
  (`8×peak > 512`). Confirmed by job 5061649: 1/2/4-GPU succeed, 8/16/32-GPU OOM at simulate.
- This is an **ordinary aggregate-memory limit, not a bug**. (The earlier catastrophic
  simulate OOM — a `(50M × thousands)` padded-tensor blow-up from a shared-list aliasing bug —
  is a separate, already-fixed issue; see "History" below.)
- **Immediate mitigation (in use): reduce K** so `12500·K` synapses/rank shrinks per-rank memory
  under `512/8 ≈ 60 GB`.
- **Better later:** the roadmap in the last section fits 8/node at full K by freeing resident
  structures after GPU upload and compressing the ~50M-entry bookkeeping dicts.

## History: two different simulate OOMs (do not conflate)

1. **Shape-explosion OOM (FIXED).** The columnar builder deduplicates property columns by
   sharing ONE list object across all ~50M synapse rows (`_syn_col → [vals[0]]*M`,
   `superneuroabm/model.py`). The host-side spike-injection helpers appended to that list **in
   place**, so injecting the 12500 input synapses grew the single shared object that every row
   aliases; `convert_to_padded_gpu_tensor` then allocated `(50M × max_len≈thousands)` float32 →
   hundreds of GB → OOM even at **1 GPU**. Fixed with copy-on-write in the 4 injection helpers
   (`add_spike`/`add_local_spike`/`add_spike_list`/`add_local_spike_list`). Record/dict path was
   never affected (it builds a fresh `[-1,0.0]` per synapse, `model.py` `_get_synapse_properties`).
2. **Aggregate per-rank OOM (THIS document).** With (1) fixed, per-rank memory is "normal" but
   still large because every synapse is a full agent. `8 × per-rank-peak > 512 GB`.

## Why per-rank memory is ~50 M-scale: synapses are agents

The model registers two SAGESim breeds — somas and synapses — and makes **each synapse a
first-class agent with its own id** (`superneuroabm/model.py` `_build_post_owned_columnar`:
`agent_ids = concat(soma_ids, synapse_ids)`). So per rank there are

```
agents ≈ neurons_per_worker + neurons_per_worker × K  =  12500 + 12500·K
       =  ~50,012,500  at K=4000     (synapse-dominated)
```

Every per-agent host structure is therefore ~50 M entries, not ~12 500. Memory scales ~linearly
with the synapse count, hence ~linearly with **K**.

## Per-rank host-memory breakdown at simulate (K=4000, ~50 M agents/rank)

Estimates (order-of-magnitude; agent ids exceed CPython's small-int cache so dict keys/values
are real ~28-byte PyLongs). "Resident" = built at load, alive through simulate; "First tick" =
added in SAGESim `_build_gpu_buffers` / `discover_ghost_topology`; nothing is freed in between.

| Structure | Where | ~Size/rank |
|---|---|---|
| `_agent2rank` dict (50M) | superneuroabm agent-factory (resident) | ~4.5 GB |
| `_agent2breed` dict (50M) | resident | ~4.5 GB |
| `_rank2agentid2agentidx` OrderedDict (50M) | resident | ~5 GB |
| `agent_id_to_index` dict (50M) | SAGESim `model.py:1316` (first tick) | ~4.5 GB |
| 9 property columns (50M pointer lists; values deduped) | resident | ~3.6 GB |
| `combined = local_data + ghost_data` copies of the 9 columns | `model.py:1409` (first tick) | ~3.6 GB |
| Prebuilt CSR (`_prebuilt_csr_*`, ~150M int32 vals + offsets) | `space.py` (resident) | ~0.8 GB |
| GPUHashMap host mirror (`_cpu_keys` 100M i64 + `_cpu_values` 100M i32) | `gpu_kernels.py:118` | ~1.2 GB |
| Padded host staging for GPU upload (CSR 100M×2 @ CSR_SLACK=2.0; per-prop `(75M×width)` @ AGENT_SLACK=1.5) | `gpu_kernels.py` / `internal_utils.py` | several GB |
| `discover_ghost` transients on the 150M CSR (concatenate/unique/searchsorted) | `gpu_kernels.py:56` | ~2–3 GB (transient) |
| `__rank_local_agent_ids`, `all_agent_ids_np` (50M i64 each) | `model.py:1315,1601` | ~0.8 GB |

**Biggest bucket: the ~4 parallel ~50M-entry Python dicts ≈ 15–20 GB.** The rest (columns +
copies + hashmap mirror + padded staging + transients) adds ~15–25 GB. Load-time and
simulate-time footprints **stack** because `_build_gpu_buffers` never frees the load-time
structures after uploading to GPU. Total lands in the observed **~62–125 GB** window.

### Why the 62–125 GB bracket (not an exact number)

We infer it from the boundary rather than a profiler:
- 4 ranks/node **fit**  ⇒ `4 × peak ≲ 512 − OS`  ⇒ `peak ≲ ~125 GB`.
- 8 ranks/node **OOM** ⇒ `8 × peak ≳ 512 − OS`  ⇒ `peak ≳ ~62 GB`.

A precise number would come from wrapping a **surviving** case (e.g. 4-GPU) in
`/usr/bin/time -v` and reading `Maximum resident set size`. Recommended as a follow-up to
calibrate the K choice and to measure the effect of each optimization below. (An OOM-killed
case can't report its peak.)

## Immediate mitigation: reduce K

Per-rank memory ≈ `fixed_overhead + slope × (12500·K)`. To fit 8/node we need `peak(K) < ~60 GB`.
Since `peak(4000) ∈ (62,125) GB`, the fitting K is roughly in the **2000–3000** range depending
on the fixed-overhead fraction. `probe_k_8pernode.sh` measures it directly: it runs the 8-GPU/
1-node case (the binding per-node pressure) at descending K with `--ticks 2` and reports the
largest K that gets past `RUNNING SIMULATION` with no `oom_kill`. The full curve then runs at
that K with 8/node packing unchanged; the shared CSV auto-names by K
(`weak_3d_curve_K<K>_npp12500.csv`) so it never mixes with the K=4000 data.

Note: reducing K changes the network's in-degree (a modeling parameter), so K-reduced and
K=4000 curves are separate experiments — fine for a weak-scaling study (per-rank work is still
held constant across ranks within each curve).

## Roadmap: fit 8/node at FULL K (address later)

Ranked by impact/effort. Target: shave per-rank peak from ~90 GB to < 60 GB.

1. **Free load-time structures after GPU upload (~5–10 GB, low-med effort).** In
   `_build_gpu_buffers`, once each property tensor / the CSR is uploaded to GPU, the host copies
   (`af._property_name_2_agent_data_tensor`, `space._prebuilt_csr_*`, padded staging arrays) are
   dead but never released — add explicit `del`/`.clear()` + drop references so load and simulate
   footprints don't stack. (Confirmed there are currently no frees in `_build_gpu_buffers`.)
2. **Compress the ~50M-entry bookkeeping dicts (~10–15 GB, med-high effort — biggest win).**
   On the post-owns-synapse columnar path `_agent2rank` is entirely local (a single rank value)
   and agent ids are near-contiguous, so `_agent2rank`/`_agent2breed`/`_rank2agentid2agentidx`/
   `agent_id_to_index` can be numpy arrays / ranges / a compact typed map instead of four
   parallel Python dicts of PyLongs. Ties into the existing bookkeeping-cleanup effort.
3. **Skip redundant first-tick copies (~3.6 GB, low effort).** Use `combined = local_data` when
   `num_ghost == 0`, or make the MPI width-padding copy-on-write so the `local_data + ghost_data`
   duplication of every property column is avoided.
4. **Stream property-tensor upload (several GB, med effort).** Build → upload → free each
   property tensor in turn instead of holding all `combined_lists` simultaneously.
5. **Done already this session:** dropped the ~5 GB `.tolist()` of the 150M-value CSR (kept it a
   numpy int32 array via a `return_arrays` flag in `convert_agent_ids_to_indices`); added
   `_warn_if_huge_padded` (SAGESim `internal_utils.py`) so any future `(capacity × width)` blow-up
   prints its shape before allocating instead of silently OOM-killing.

Items 1–4 together plausibly reclaim ~25–35 GB/rank, which would bring K=4000 under the ~60 GB
8/node budget. Validate each with `/usr/bin/time -v` on a surviving case.

## Key code references
- Agents = somas + synapses: `superneuroabm/model.py` `_build_post_owned_columnar`.
- Column dedup (the shared-object source): `_syn_col` in `superneuroabm/model.py`.
- First-tick buffer build + the 50M dict + `combined` copies: `SAGESim/sagesim/model.py`
  `_build_gpu_buffers` (~1266) and `worker_coroutine` (~1582).
- Padded-tensor allocation (the shape-explosion site) + the new guard:
  `SAGESim/sagesim/internal_utils.py` `convert_to_padded_gpu_tensor` / `_warn_if_huge_padded`.
- CSR remap without `.tolist()`: `convert_agent_ids_to_indices(..., return_arrays=True)` in
  `SAGESim/sagesim/model.py`.
- Slack factors: `AGENT_SLACK_FACTOR=1.5`, `CSR_SLACK_FACTOR=2.0` in `SAGESim/sagesim/gpu_kernels.py`.
