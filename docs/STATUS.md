# Status and next steps

Last updated 2026-10-16. This is the hand-off page: where the performance work stands,
what was measured, and what to do next, so that work can continue on another machine
from the repositories alone. The detailed log is `EVENT_DRIVEN_NOTES.md` (sections 15-16
and the "Step 2" entries are the latest); the measured positioning against other
simulators is `WHY_SUPERNEUROABM.md` and `WHY_SUPERNEUROABM_EVIDENCE.md`.

## Where it stands

SuperNeuroABM runs one input path: synapses are agents woken when their pre-soma fires
and deliver into the post-soma's inbox; the synaptic kinetics live on the soma. Learning
rules are event-woken and decay their traces on wake. Both repos' branches are
`perf/event-driven`.

Measured on a laptop GPU (RTX 3000 Ada), Brunel 12,500 neurons / 12.5 M synapses, 1 ms
ticks, 33 Hz:

| | per tick |
|---|---|
| one GPU, all ticks in one launch (the default single-GPU path) | 0.25-0.28 ms |
| NEST GPU 2.0, same network, same GPU | 0.20 ms |
| one GPU, one launch per tick (what multi-rank runs and verbose timing do) | 0.60 ms |
| 2 / 3 / 4 ranks sharing the one laptop GPU | 1.68 / ~2.0 / 2.43 ms |

Same spike trains at 2, 3 and 4 ranks, spike for spike. Masquelier 2008 (2,000 plastic
synapses, 0.1 ms ticks): 1.07 s for 10 s of model time, faster than Brian2 on CPU (5.2 s)
and NEST GPU (11.0 s). Construction is the big remaining gap: Brunel load + setup + first
tick is ~35 s against NEST GPU's 0.08 s.

What the 2026-10-15 round did (all committed, all gated by both suites and the MPI
runners at 1-4 ranks):

1. Fixed per-tick cost decomposed (section 15): the kernel's own fixed cost is 0.12 ms,
   of which the wake-expansion scan over every emitter is 0.08; the rest of the old
   "fixed" 0.6 ms was the host path of one launch per tick.
2. Launch path: SAGESim launches the compiled kernel directly after the first call
   (`sagesim/launch.py`); 0.36 -> 0.07 ms per launch.
3. Ghost exchange in a fixed number of device operations per tick (one gather per
   field, one permutation, one host copy, mirrored on receive; segmented add for the
   reverse inbox reduce). Was ~1 ms per peer, now ~0.2 ms per peer on the laptop, and
   that remainder is the shared GPU.
4. Only a soma's spike slot crosses ranks (8 bytes per ghost per tick instead of 28);
   custom kernels opt other properties in with `set_property_neighbor_visible`.

## Next: the Frontier campaign

Goal: replace the 2026-07-30 weak/strong scaling curves (pull-era kernel, silent
network) with the event path on a FIRING network, and get the per-tick breakdown on
real multi-GPU hardware (one GCD per rank, GPU-aware Cray MPICH).

Checklist, in order:

1. **Environment.** cupy (ROCm build), mpi4py, both repos `pip install -e`. Export
   `MPICH_GPU_SUPPORT_ENABLED=1` (SAGESim then sends device buffers straight to MPI; the
   laptop's Open MPI is not CUDA-aware, so that code path has NOT been exercised yet).
2. **Gate first, on 2 and 4 ranks of one node:** `tests/run_mpi_tests.sh` in SAGESim and
   in SuperNeuroABM with `PYTHON` set. The reduce tests must pass bitwise; a failure in
   `test_worker_sync` / `test_reduce_bla` with device buffers points at the GPU-aware
   path (fallback: unset `MPICH_GPU_SUPPORT_ENABLED` to stage through the host).
3. **Per-tick breakdown on real hardware:** `scaling_analysis/per_tick_probes/README.md`
   (split the 12.5k / K=1000 partition, run 1-4 ranks, compare spikes). Expect the
   launch column at ~0.1 ms, pack / unpack / reverse flat in the number of peers, and
   `mpi_wait_time` small; record the table next to section 16's.
4. **Weak scaling, firing:** `scaling_analysis/weak_scaling.py` with the Brunel
   asynchronous-irregular drive, e.g.
   `--neurons-per-worker 12500 --in-degree 1000 --J-E 0.02581 --g 5 --external-inputs 800 --firing-rate 14.53 --topology torus3d --ticks 1000 --diagnostics`
   over the same worker counts as before (`weak_3d_chunk.sh`). The driver now prints and
   records the spike count and rate; a silent run is flagged. The per-tick CSV has a
   `reverse` column since 2026-10-15; `analyze_weak.py` / `analyze_strong.py` read it
   and tolerate older CSVs without it.
5. **Strong scaling** the same way (`strong_scaling.py`, same drive flags).
6. Write the numbers into `EVENT_DRIVEN_NOTES.md` (new section) and update
   `WHY_SUPERNEUROABM_EVIDENCE.md` section 3.1a, which still describes the silent
   pull-era campaign.

Things to know on the cluster:
- Weak-scaling efficiency will read lower than the old 65-83 % even though every
  absolute number is lower: the kernel shrank from 1.1 ms to ~0.3 ms, so the constant
  launch + exchange term is a larger share (section 16 explains).
- One launch per tick is forced on multi-rank runs; `sync_workers_every_n_ticks` > 1
  is not valid with 1-tick delays.
- Spike recording is on for every soma by default and sized `somas x ticks / 4` per
  launch; `set_recorded_somas([])` if memory matters on long runs.
- Construction per run at 12,500 neurons / rank, K=1000, on the laptop: load 24 s,
  setup 6 s, first tick 41 s at 2 ranks (ghost discovery 12 s, device build 10 s,
  communication maps 16 s for 12,500 ghosts; ~2 ms of host Python per ghost).
  Bounded-halo topologies have fewer ghosts. Budget it; it is the next big item.

## After Frontier (not started)

- Fired-emitter compaction in SAGESim: the soma phase already tests the spike for
  recording; append fired rows to a list and expand wakes from that list instead of
  scanning every emitter (0.08 -> ~0.01 ms per tick, O(fired) instead of O(emitters)).
- Event-based spike exchange (ship ids of ghosts that spiked, O(spikes) instead of
  O(ghost rows)), and exchanging every min-delay ticks instead of every tick.
- Construction on the GPU (load 25 s + first tick 6 s vs NEST GPU 0.08 s).
- `poc/hetero_chip` is on the pre-inbox kernel API and does not run; it moves to its own
  repository with the flagship work.
- `origin/memristor_step_function` is not merged (old `globals` kernel signature).
