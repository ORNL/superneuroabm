# SuperNeuroABM vs NEST GPU on Perlmutter: status and how to resume

Last updated 2026-11-14, 21:15. This page is the hand-off: what the comparison is, what is
verified, the numbers so far, what is running and what comes next.

## The plan (owner's, 2026-11-14)

Compare the speed of SuperNeuroABM and NEST GPU on several GPUs, each simulator given the
network as data: per-rank partition files, and an external drive replayed from spike lists.
NEST GPU's connection rules and its Poisson generator are not used, so the comparison is what
a real-world network would cost. In order:

1. NEST GPU is right: our file-driven runner checked against itself across rank counts, and
   against NEST GPU's own released scaling script (`python/examples/hpc_benchmark/hpc_benchmark.py`).
2. SuperNeuroABM gives the same results on several GPUs.
3. SuperNeuroABM and NEST GPU give the same results.
4. Then compare construction and run times, keeping many timers.

Size: NEST GPU's hpc_benchmark at scale 1, i.e. 11,250 neurons and in-degree 11,250
(126.6 M synapses) on every GPU, the network growing with the GPU count. Every GPU then
listens to every neuron, so this is a speed comparison per size, not a weak-scaling claim for
SuperNeuroABM. First the debug queue (at most 8 nodes = 32 GPUs, 30 minutes, 2 running and 5
submitted jobs), then up to 128 nodes in the regular queue.

## Where things are

- Repo: `/global/cfs/cdirs/m5460/zhangx8/superneuroabm`, branch `perf/event-driven`.
- Environments: `source /global/cfs/cdirs/m5460/envs/sna-dev.sh` (SuperNeuroABM + SAGESim,
  editable) and `source /global/cfs/cdirs/m5460/envs/nestgpu.sh` (NEST GPU v2.0, source in
  `/global/cfs/cdirs/m5460/zhangx8/software/nest-gpu/src`). Account `m5460_g`.
- Work directory with partitions, spikes and job outputs: `/pscratch/sd/z/zhangx8/partitioned`
  (`brunel_alpha_12500/` is the test network, `hpc_11250/` the benchmark size).
- Job scripts: `perlmutter/` next to this file (copies of the ones in the work directory;
  they hold absolute paths).

| script | what it runs |
|---|---|
| `regress_12500.sbatch` | both runners on the 12,500-neuron network; run after any change to them |
| `hpc_node_check.sbatch` | benchmark size on one node: writes the parts for `RANKS`, runs both simulators, compares spikes |
| `hpc_vm_check.sbatch` | the same with random starting potentials, plus NEST GPU with weights x (1 + 1e-7) |
| `tick_breakdown.sbatch` | each simulator's own per-tick timers at 2 ranks |
| `nest_own_hpc.sbatch` | NEST GPU's own hpc_benchmark.py, unmodified |
| `sna_mpi_variants.sbatch` | SuperNeuroABM under three MPI / GPU setups |
| `own_gpu_first.sh` | wrapper: every GPU visible to a rank, its own first |

Run flags at benchmark size, both runners:
`<root> --config hpc_alpha_config.yaml --eta 1.685 --presim-ms 50 --ms 300 [--vm-sigma 7.2]`.

## Verified

On the 12,500-neuron test network (15.6 M synapses, 1.5 s, about 29 Hz):

- NEST GPU at 2 and 4 ranks reproduces its 1-rank run spike for spike over the whole run
  (539,278 spikes).
- SuperNeuroABM at 2 and 4 ranks: 28.83 to 28.92 Hz against 28.86 Hz, identical to 1 rank for
  the first 518 ms. It is not bitwise reproducible run to run (two 1-rank runs agree for
  837 ms), so that is the criterion.
- SuperNeuroABM against NEST GPU at 1 rank: identical for 94 ms, rates within 0.1 %.

At benchmark size, every soma started at normal(5.7, 7.2) mV as hpc_benchmark.py does
(`--vm-sigma 7.2`), 300 ms (job 61324320):

- 1 GPU: SuperNeuroABM and NEST GPU give the same 32,712 spikes, spike for spike (9.80 Hz).
- 2 GPUs: the same 64,194 spikes, spike for spike (9.50 Hz).
- 4 GPUs: identical for 253.7 ms, then 105,231 against 106,329 spikes (7.95 and 8.05 Hz).

## Four things learned the hard way

1. **NEST GPU, cross-rank connections from data.** A `RemoteConnect` call must not name a
   source node twice: only the first entry of a repeated source is mapped to the right image
   node (`remote_connect.cu`, `searchNodeIndexNotInMapKernel` and `insertNodesInMapKernel`;
   the same in the 2026-10 head). Per-synapse source lists therefore give a wrong network
   with no error (64 to 148 Hz instead of 28.9 Hz). The runner sends each rank the edges that
   leave it (`edge_exchange`), splits the edges of a host pair into calls with distinct
   sources, and makes both ranks issue the same calls. The kernel is set as hpc_benchmark.py
   sets it (31 node bits, no synapse groups, minimum delay = the delay).
2. **Perlmutter, GPU-aware MPI.** `--gpus-per-task 1` isolates each rank's GPU with cgroups,
   and Cray MPICH's GPU-to-GPU IPC then fails (`cuIpcOpenMemHandle: invalid argument`):
   SuperNeuroABM hangs in the ghost exchange. What works: `MPICH_GPU_SUPPORT_ENABLED=0`
   (host-staged, used so far), or every GPU visible to every rank
   (`srun --gpus N --gpu-bind=none own_gpu_first.sh ...`), or `MPICH_GPU_IPC_ENABLED=0` (twice
   as slow on one node). Not yet tried across nodes.
3. **NEST GPU, replayed input and two Simulate() calls.** A `spike_generator` matches its
   spike times against the step counter, which restarts at 0 in every `Simulate()` call
   (`nestgpu.cu` StartSimulation, `spike_generator.cu`). After a warm-up call of T ms the
   replayed drive is silent for T ms and then T late: 35 spikes instead of 13,454 in the
   50 ms after a 50 ms warm-up. The runner therefore runs ONE simulation, stepped from Python
   with `NESTGPU_StartSimulation` / `NESTGPU_SimulationStep` (what `Simulate()` does), and
   times the warm-up steps apart. NEST GPU numbers at benchmark size from jobs 61322433,
   61323427 and 61323573 were taken with the drive paused and are void.
4. **Job hygiene.** Wrap every `srun` in `timeout`, pipe through `grep --line-buffered`, and
   remember that debugging with `srun --jobid --overlap` uses up step numbers.

## Numbers so far

Test network, simulate 1.5 s: NEST GPU 2.46 / 5.15 / 4.69 s at 1 / 2 / 4 GPUs, SuperNeuroABM
0.95 / 7.77 / 8.36 s.

Benchmark size, random starting potentials, 50 ms warm-up then the timed 250 ms, one node
(job 61324320; the two simulators' spikes are the same, see Verified):

| GPUs | | SuperNeuroABM | NEST GPU, same files |
|---|---|---|---|
| 1 | simulate 250 ms | 0.230 s (0.091 ms per tick) | 0.403 s (0.161 ms per tick) |
| 1 | build | 71.4 s (load 54.4, setup 4.0, first tick 13.0) | 11.6 s (read 7.3, connect 3.6, calibrate 0.7) |
| 2 | simulate 250 ms | 1.450 s (0.577 ms per tick) | 1.156 s (0.462 ms per tick) |
| 2 | build | 102.0 s (load 64.5, setup 4.3, first tick 33.2) | 38.8 s (read 8.0, boundary 2.1, local 3.5, edge exchange 2.3, remote 22.1 in 23,531 calls, calibrate 0.8) |
| 4 | simulate 250 ms | 1.824 s (0.726 ms per tick) | 1.224 s (0.490 ms per tick) |
| 4 | build | 151.8 s (load 88.5, setup 5.1, first tick 58.2) | 54.4 s (read 8.6, boundary 5.6, local 2.4, edge exchange 4.4, remote 32.7 in 36,107 calls, calibrate 0.7) |
| any | drive | draw 1.9 s + to device 0.9 s | draw 1.8 s + generator setup 6.5 s |
| any | host / GPU memory per rank | 47 to 51 / 20.2 GiB | 10 to 11 / 2.3 to 2.4 GiB |

The same on 2, 4 and 8 nodes (4 GPUs each; jobs 61325020, 61325021, 61325023, script
`perlmutter/hpc_scale.sbatch`), with one node repeated from above. "Own script" is NEST GPU's
hpc_benchmark.py, unmodified: rule-based connect, Poisson generator.

| GPUs | neurons | synapses | simulate 250 ms: SuperNeuroABM | NEST GPU, same files | NEST GPU, own script | rate (Hz): SNA / NEST files / own |
|---|---|---|---|---|---|---|
| 1 | 11,250 | 0.13 B | 0.230 s (0.091 ms/tick) | 0.403 s (0.161) | segfault | 9.80 / 9.80 / - |
| 2 | 22,500 | 0.25 B | 1.450 s (0.577) | 1.156 s (0.462) | 0.549 s | 9.50 / 9.50 / 52.0 |
| 4 | 45,000 | 0.51 B | 1.824 s (0.726) | 1.224 s (0.490) | 0.642 s | 7.95 / 8.05 / 25.9 |
| 8 | 90,000 | 1.01 B | 2.345 s (0.933) | 1.318 s (0.527) | 0.893 s | 8.05 / 8.14 / 8.52 |
| 16 | 180,000 | 2.03 B | 3.656 s (1.455) | 1.486 s (0.594) | 1.077 s | 7.62 / 7.89 / 7.90 |
| 32 | 360,000 | 4.05 B | 7.382 s (2.937) | 1.830 s (0.732) | 1.431 s | 7.56 / 7.64 / 7.88 |

| GPUs | build: SuperNeuroABM | NEST GPU, same files | NEST GPU, own script |
|---|---|---|---|
| 8 | 194 s (load 99.9, setup 4.1, first tick 90.1) | 61 s (read 8.8, boundary 8.0, local 1.4, edge exchange 5.5, remote 36.4 in 43,339 calls, calibrate 0.8) | 0.32 s |
| 16 | 196 s (load 94.3, setup 4.3, first tick 97.6) | 64 s (remote 36.2 in 48,241 calls, boundary 10.6, edge exchange 7.0) | 0.33 s |
| 32 | 220 s (load 110.5, setup 4.7, first tick 104.7) | 65 s (remote 36.1 in 52,525 calls, boundary 11.3, edge exchange 6.9) | 0.36 s |

Spikes, SuperNeuroABM against NEST GPU on the same files: identical for 62.0 ms at 8 GPUs,
116.6 ms at 16, 27.8 ms at 32, then rounding drift; rates within 1 to 3.5 %. From 8 GPUs on,
hpc_benchmark.py's own rate (8.5, 7.9, 7.9 Hz) agrees with the file-driven runs; at 2 and 4
processes it gives 52 and 26 Hz, in the dense regime its docstring warns about (unstable
below scale 10), and 1 process segfaults.

Per rank at every size: SuperNeuroABM 51 GiB host and 20.2 to 20.5 GiB GPU, NEST GPU 11 GiB
and 2.3 to 2.9 GiB.

SuperNeuroABM's tick at 2 GPUs by SAGESim's timers (job 61323573): 0.63 ms, of which 0.38 ms
is the ghost exchange (`data_prep`) and 0.23 ms the per-tick kernel launch; 0.60 ms with GPU
buffers straight to MPI. One GPU runs all ticks in one launch at 0.09 ms per tick.

## Open

- NEST GPU's own per-tick timers (`--kernel-timers`) must be retaken with the stepped
  simulation; the ones in `tick-breakdown-61323573.out` had the drive paused.
- `--perturb 1e-7` is below float32 resolution for these weights and changes nothing; a
  control run needs 1e-6.
- In the regular queue since 2026-11-14 21:10: 16, 32, 64 and 128 nodes (jobs 61326380,
  61326383, 61326384, 61326385; outputs `hpc-scale-<job>.out` in the work directory). Things
  to watch there: SuperNeuroABM's tick grows with the number of peer ranks (0.93, 1.46,
  2.94 ms at 7, 15, 31 peers) and its first tick with the ghost somas; the runner's
  `get_all_spike_times` hands every rank every spike; host memory is at 51 of about 62 GiB
  per rank.
- The repo's test gate (pytest and the MPI tests) has not been run for these commits; they
  touch tutorials and the comparison scripts, not the library.
