# SuperNeuroABM vs NEST GPU on Perlmutter: status and how to resume

Last updated 2026-11-14 (evening). This page is the hand-off: what the comparison is, what is
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

## Three things learned the hard way

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
3. **Job hygiene.** Wrap every `srun` in `timeout`, pipe through `grep --line-buffered`, and
   remember that debugging with `srun --jobid --overlap` uses up step numbers.

## Numbers so far

Test network, simulate 1.5 s: NEST GPU 2.45 / 5.18 / 4.67 s at 1 / 2 / 4 GPUs, SuperNeuroABM
0.95 / 7.85 / 8.36 s.

Benchmark size, neurons started at rest, 50 ms warm-up then the timed 250 ms (job 61322433):

| | SuperNeuroABM | NEST GPU, same files | NEST GPU, own script |
|---|---|---|---|
| 1 GPU: simulate 250 ms | 0.236 s | 0.344 s | segfault |
| 1 GPU: build | 74.9 s (load 52.6, setup 4.0, first tick 18.3) | 12.0 s (read 7.7, connect 3.7, calibrate 0.7) | |
| 1 GPU: host / GPU memory | 47 / 20.2 GiB | 11 / 2.3 GiB | |
| 2 GPUs: simulate 250 ms | 1.475 s | 0.944 s | 0.549 s (at 52 Hz) |
| 2 GPUs: build | 88.8 s | 40.7 s (22.8 s in 23,531 RemoteConnect calls) | 0.26 s |

SuperNeuroABM's tick at 2 GPUs by SAGESim's timers (job 61323573): 0.63 ms, of which 0.38 ms
is the ghost exchange (`data_prep`) and 0.23 ms the per-tick kernel launch; one GPU runs all
ticks in one launch at 0.09 ms per tick.

## Open

- Activity at benchmark size is not matched yet: the two simulators agree spike for spike
  for 31 ms (1 GPU) and 52 ms (2 GPUs), then 11.6 vs 9.8 Hz and 9.0 vs 9.6 Hz. NEST GPU's own
  script gives 52 Hz at 2 GPUs and 26 Hz at 4. Job 61323427 (`hpc-vm-check-61323427.out`)
  repeats the runs with hpc_benchmark's random starting potentials and adds the NEST GPU
  control with weights x (1 + 1e-7), at 1, 2 and 4 GPUs.
- Job 61323573 (`tick-breakdown-61323573.out`): the remaining two parts, SuperNeuroABM with
  GPU buffers straight to MPI and NEST GPU's own timers.
- Then: 2, 4 and 8 nodes in the debug queue (partitions for 8, 16, 32 ranks, 4 GB each),
  and up to 128 nodes in the regular queue. At hundreds of ranks the SuperNeuroABM runner's
  `get_all_spike_times` (every rank receives every spike) needs a cheaper path.
- The repo's test gate (pytest and the MPI tests) has not been run for these commits; they
  touch tutorials and the comparison scripts, not the library.
