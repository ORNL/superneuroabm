# SuperNeuroABM vs NEST GPU on Perlmutter: status and how to resume

Last updated 2026-11-15. This page is the hand-off: what the comparison is, what is
verified, the numbers so far, what is running and what comes next. The comparison has now
run on 1 to 512 GPUs (128 nodes); the tables are under "Numbers so far". The same benchmark
on a 3D torus is under way: see "The same benchmark on a 3D torus", which lists the jobs in
the queue.

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
| `hpc_scale.sbatch` | benchmark size on N nodes (`sbatch -N <nodes>`): parts, both simulators, spike comparison, NEST GPU's own script |
| `tick_breakdown.sbatch` | each simulator's own per-tick timers at 2 ranks |
| `tick_breakdown_32.sbatch` | the same at 32 ranks (8 nodes), and NEST GPU with excitatory weights x (1 + 1e-6) as the rounding control |
| `nest_own_hpc.sbatch` | NEST GPU's own hpc_benchmark.py, unmodified |
| `torus_node_check.sbatch` | the benchmark on a 3D torus, one node (`NET=library`, `exact` or `mixed`): parts for `RANKS`, both simulators, spike comparison |
| `torus_scale.sbatch` | the same on N nodes (`sbatch -N <nodes> --export=ALL,NET=mixed`); `TIMERS=1` adds a run of each simulator with its per-tick timers; `STRONG=92160` runs one fixed network of that many neurons instead (strong scaling) |
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

The rounding control, at 32 GPUs (job 61334376):

- NEST GPU is reproducible run to run: two runs give the same 825,535 spikes.
- NEST GPU with the excitatory weights multiplied by 1 + 1e-6 differs from the plain run
  from 5.9 ms on (one soma one tick apart, then everything): 814,870 spikes, 7.50 Hz
  against 7.64 Hz, 1.9 % apart.
- SuperNeuroABM differs from NEST GPU from 27.8 ms on and by 1.0 % in rate at that size
  (two SuperNeuroABM runs: 7.56 and 7.52 Hz). So the differences between the two simulators
  are no larger than what one part in a million in the weights does to NEST GPU alone, and
  the rate over 250 ms carries 0.5 to 2 % of such spread at this size.

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

The same on 2 to 128 nodes (4 GPUs each; script `perlmutter/hpc_scale.sbatch`; debug-queue
jobs 61325020, 61325021, 61325023 for 2, 4, 8 nodes, regular-queue jobs 61326380, 61326383,
61326384, 61326385 for 16, 32, 64, 128 nodes), with one node repeated from above. "Own
script" is NEST GPU's hpc_benchmark.py, unmodified: rule-based connect, Poisson generator.

| GPUs | neurons | synapses | simulate 250 ms: SuperNeuroABM | NEST GPU, same files | NEST GPU, own script | rate (Hz): SNA / NEST files / own |
|---|---|---|---|---|---|---|
| 1 | 11,250 | 0.13 B | 0.230 s (0.091 ms/tick) | 0.403 s (0.161) | segfault | 9.80 / 9.80 / - |
| 2 | 22,500 | 0.25 B | 1.450 s (0.577) | 1.156 s (0.462) | 0.549 s | 9.50 / 9.50 / 52.0 |
| 4 | 45,000 | 0.51 B | 1.824 s (0.726) | 1.224 s (0.490) | 0.642 s | 7.95 / 8.05 / 25.9 |
| 8 | 90,000 | 1.01 B | 2.345 s (0.933) | 1.318 s (0.527) | 0.893 s | 8.05 / 8.14 / 8.52 |
| 16 | 180,000 | 2.03 B | 3.656 s (1.455) | 1.486 s (0.594) | 1.077 s | 7.62 / 7.89 / 7.90 |
| 32 | 360,000 | 4.05 B | 7.382 s (2.937) | 1.830 s (0.732) | 1.431 s | 7.56 / 7.64 / 7.88 |
| 64 | 720,000 | 8.10 B | 14.370 s (5.718) | 2.563 s (1.025) | 3.906 s | 7.60 / 7.56 / 7.82 |
| 128 | 1,440,000 | 16.20 B | 27.472 s (10.932) | 3.784 s (1.513) | 6.971 s | 7.58 / 7.52 / 7.71 |
| 256 | 2,880,000 | 32.40 B | 58.591 s (23.315) | 6.381 s (2.552) | 12.647 s | 7.56 / 7.56 / 7.58 |
| 512 | 5,760,000 | 64.81 B | 124.591 s (49.579) | 12.131 s (4.852) | 24.768 s | 7.54 / 7.56 / 7.63 |

| GPUs | build: SuperNeuroABM | NEST GPU, same files | NEST GPU, own script |
|---|---|---|---|
| 8 | 194 s (load 99.9, setup 4.1, first tick 90.1) | 61 s (read 8.8, boundary 8.0, local 1.4, edge exchange 5.5, remote 36.4 in 43,339 calls, calibrate 0.8) | 0.32 s |
| 16 | 196 s (load 94.3, setup 4.3, first tick 97.6) | 64 s (remote 36.2 in 48,241 calls, boundary 10.6, edge exchange 7.0) | 0.33 s |
| 32 | 220 s (load 110.5, setup 4.7, first tick 104.7) | 65 s (remote 36.1 in 52,525 calls, boundary 11.3, edge exchange 6.9) | 0.36 s |
| 64 | 230 s (load 101.3, setup 5.5, first tick 122.9) | 65 s (remote 36.0 in 57,181 calls, boundary 11.2, edge exchange 6.7) | 0.43 s |
| 128 | 267 s (load 109.9, setup 9.2, first tick 147.7) | 69 s (remote 37.8 in 63,218 calls, boundary 13.2, edge exchange 6.8) | 0.51 s |
| 256 | 329 s (load 120.3, setup 12.8, first tick 195.9) | 67 s (remote 36.8 in 71,739 calls, boundary 11.9, edge exchange 7.1) | 0.57 s |
| 512 | 1,225 s (load 155.1, setup 40.4, first tick 1,029.7) | 73 s (remote 39.8 in 84,062 calls, boundary 13.7, edge exchange 7.7) | 0.73 s |

Build is load + setup + first tick for SuperNeuroABM, read + boundary + create + connect +
calibrate for NEST GPU, construct + calibrate for its own script. The whole run, start to
last tick, took 254 / 310 / 415 / 1,406 s for SuperNeuroABM and 93 / 99 / 102 / 119 s for
NEST GPU on the same files at 64 / 128 / 256 / 512 GPUs. Setting the starting potentials in
SuperNeuroABM (one get and one set per soma) is a further 33 to 86 s on the slowest rank
and is in those totals, not in "build".

Spikes, SuperNeuroABM against NEST GPU on the same files: identical for 62.0 ms at 8 GPUs,
116.6 ms at 16, 27.8 ms at 32, 6.3 ms at 64, 11.2 ms at 128, 4.1 ms at 256 and 3.2 ms at 512.
The first difference is each time one soma firing one tick apart in the two simulators;
after it the runs separate, with the rates within 1 to 3.5 % up to 32 GPUs and within 0.1 to
0.7 % from 64 on. From 8 GPUs on, hpc_benchmark.py's own rate (8.5, 7.9, 7.9, 7.8, 7.7, 7.6,
7.6 Hz) agrees with the file-driven runs; at 2 and 4 processes it gives 52 and 26 Hz, in the
dense regime its docstring warns about (unstable below scale 10), and 1 process segfaults.

Per rank: SuperNeuroABM 51 to 52 GiB host and 20.2 to 21.9 GiB GPU at every size. NEST GPU
11 to 12 GiB host up to 256 GPUs and 13.7 at 512; its GPU memory grows with the rank count: 2.3 to
2.9 GiB up to 32 GPUs, then 3.4 / 4.5 / 6.6 / 10.9 GiB at 64 / 128 / 256 / 512.

### What the 1 to 512 GPU runs say

- **Run time.** One GPU: SuperNeuroABM 1.75 times faster. Several GPUs: NEST GPU on the same
  files is faster, by 1.25 times at 2 GPUs, 1.8 at 8, 4.0 at 32, 5.6 at 64, 7.3 at 128, 9.2
  at 256 and 10.3 at 512.
- **Both ticks grow with the number of peer ranks.** At this size every rank listens to
  every other. NEST GPU's tick fits 0.45 ms + 0.0086 ms per peer rank within 4 % from 8 to
  512 GPUs. SuperNeuroABM's costs 0.086 to 0.097 ms per peer rank from 16 to 512 GPUs: ten
  times NEST GPU's price per peer. NEST GPU sends each peer one message per step holding the
  ids of the neurons that fired (`mpi_comm.cu`, `MPI_Isend` / `MPI_Irecv` per host, then
  `MPI_Waitall`); SAGESim sends each peer its ghost somas' state every tick. At 32 GPUs that
  exchange is half of SuperNeuroABM's tick; the other half is the kernel's own GPU time,
  which also grows with the ghost somas ("Where a tick goes" below).
- **NEST GPU's own script is slower than NEST GPU on our files from 64 GPUs on** (1.5 to 2.0
  times; it is faster up to 32). Known difference: hpc_benchmark.py puts all ranks in one
  host group, which makes NEST GPU exchange spikes with `MPI_Allgather` / `MPI_Allgatherv`,
  and our runner calls `RemoteConnect` without a host group, which makes it point to point.
  Whether that is the whole gap is not tested. So reading the network from files does not
  handicap NEST GPU's run time; it costs it the build (65 to 73 s against under 1 s).
- **Build.** NEST GPU from files stays at 65 to 73 s. SuperNeuroABM grows from 194 s at 8
  GPUs to 329 s at 256, and to 1,225 s at 512, where the first tick alone takes 1,030 s (196 s
  at 256) and setup 40 s (13 s at 256). What the first tick spends that time on at 512 ranks
  is not measured yet.

### Where a tick goes, 2 and 32 GPUs

SuperNeuroABM, SAGESim's timers, ms per tick, mean over the timed ticks (jobs 61323573 and
61334376; `--tick-timers`). One GPU runs all ticks in one launch at 0.09 ms per tick.

| | 2 GPUs | 32 GPUs |
|---|---|---|
| tick | 0.633 | 2.848 |
| ghost exchange (`data_prep`) | 0.376 | 1.279 |
| the kernel on the GPU (`gpu_sync`: the wait for the launched kernel) | 0.117 | 1.411 |
| the rest: launch, arguments, read-backs | 0.140 | 0.158 |

With GPU buffers straight to MPI the 2-GPU tick is 0.60 ms. At 32 GPUs the exchange is 45 %
of the tick and the kernel 50 %. The kernel's GPU time is 12 times that at 2 GPUs with the
same local somas and synapses; what grew is the ghost somas (11,250 to 348,750). Which part
of the kernel works over the ghosts is not identified yet. Sending spikes as events would
remove the exchange half only.

NEST GPU, its own timers, ms per step, mean over the hosts (job 61334376; `--kernel-timers`).
They cannot be read or reset after the warm-up, so they are over all 3,000 steps.

| | 2 GPUs | 32 GPUs |
|---|---|---|
| tick, timed part | 0.463 | 0.731 |
| `organizeExternalSpike` | 0.200 | 0.201 |
| `copy_ext_spike` (one integer from the GPU; probably where the step's GPU work is waited for) | 0.161 | 0.149 |
| `SendSpikeToRemote` (of which MPI) | 0.044 (0.008) | 0.178 (0.139) |
| `RecvSpikeFromRemote` | 0.067 | 0.506 |
| delivery, neuron update, copies, resets | 0.043 | 0.053 |

The receive timer includes the wait for the slowest rank in the first steps (the warm-up
takes 0.37 to 1.82 s depending on the rank), so it overstates the timed ticks: there the
receive is what the other phases leave, about 0.015 ms at 2 GPUs and 0.15 ms at 32. About
0.35 ms of NEST GPU's step is the same at 2 and 32 GPUs; the MPI send and receive are what
grow.

## The same benchmark on a 3D torus (2026-11-15, under way)

The owner's question: the all-to-all benchmark above is no weak-scaling test for
SuperNeuroABM, whose exchange costs per ghost soma; is the 3D torus of the Frontier
weak-scaling runs (`docs/BRUNEL_SCALING.md` 5.5a) better for it, and how do the two
simulators compare there? Same neuron, weights, replayed drive, starting potentials and size
(11,250 neurons and in-degree 11,250 per GPU, 126.6 M synapses); only the structure changes.

**The structure.** `make_brunel_partitions.py --topology torus3d --connection-radius 17`:
every rank is one 18 x 25 x 25 tile of neurons, the tiles form a periodic near-cube grid,
and a soma draws its 11,250 sources without repeats from the 20,478 neurons within radius 17
of it. Radius 17 because the library's default for this in-degree, 18, is not allowed on 1,
2 and 4 ranks (it must stay below the 18-neuron tile edge). A rank then listens to its
neighbour tiles only:

| GPUs | grid of tiles | peer ranks | remote somas a rank listens to | sources from its own rank |
|---|---|---|---|---|
| 1 | 1x1x1 | 0 | 0 | 100 % |
| 2 | 1x1x2 | 1 | 11,250 | 74.5 % |
| 4 | 1x2x2 | 3 | 33,580 | 55.0 % |
| 8 | 2x2x2 | 7 | 78,000 | 34.2 % |
| 16 | 2x2x4 | 11 | 87,380 | 34.2 % |
| 32 | 2x4x4 | 17 | 96,740 | 34.2 % |
| 64 | 4x4x4 | 26 | 124,100 | 34.2 % |
| 128, 256 | 4x4x8, 4x8x8 | 26 | 124,100 to 124,150 | 34.2 % |

Against 11,250 x (ranks - 1) on the all-to-all network: 708,750 at 64 GPUs, 5.75 M at 512.
NEST GPU gains from it too: it exchanges spikes only between host pairs that share
connections (`mpi_comm.cu`, `p2p_host_conn_matrix`).

**The firing rate does not survive the change of structure.** Three ways to draw a soma's
sources from its ball were run (`NET=` in the job scripts):

| `NET` | the draw | rate at 1 / 2 / 4 / 8 GPUs | SuperNeuroABM against NEST GPU |
|---|---|---|---|
| `library` | `brunel_partition`'s own torus3d: one pool of 11,250 | 0.44 / 0.39 / 0.38 / 0.72 Hz | identical spikes over the whole run at 1, 2 and 8 GPUs; at 4, one spike one tick apart at 3.0 ms and nothing else |
| `exact` | exactly 9,000 excitatory + 2,250 inhibitory (`--exact-ei`), inhibitory neurons where the library puts them | 22-24 / 108-137 / 102-129 / 560 Hz | identical for 26.0 / 14.5 / 3.2 / 7.4 ms |
| `mixed` | the same exact draw, inhibitory neurons spread evenly over the tile (`--interleave-ei`) | - / - / - / 45-48 Hz; 38-45 Hz at 16, 32-34 Hz at 64 | identical for 15.7 ms at 8 GPUs, 39.7 ms at 16, 3.7 ms at 64 |

- `library`: the library puts a rank's inhibitory neurons last in its id block, which its
  positions turn into one slab of the tile (x layers 15 to 17 of 18). With one pool per ball
  an excitatory soma in the middle of its slab gets 2,590 inhibitory sources and an
  inhibitory soma 1,800 (per soma 1,704 to 2,676 measured, 2,250 in the benchmark). The
  inhibitory neurons fire at 2 to 3 Hz and hold the excitatory ones silent (0.00 to 0.08 Hz
  per layer; job 61344221). This is the wiring of the Frontier weak-scaling campaign, so a
  firing campaign there needs a look at this too.
- `exact`: every soma has the benchmark's in-degrees, and the network saturates: at 8 GPUs
  a part of the excitatory somas fire at up to 1,588 Hz (the refractory limit is 2,000) and
  the mean is 560 Hz (job 61344538).
- `mixed`: fires at 45 to 48 Hz in bursts (spikes per 50 ms window between 125,000 and
  273,000 at 8 GPUs; job 61345199). Not the 7.5 Hz of the all-to-all network: at this
  in-degree a soma takes 55 % of its ball, a local density the all-to-all network has only
  at 2 GPUs.

The 64 to 512 GPU runs use `mixed` (it fires, with the benchmark's in-degrees); `library`
runs to 64 GPUs as the near-silent reference with spike-for-spike agreement.

**Numbers so far** (jobs 61344221 and 61344536 for 1 to 4 GPUs; 61344538 `exact` at 8;
61345199 / 61345200 `mixed` / `library` at 8; 61345636 / 61345637 at 16; 61345640 / 61345641
at 64). Timed 250 ms, ms per tick, SuperNeuroABM / NEST GPU, with the rate each measured;
the last column is the all-to-all network from "Numbers so far" above.

| GPUs | peers | `library` | `mixed` | `exact` | all-to-all |
|---|---|---|---|---|---|
| 1 | 0 | 0.052 / 0.157 (0.44 Hz) | | 0.103 / 0.167 (22.5 / 24.3 Hz) | 0.091 / 0.161 (9.8 Hz) |
| 2 | 1 | 0.530 / 0.419 (0.39 Hz) | | 0.862 / 0.518 (108 / 137 Hz) | 0.577 / 0.462 |
| 4 | 3 | 0.713 / 0.449 (0.38 Hz) | | 1.070 / 0.556 (102 / 129 Hz) | 0.726 / 0.490 |
| 8 | 7 | 0.927 / 0.505 (0.72 Hz) | 1.177 / 0.519 (48.0 / 45.5 Hz) | 2.569 / 0.913 (560 Hz) | 0.933 / 0.527 |
| 16 | 11 | 0.948 / 0.538 (0.77 Hz) | 1.307 / 0.613 (38.2 / 45.1 Hz) | | 1.455 / 0.594 |
| 32 | 17 | 1.194 / 0.585 (0.79 / 0.80 Hz) | 1.618 / 0.707 (41.7 / 40.0 Hz) | | 2.937 / 0.732 |
| 64 | 26 | 1.298 / 0.655 (0.74 / 0.73 Hz) | 1.992 / 0.828 (32.2 / 34.3 Hz) | | 5.718 / 1.025 |
| 128 | 26 | pending (61346213) | 2.336 / 0.870 (34.2 / 33.3 Hz) | | 10.932 / 1.513 |
| 256 | 26 | cancelled | timed out in the first tick / 0.898 (33.2 Hz) | | 23.315 / 2.552 |
| 512 | 26 | cancelled | cancelled | | 49.579 / 4.852 |

Jobs 61345638 / 61345639 (32 GPUs), 61345642 (128), 61345643 (256). On `mixed`, 64 to 128
GPUs is not flat for SuperNeuroABM: 1.992 to 2.336 ms, 17 % more with the same 26 peers and
124,100 ghost somas (the rates differ too, 32.2 and 34.2 Hz). NEST GPU: 0.828, 0.870, 0.898 ms
at 64, 128, 256 GPUs, 5 and 3 % more per doubling. NEST GPU is 2.4 to 2.7 times faster at
64 and 128 GPUs.

**SuperNeuroABM's first tick grows with the rank count at the same work per rank, and it
is what stops the weak-scaling runs.** First tick, `mixed`, slowest rank: 73.7 / 78.8 /
78.2 / 80.4 s at 8 / 16 / 32 / 64 GPUs, then 309.4 s at 128, and at 256 it was still
running when the 960 s limit of the run killed it, 801 s after it began (job 61345643:
"drive drawn" at 08:18:15, killed at 08:31:36). Peers and ghost somas are the same from 64
on, so it is not the halo. It is not the file system either: the 32-GPU job ran its first
tick in the same minutes, on the same directories, in 78 s. On the all-to-all network the
first tick was 123 / 148 / 196 / 1,030 s at 64 / 128 / 256 / 512 GPUs, so the torus is slower
here at 128 and 256 GPUs although it has a fifth to a twenty-third of the ghost somas. The
whole SuperNeuroABM run at 128 GPUs took 421 s (171 s at 32); NEST GPU's 81 / 90 / 96 s at
32 / 128 / 256 GPUs. Nothing in `CommunicationManager.build_communication_maps` or the
ghost discovery loops over the ranks in an obvious way (read 2026-11-15); the phase is
being measured: `run_partitioned_sna.py --first-tick-timers` turns SAGESim's timers on for
that tick and prints each phase (`perlmutter/torus_first_tick.sbatch`, jobs 61347856 at 64
GPUs and 61347857 at 128, both pending). Also growing with the ranks, in the runner and
not in the library: the starting potentials, one collective `get_agent_property_value`
per soma (about 20 s at 64 GPUs, about 75 s at 256).

Because of it the 256-GPU `library` job (61346214) and both 512-GPU jobs (61345644 `mixed`,
61346215 `library`) were cancelled while pending: SuperNeuroABM would have timed out in
each, about 140 node-hours for three NEST GPU numbers. NEST GPU alone at those sizes is
about 4 minutes of the nodes (4 node-hours at 256 GPUs, 9 at 512) if it is wanted.

At 64 GPUs NEST GPU is 2.0 (`library`) to 2.4 (`mixed`) times faster, against 5.6 times on
the all-to-all network; SuperNeuroABM's tick is 1.3 to 2.0 ms instead of 5.7. NEST GPU is
the faster one at every size but one GPU. At 8 GPUs the torus and the all-to-all network
have the same peers and remote somas and the ticks are the same (0.927 and 0.933 ms).

Spikes on `library`: identical over the whole run at 1, 2 and 8 GPUs; one spike one tick
apart and nothing else at 4 (3.0 ms) and at 16 (two spikes, 43,770 each); identical for
60.9 ms at 64, then apart (169,726 against 168,509 spikes).

Build and whole run, slowest rank, seconds:

| GPUs, `NET` | SuperNeuroABM: load / setup / first tick / whole run | NEST GPU: remote connect (calls per rank) / whole run |
|---|---|---|
| 8, library | 64.1 / 4.2 / 71.4 / 146 | 28.3 (75,535) / 67 |
| 8, mixed | 73.6 / 4.1 / 73.7 / 160 | 29.1 (76,895) / 70 |
| 16, library | 70.7 / 4.5 / 74.6 / 156 | 32.5 (110,094) / 74 |
| 16, mixed | 77.2 / 4.2 / 78.8 / 169 | 32.5 (105,994) / 74 |
| 64, library | 72.4 / 5.4 / 80.3 / 167 | 41.4 (183,586) / 84 |
| 64, mixed | 74.1 / 5.2 / 80.4 / 174 | 46.8 (221,606) / 89 |

On the all-to-all network at 64 GPUs SuperNeuroABM's whole run took 254 s (first tick
122.9 s) and NEST GPU's 93 s. Per rank on the torus: SuperNeuroABM 50.9 GiB host and 20.2 GiB
GPU, NEST GPU 9.7 to 10.0 GiB host and 2.5 (8, 16 GPUs) to 3.15 GiB (64) GPU.

Where a tick goes at 64 GPUs, `mixed` (job 61345640, `TIMERS=1`):

- SuperNeuroABM, timers on, 2.044 ms per tick: ghost exchange (`data_prep`) 1.435 ms, the
  kernel on the GPU (`gpu_sync`) 0.458 ms, the rest 0.15 ms. On the all-to-all network at 32
  GPUs (31 peers, 348,750 ghosts) these were 1.279 and 1.411 ms: the kernel's GPU time fell
  with the ghost somas, the exchange did not (26 peers against 31).
- NEST GPU, its own timers over all 3,000 steps (`torus_mixed_11250_r17/nest_timers_np64.txt`),
  ms per step, mean over the hosts: `organizeExternalSpike` 0.203, `copy_ext_spike` 0.161,
  `SendSpikeToRemote` 0.134 (MPI 0.097), `RecvSpikeFromRemote` 0.506 (includes the wait for
  the slowest rank during the warm-up), delivery 0.021, neuron update 0.010; timed tick
  0.825 ms.

**In the queue** at 2026-11-15 09:15 (outputs `torus-scale-<job>.out` and
`torus-first-tick-<job>.out` in the work directory; parts and spikes in
`torus_mixed_11250_r17/`, `torus_11250_r17/` (library), `torus_ei_11250_r17/` (exact)):

| job | queue | nodes (GPUs) | what |
|---|---|---|---|
| 61346213 | regular | 32 (128) | weak, `library`, `TIMERS=1` (the per-tick breakdown of both simulators on the near-silent network); about 20 minutes with two SuperNeuroABM builds of 7 minutes |
| 61347856 | regular | 16 (64) | SuperNeuroABM's first tick by phase, `mixed` |
| 61347857 | regular | 32 (128) | the same at 128 GPUs |
| 61346439, 61346441, 61346443 | debug, regular, regular | 8, 32, 64 (32, 128, 256) | strong scaling, below |

**To resume:** `sacct -X -S 2026-11-15T07:30 --format=JobID,JobName%18,NNodes,State,Elapsed,Start`
for the state of all of them; read a `torus-scale` output with
`grep -vE "^\[Rank|^\[TIMING\]|^\s+\[|Killed" torus-scale-<job>.out` (a `TIMERS=1` output
from before 08:20 holds one line per rank per tick, 30 MB); fill the pending cells. First
the two first-tick jobs: which phase grows from 64 to 128 ranks is the lead for the owner,
who wants these runs as input for optimizing SuperNeuroABM. The strong-scaling 128 and 256
GPU jobs give the first tick at 720 and 360 neurons per rank, which says whether the growth
depends on the work per rank. A job that fails is resubmitted from the work directory with
`sbatch -N <nodes> [-q regular -t 00:30:00] --export=ALL,NET=<net>[,STRONG=92160][,TIMERS=1] torus_scale.sbatch`;
the parts already written are reused.

### Strong scaling on the torus (submitted 2026-11-15 08:10; 8, 16 and 64 GPUs done)

Everything above is weak scaling (11,250 neurons on every GPU). The owner asked for strong
scaling as well. One fixed network on 8 to 256 GPUs:

- 92,160 neurons, in-degree 9,000 + 2,250 (1.04 G synapses), the same neuron, weights, drive
  and starting potentials, on one 40 x 48 x 48 torus, radius 17, the `mixed` draw.
- It is drawn once, as 512 tiles of 180 neurons (5 x 6 x 6), and a rank holds a brick of
  tiles (`make_brunel_partitions.py --drawn-as 512 --neurons 180`; `STRONG=92160` in
  `torus_scale.sbatch`; parts in `torus_mixed_11250_r17_strong92160/`). Every soma, synapse
  and weight is the same at every rank count: checked at small size on a login node (1 to 64
  ranks hold the identical network; post-owned; remote lists right). So the spikes of one
  simulator should agree across rank counts, up to rounding.
- 8 GPUs is the smallest size: 11,520 neurons and 129.6 M synapses per GPU fill
  SuperNeuroABM's host memory (about 52 of 62 GiB per rank). From there the tiles shrink
  under the radius: 5,760 / 2,880 / 1,440 / 720 / 360 neurons per GPU at 16 / 32 / 64 / 128 /
  256 GPUs, with more peer ranks each time. 512 GPUs (180 neurons each) was left out: about
  17 node-hours for the far end of the curve.
- What to expect: at this in-degree a soma's ball (20,478 neurons) is larger than a GPU's
  share of the network from 16 GPUs on, so the remote somas per GPU do not shrink with the
  local ones. No speed-up is expected for SuperNeuroABM, whose tick at 64 GPUs is already
  70 % exchange; the runs measure how fast the tick grows with the peers, for both simulators.

| job | queue | nodes (GPUs) | neurons per GPU | per-tick timers |
|---|---|---|---|---|
| 61346437 | debug | 2 (8) | 11,520 | yes |
| 61346438 | debug | 4 (16) | 5,760 | yes |
| 61346439 | debug | 8 (32) | 2,880 | yes |
| 61346440 | regular | 16 (64) | 1,440 | yes |
| 61346441 | regular | 32 (128) | 720 | yes |
| 61346443 | regular | 64 (256) | 360 | no |

Results so far (jobs 61346437, 61346438, 61346440). This network fires at 7.9 Hz in
SuperNeuroABM and 7.6 Hz in NEST GPU, close to the all-to-all benchmark's 7.5 Hz and unlike
the weak-scaling `mixed` network (32 to 48 Hz, in bursts) although the draw is the same;
the difference in the wiring is where the inhibitory neurons sit (every fifth position of a
5 x 6 x 6 tile is a diagonal lattice, of an 18 x 25 x 25 tile every fifth z layer).

| GPUs | neurons per GPU | peer ranks | remote somas per GPU | tick: SuperNeuroABM / NEST GPU (ms) | against 8 GPUs | whole run (s) | SuperNeuroABM's first tick (s) |
|---|---|---|---|---|---|---|---|
| 8 | 11,520 | 7 | 80,141 | 1.055 / 0.537 | 1 / 1 | 177 / 72 | 79.8 |
| 16 | 5,760 | 15 | 74,291 | 0.930 / 0.466 | 1.13 / 1.15 times faster | 79 / 44 | 37.9 |
| 32 | 2,880 | | | pending (61346439) | | | |
| 64 | 1,440 | 53 | 57,170 | 1.141 / 0.631 | 0.92 / 0.85 | 27 / 25 | 11.6 |
| 128 | 720 | | | pending (61346441) | | | |
| 256 | 360 | | | pending (61346443) | | | |

Neither simulator's tick gets faster beyond 16 GPUs: the remote somas per GPU stay at
57,000 to 80,000 while the local ones fall eightfold, and the peers go from 7 to 53.
SuperNeuroABM's tick with its timers on, ms: ghost exchange 0.873 / 0.674 / 0.968 and the
kernel on the GPU 0.271 / 0.227 / 0.166 at 8 / 16 / 64 GPUs, of 1.309 / 1.052 / 1.278. The
exchange is two thirds to three quarters of the tick at every size, and the kernel's GPU
time falls by 39 % for an eighth of the local synapses. Per rank at 8 / 16 / 64 GPUs:
SuperNeuroABM 52.0 / 26.3 / 7.1 GiB host and 20.7 / 10.6 / 3.1 GiB GPU; NEST GPU 10.1 / 5.5 /
1.8 GiB host and 2.5 / 1.6 / 0.9 GiB GPU. What strong scaling does shorten is the build:
SuperNeuroABM's whole run 177 to 27 s, NEST GPU's 72 to 25 s.

Spikes (the network is the same at every rank count): NEST GPU gives the same 205,580
spikes at 8, 16 and 64 GPUs, spike for spike. SuperNeuroABM gives the same 214,154 spikes
at 8 and 16 GPUs and in a second run at 8; at 64 GPUs it is identical for 171.1 ms and then
apart (214,308 spikes). SuperNeuroABM against NEST GPU: identical for 31.9 ms at each size.

Still to do when the last three are done: their rows, NEST GPU's own timers
(`nest_timers_np<N>.txt` in the parts directory), and the comparison across rank counts
again: `python compare_partitioned_spikes.py <dir>/spikes_sna_np8_vm.npz <dir>/spikes_sna_np32_vm.npz ... --burn-in-ms 50 --window-ms 50`
(sna-dev environment, login node), the same for `spikes_nestgpu_np*_vm.npz`.

`compare_partitioned_spikes.py` now compares sorted integer keys (4 s for 2 x 12.7 M spikes,
44 s before with Python sets); same output on six earlier pairs.

## Open

- The torus comparison: the queued jobs above, then the pending cells.
- SuperNeuroABM's first tick against the rank count (80 s at 64 GPUs, 309 s at 128, over
  800 s at 256 on the torus): find the phase, then the cause. It decides whether weak scaling
  beyond 128 GPUs can be measured at all.
- SuperNeuroABM's tick on `mixed` is 17 % longer at 128 GPUs than at 64 with the same peers
  and ghosts: not explained.
- Why the strong-scaling network fires at 7.6 to 7.9 Hz and the weak-scaling `mixed` one at
  32 to 48 Hz with the same draw.
- A torus network that fires near the benchmark's 7.5 Hz was not found by changing the
  structure alone (0.4 to 0.7, 45 to 48 or 560 Hz). Whether to tune the drive or the
  in-degree for it is the owner's call.

- SuperNeuroABM's kernel takes 1.41 ms of GPU time per tick at 32 ranks against 0.12 ms at
  2: find what in it works over the ghost somas.
- SuperNeuroABM's first tick at 512 ranks (1,030 s) and its setup there (40 s): find what
  grows.
- NEST GPU with a host group in our runner, to test whether collective MPI is why its own
  script is slower from 64 GPUs on.
- The 128-node run used 31 of its 45 minutes, 24 of them in SuperNeuroABM. Host memory is at
  52 of about 62 GiB per rank.
- The network files take 3.9 TB of scratch (`hpc_11250/np<N>`, 4 GB per rank; np512 alone is
  2.0 TB). They are rewritten in about a minute by `hpc_scale.sbatch` if deleted.
- The repo's test gate (pytest and the MPI tests) has not been run for these commits; they
  touch tutorials and the comparison scripts, not the library.
