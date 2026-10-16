# Per-tick probes

The scripts behind `docs/EVENT_DRIVEN_NOTES.md` sections 15 and 16: where a tick's time
goes on one GPU (fixed cost) and across ranks (launch path, ghost exchange), with the
same network at every rank count so spike trains can be compared spike for spike.

| script | measures |
|---|---|
| `fixed_cost_probe.py` | silent-network fixed cost on one GPU by variant: floor (barriers + loop), somas (+ soma phase), ring (+ wake-expansion scan); fused vs one launch per tick |
| `split_partition.py` | the single-rank columnar Brunel partition split by post into N post-owned partitions (identical network at every N) |
| `multirank_tick.py` | per-tick timer breakdown at N ranks (launch, GPU, pack, MPI wait, unpack, reverse), partition-invariant drive, spikes saved for comparison |
| `compare_spikes.py` | per-spike comparison across rank counts: identical, or first differing tick |

## Commands (laptop: one GPU shared by N ranks, `--oversubscribe`)

```sh
PY=/home/xxz/miniforge3/envs/sna-dev/bin/python          # the env with both repos installed
cd scaling_analysis/per_tick_probes

# 1. fixed cost on one GPU (section 15)
$PY fixed_cost_probe.py floor 1 2000 fused
$PY fixed_cost_probe.py somas 12500 2000 fused
$PY fixed_cost_probe.py ring 12500 2000 fused
$PY fixed_cost_probe.py ring 12500 2000 unfused

# 2. the single-rank Brunel partition the nest_comparison benchmark writes
#    (12,500 neurons, K=1000; ~25 s to generate once)
$PY ../nest_comparison/brunel_sna_local.py --ticks 10 --keep-partition --no-learning-rules
SRC=../nest_comparison/partitions/1w_12500n_K1000_global_rauto/partition_0.npz

# 3. split it for 2, 3 and 4 ranks and time every rank count (section 16)
$PY split_partition.py $SRC parts 2 3 4
for n in 1 2 3 4; do mpirun --oversubscribe -n $n $PY multirank_tick.py $SRC parts 400; done
$PY compare_spikes.py parts 1 2 3 4
```

On a cluster (one GPU per rank, GPU-aware MPI such as Cray MPICH with
`MPICH_GPU_SUPPORT_ENABLED=1`): drop `--oversubscribe`, keep the rest. The columns to
look at are `kernel_launch_overhead` (should be ~0.1 ms), `gpu_sync` (the kernel),
`mpi_gpu_pack` / `mpi_gpu_unpack` / `mpi_reverse` (should not grow with peers) and
`mpi_wait_time` (rank skew). Expected on the laptop after the 2026-10-15 round: 1 rank
0.60 ms per tick, 2 ranks 1.68, 3 ranks ~2.0, 4 ranks 2.43, with 156,183 spikes at every
rank count from 2 up (section 16 explains why 1 rank differs after tick 108).

Generated `step_func_code_*.py` files land in the working directory (gitignored).
