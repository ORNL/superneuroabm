#!/bin/bash
#SBATCH -A m5460_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-node=4
#SBATCH -t 00:30:00
#SBATCH -J hpc-scale
#SBATCH -o /pscratch/sd/z/zhangx8/partitioned/hpc-scale-%j.out
# NEST GPU's hpc_benchmark network at scale 1 on N nodes (4 GPUs each, one rank per GPU):
#   sbatch -N 2 hpc_scale.sbatch        (2, 4, 8 nodes fit the debug queue)
# Writes the parts for 4N ranks if missing, runs SuperNeuroABM and NEST GPU on them (random
# starting potentials, 50 ms warm-up, 250 ms timed), compares their spikes, then runs NEST
# GPU's own hpc_benchmark.py at the same rank count.
W=/pscratch/sd/z/zhangx8/partitioned
P=$W/hpc_11250
NC=/global/cfs/cdirs/m5460/zhangx8/superneuroabm/scaling_analysis/nest_comparison
HPC=/global/cfs/cdirs/m5460/zhangx8/software/nest-gpu/src/python/examples/hpc_benchmark/hpc_benchmark.py
CFG=$NC/hpc_alpha_config.yaml
NODES=$SLURM_JOB_NUM_NODES
n=$((4 * NODES))
RUN="--config $CFG --eta 1.685 --presim-ms 50 --ms 300 --vm-sigma 7.2"
SRUN="srun -N $NODES -n $n --ntasks-per-node=4 --cpu-bind=none"
cd $W
echo "== $NODES node(s), $n ranks, first node $(hostname)  ($(date +%H:%M:%S))"
NFILTER='^Building time|^Simulation time|^Calibrating|^Simulating|Neural activity|^\s*$|NEST GPU --|Copyright|licen|Homepage|^ *-{3,}|Host [0-9]+ :|AS IS|WARRANTY'
SFILTER='Warning|experimental|cupy._util|warnings.warn'
source /global/cfs/cdirs/m5460/envs/sna-dev.sh
echo "===== partitions  ($(date +%H:%M:%S))"
MPICH_GPU_SUPPORT_ENABLED=0 timeout -k 15 600 $SRUN python -u $NC/make_brunel_partitions.py $P \
    --ranks $n --neurons 11250 --excitatory-in-degree 9000 --inhibitory-in-degree 2250 --config $CFG 2>&1 \
  | grep --line-buffered -vE "$SFILTER" | tail -3
echo "  -> exit ${PIPESTATUS[0]};  $(ls $P/np$n 2>/dev/null | wc -l) parts, $(du -sh $P/np$n 2>/dev/null | cut -f1)"
echo "===== SuperNeuroABM np$n, host-staged MPI  ($(date +%H:%M:%S))"
MPICH_GPU_SUPPORT_ENABLED=0 timeout -k 15 ${SNA_TIMEOUT:-960} $SRUN --gpus-per-task 1 \
    python -u $NC/run_partitioned_sna.py $P $RUN --out $P/spikes_sna_np${n}_vm.npz 2>&1 | grep --line-buffered -vE "$SFILTER"
echo "  -> exit ${PIPESTATUS[0]} (124 = timed out)  ($(date +%H:%M:%S))"
conda deactivate
source /global/cfs/cdirs/m5460/envs/nestgpu.sh
echo "===== NEST GPU np$n  ($(date +%H:%M:%S))"
timeout -k 15 420 $SRUN --gpus-per-task 1 \
    python -u $NC/run_partitioned_nestgpu.py $P $RUN --out $P/spikes_nestgpu_np${n}_vm.npz 2>&1 | grep --line-buffered -vE "$NFILTER"
echo "  -> exit ${PIPESTATUS[0]} (124 = timed out)  ($(date +%H:%M:%S))"
echo "===== compare np$n"
[ -f $P/spikes_sna_np${n}_vm.npz ] && [ -f $P/spikes_nestgpu_np${n}_vm.npz ] && \
  python $NC/compare_partitioned_spikes.py $P/spikes_sna_np${n}_vm.npz $P/spikes_nestgpu_np${n}_vm.npz --burn-in-ms 50 --window-ms 50
D=$W/nest_own_hpc/scale1_np$n; mkdir -p $D
echo "===== NEST GPU's own hpc_benchmark.py np$n  ($(date +%H:%M:%S))"
timeout -k 15 240 $SRUN --gpus-per-task 1 python -u $HPC --path $D --scale 1 --simtime 250 --verbosity 0 > $D/stdout.txt 2>&1
echo "  -> exit $? (124 = timed out)"
python - $D $n <<'EOF'
import json, sys, glob
d, n = sys.argv[1], int(sys.argv[2])
logs = [json.load(open(f)) for f in sorted(glob.glob(d + "/log_*.json"))]
if len(logs) != n:
    print(f"  {len(logs)} of {n} rank logs found; last lines of stdout:")
    print("".join(open(d + "/stdout.txt").readlines()[-8:]))
else:
    for key in ("time_construct", "time_calibrate", "time_presimulate", "time_simulate", "time_total"):
        v = [l["timers"][key] / 1e9 for l in logs]
        print(f"  {key:18s} max {max(v):8.3f} s   min {min(v):8.3f} s")
    if "stats" in logs[0]:
        e = [l["stats"]["excitatory_firing_rate"] for l in logs]; i = [l["stats"]["inhibitory_firing_rate"] for l in logs]
        print(f"  firing rate: E {sum(e)/len(e):.3f} Hz  I {sum(i)/len(i):.3f} Hz  (mean over ranks)")
EOF
echo "== done ($(date +%H:%M:%S))"
