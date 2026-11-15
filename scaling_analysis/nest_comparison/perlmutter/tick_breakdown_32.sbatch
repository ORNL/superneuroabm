#!/bin/bash
#SBATCH -A m5460_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 8
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-node=4
#SBATCH -t 00:30:00
#SBATCH -J tick-breakdown-32
#SBATCH -o /pscratch/sd/z/zhangx8/partitioned/tick-breakdown-32-%j.out
# Where a tick goes at 32 ranks (8 nodes) on the hpc_benchmark network, in each simulator's
# own timers, and the rounding control for NEST GPU. Needs hpc_11250/np32 and np2 and the
# spikes of an earlier 32-rank run (hpc_scale.sbatch on 8 nodes).
#   1. NEST GPU, 32 ranks, its own per-phase timers; spikes against the earlier run
#      (is NEST GPU reproducible run to run?)
#   2. NEST GPU, 32 ranks, excitatory weights x (1 + 1e-6); spikes against the earlier run
#      (how fast does a rounding-size change separate two runs?)
#   3. NEST GPU, 2 ranks on one node, its own per-phase timers
#   4. SuperNeuroABM, 32 ranks, SAGESim's per-tick timers
W=/pscratch/sd/z/zhangx8/partitioned
P=$W/hpc_11250
NC=/global/cfs/cdirs/m5460/zhangx8/superneuroabm/scaling_analysis/nest_comparison
CFG=$NC/hpc_alpha_config.yaml
T=$W/timers; mkdir -p $T
RUN="--config $CFG --eta 1.685 --presim-ms 50 --ms 300 --vm-sigma 7.2"
SRUN="srun -N 8 -n 32 --ntasks-per-node=4 --cpu-bind=none --gpus-per-task 1"
REF=$P/spikes_nestgpu_np32_vm.npz
cd $W
echo "== 8 nodes, first node $(hostname)  ($(date +%H:%M:%S))"
SFILTER='Warning|experimental|cupy._util|warnings.warn'
KEEP='^NEST GPU, |^  [a-z_]+ +max|timed simulate|rate after|peak host|Error|error|Traceback'

# NEST GPU's timers: the last value each host printed, as ms per step over all 3000 steps
timers() {
  awk -v steps=3000 '
    /^Host +[0-9]+ *: +[A-Za-z_]+time: / { h = $2; k = $4; sub(":", "", k); v[k "," h] = $5; keys[k] = 1; hosts[h] = 1 }
    END {
      for (k in keys) {
        n = 0; sum = 0; max = 0
        for (h in hosts) if ((k "," h) in v) { x = v[k "," h] / steps * 1e3; sum += x; n++; if (x > max) max = x }
        if (n) printf "    %-36s max %9.4f   mean %9.4f   (ms per step, %d hosts)\n", k, max, sum / n, n
      }
    }' "$1" | sort
}

source /global/cfs/cdirs/m5460/envs/nestgpu.sh
echo "===== 1. NEST GPU np32, kernel timers  ($(date +%H:%M:%S))"
timeout -k 15 420 $SRUN python -u $NC/run_partitioned_nestgpu.py $P $RUN --kernel-timers \
    --out $P/spikes_nestgpu_np32_timers.npz > $T/nest_np32_$SLURM_JOB_ID.txt 2>&1
echo "  -> exit $? (124 = timed out)"
grep -E "$KEEP" $T/nest_np32_$SLURM_JOB_ID.txt
timers $T/nest_np32_$SLURM_JOB_ID.txt
[ -f $P/spikes_nestgpu_np32_timers.npz ] && [ -f $REF ] && \
  python $NC/compare_partitioned_spikes.py $REF $P/spikes_nestgpu_np32_timers.npz --burn-in-ms 50 --window-ms 50

echo "===== 2. NEST GPU np32, excitatory weights x (1 + 1e-6)  ($(date +%H:%M:%S))"
timeout -k 15 420 $SRUN python -u $NC/run_partitioned_nestgpu.py $P $RUN --perturb 1e-6 \
    --out $P/spikes_nestgpu_np32_p1e-6.npz 2>&1 | grep --line-buffered -E "$KEEP"
echo "  -> exit ${PIPESTATUS[0]} (124 = timed out)"
[ -f $P/spikes_nestgpu_np32_p1e-6.npz ] && [ -f $REF ] && \
  python $NC/compare_partitioned_spikes.py $REF $P/spikes_nestgpu_np32_p1e-6.npz --burn-in-ms 50 --window-ms 50

echo "===== 3. NEST GPU np2 on one node, kernel timers  ($(date +%H:%M:%S))"
timeout -k 15 300 srun -N 1 -n 2 --cpu-bind=none --gpus-per-task 1 \
    python -u $NC/run_partitioned_nestgpu.py $P $RUN --kernel-timers \
    --out $P/spikes_nestgpu_np2_timers.npz > $T/nest_np2_$SLURM_JOB_ID.txt 2>&1
echo "  -> exit $? (124 = timed out)"
grep -E "$KEEP" $T/nest_np2_$SLURM_JOB_ID.txt
timers $T/nest_np2_$SLURM_JOB_ID.txt
conda deactivate

source /global/cfs/cdirs/m5460/envs/sna-dev.sh
echo "===== 4. SuperNeuroABM np32, host-staged MPI, tick timers  ($(date +%H:%M:%S))"
MPICH_GPU_SUPPORT_ENABLED=0 timeout -k 15 720 $SRUN \
    python -u $NC/run_partitioned_sna.py $P $RUN --tick-timers --out $P/spikes_sna_np32_timers.npz 2>&1 \
  | grep --line-buffered -vE "$SFILTER|^\[Rank|^\[TIMING\]|^\s+\["
echo "  -> exit ${PIPESTATUS[0]} (124 = timed out)"
echo "== done ($(date +%H:%M:%S))"
