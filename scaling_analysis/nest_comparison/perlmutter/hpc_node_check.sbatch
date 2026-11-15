#!/bin/bash
#SBATCH -A m5460_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 32
#SBATCH --gpus 4
#SBATCH -t 00:30:00
#SBATCH -J hpc-node-check
#SBATCH -o /pscratch/sd/z/zhangx8/partitioned/hpc-node-check-%j.out
# NEST GPU's hpc_benchmark network at scale 1 (11,250 neurons and in-degree 11,250 on every
# rank), from partition files, on one node: write the parts for RANKS, run SuperNeuroABM and
# NEST GPU on them, compare their spikes.  RANKS="1 2" sbatch hpc_node_check.sbatch
W=/pscratch/sd/z/zhangx8/partitioned
P=$W/hpc_11250
NC=/global/cfs/cdirs/m5460/zhangx8/superneuroabm/scaling_analysis/nest_comparison
CFG=$NC/hpc_alpha_config.yaml
RANKS=${RANKS:-1 2}
RUN="--config $CFG --eta 1.685 --presim-ms 50 --ms 300"
cd $W
echo "== node $(hostname)  ranks $RANKS  ($(date +%H:%M:%S))"
NFILTER='^Building time|^Simulation time|^Calibrating|^Simulating|Neural activity|^\s*$|NEST GPU --|Copyright|licen|Homepage|^ *-{3,}|Host [0-9]+ :|AS IS|WARRANTY'
SFILTER='Warning|experimental|cupy._util|warnings.warn'
source /global/cfs/cdirs/m5460/envs/sna-dev.sh
echo "===== partitions  ($(date +%H:%M:%S))"
MPICH_GPU_SUPPORT_ENABLED=0 timeout -k 15 900 srun -n 4 --cpu-bind=none python -u $NC/make_brunel_partitions.py $P \
    --ranks $RANKS --neurons 11250 --excitatory-in-degree 9000 --inhibitory-in-degree 2250 --config $CFG 2>&1 \
  | grep --line-buffered -vE "$SFILTER"
echo "  -> exit ${PIPESTATUS[0]}"; du -sh $P/np* 2>/dev/null | tr '\n' ' '; echo
for n in $RANKS; do
  echo "===== SuperNeuroABM np$n  ($(date +%H:%M:%S))"
  MPICH_GPU_SUPPORT_ENABLED=0 timeout -k 15 480 srun -n $n --gpus-per-task 1 --cpu-bind=none \
      python -u $NC/run_partitioned_sna.py $P $RUN 2>&1 | grep --line-buffered -vE "$SFILTER"
  echo "  -> exit ${PIPESTATUS[0]} (124 = timed out)"
done
conda deactivate
source /global/cfs/cdirs/m5460/envs/nestgpu.sh
for n in $RANKS; do
  echo "===== NEST GPU np$n  ($(date +%H:%M:%S))"
  timeout -k 15 480 srun -n $n --gpus-per-task 1 --cpu-bind=none \
      python -u $NC/run_partitioned_nestgpu.py $P $RUN 2>&1 | grep --line-buffered -vE "$NFILTER"
  echo "  -> exit ${PIPESTATUS[0]} (124 = timed out)"
done
for n in $RANKS; do
  echo "===== compare np$n"
  [ -f $P/spikes_sna_np$n.npz ] && [ -f $P/spikes_nestgpu_np$n.npz ] && \
    python $NC/compare_partitioned_spikes.py $P/spikes_sna_np$n.npz $P/spikes_nestgpu_np$n.npz
done
echo "== done ($(date +%H:%M:%S))"
