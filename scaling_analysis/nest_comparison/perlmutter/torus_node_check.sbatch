#!/bin/bash
#SBATCH -A m5460_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 32
#SBATCH --gpus 4
#SBATCH -t 00:30:00
#SBATCH -J torus-node-check
#SBATCH -o /pscratch/sd/z/zhangx8/partitioned/torus-node-check-%j.out
# The hpc_benchmark neurons, weights, drive and size (11,250 neurons and in-degree 11,250 per
# GPU) with only the structure changed: the 3D torus of the weak-scaling runs, every soma's
# sources drawn from the ball of radius RADIUS around it (make_brunel_partitions.py
# --topology torus3d). One node, per rank count: SuperNeuroABM, NEST GPU, spike comparison.
W=/pscratch/sd/z/zhangx8/partitioned
RADIUS=${RADIUS:-17}
# NET picks how a soma's sources are drawn from its ball:
#   library  brunel_partition's own torus3d: one pool of 11,250 (excitatory neurons silent,
#            0.4 Hz, job 61344221)
#   exact    exactly 9,000 excitatory + 2,250 inhibitory, the inhibitory neurons in one slab of
#            every tile as brunel_partition places them (saturates: 560 Hz at 8 GPUs, job 61344538)
#   mixed    the same exact draw with the inhibitory neurons spread evenly over the tile
case "${NET:?NET=library, exact or mixed}" in
  library) P=$W/torus_11250_r$RADIUS; EI=;;
  exact)   P=$W/torus_ei_11250_r$RADIUS; EI="--exact-ei";;
  mixed)   P=$W/torus_mixed_11250_r$RADIUS; EI="--exact-ei --interleave-ei";;
  *) echo "NET=$NET?"; exit 2;;
esac
NC=/global/cfs/cdirs/m5460/zhangx8/superneuroabm/scaling_analysis/nest_comparison
CFG=$NC/hpc_alpha_config.yaml
RANKS=${RANKS:-1 2 4}
RUN="--config $CFG --eta 1.685 --presim-ms 50 --ms 300 --vm-sigma 7.2"
cd $W
echo "== node $(hostname)  ranks $RANKS  radius $RADIUS  net $NET  ($(date +%H:%M:%S))"
NFILTER='^Building time|^Simulation time|^Calibrating|^Simulating|Neural activity|^\s*$|NEST GPU --|Copyright|licen|Homepage|^ *-{3,}|Host [0-9]+ :|AS IS|WARRANTY'
SFILTER='Warning|experimental|cupy._util|warnings.warn'
source /global/cfs/cdirs/m5460/envs/sna-dev.sh
echo "===== partitions  ($(date +%H:%M:%S))"
MPICH_GPU_SUPPORT_ENABLED=0 timeout -k 15 900 srun -n 4 --cpu-bind=none python -u $NC/make_brunel_partitions.py $P \
    --ranks $RANKS --neurons 11250 --excitatory-in-degree 9000 --inhibitory-in-degree 2250 --config $CFG \
    --topology torus3d --connection-radius $RADIUS $EI 2>&1 \
  | grep --line-buffered -vE "$SFILTER" | sed "s|$P/||"
echo "  -> exit ${PIPESTATUS[0]}"
for n in $RANKS; do
  echo "===== SuperNeuroABM np$n  ($(date +%H:%M:%S))"
  MPICH_GPU_SUPPORT_ENABLED=0 timeout -k 15 480 srun -n $n --gpus-per-task 1 --cpu-bind=none \
      python -u $NC/run_partitioned_sna.py $P $RUN --out $P/spikes_sna_np${n}_vm.npz 2>&1 | grep --line-buffered -vE "$SFILTER|^\s+\["
  echo "  -> exit ${PIPESTATUS[0]} (124 = timed out)"
done
conda deactivate
source /global/cfs/cdirs/m5460/envs/nestgpu.sh
for n in $RANKS; do
  echo "===== NEST GPU np$n  ($(date +%H:%M:%S))"
  timeout -k 15 480 srun -n $n --gpus-per-task 1 --cpu-bind=none \
      python -u $NC/run_partitioned_nestgpu.py $P $RUN --out $P/spikes_nestgpu_np${n}_vm.npz 2>&1 | grep --line-buffered -vE "$NFILTER"
  echo "  -> exit ${PIPESTATUS[0]} (124 = timed out)"
done
for n in $RANKS; do
  echo "===== compare np$n"
  [ -f $P/spikes_sna_np${n}_vm.npz ] && [ -f $P/spikes_nestgpu_np${n}_vm.npz ] && \
    python $NC/compare_partitioned_spikes.py $P/spikes_sna_np${n}_vm.npz $P/spikes_nestgpu_np${n}_vm.npz --burn-in-ms 50 --window-ms 50
done
echo "== done ($(date +%H:%M:%S))"
