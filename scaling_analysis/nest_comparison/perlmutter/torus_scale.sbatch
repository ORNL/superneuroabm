#!/bin/bash
#SBATCH -A m5460_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-node=4
#SBATCH -t 00:30:00
#SBATCH -J torus-scale
#SBATCH -o /pscratch/sd/z/zhangx8/partitioned/torus-scale-%j.out
# The hpc_benchmark neurons, weights, drive and size (11,250 neurons and in-degree 11,250 per
# GPU) on the 3D torus of the weak-scaling runs, on N nodes (4 GPUs each, one rank per GPU):
#   sbatch -N 2 torus_scale.sbatch        (2, 4, 8 nodes fit the debug queue)
#   sbatch -N 64 -q regular torus_scale.sbatch
# Writes the parts for 4N ranks if missing (make_brunel_partitions.py --topology torus3d,
# every soma's sources from the ball of radius RADIUS around it), runs SuperNeuroABM and NEST
# GPU on them (random starting potentials, 50 ms warm-up, 250 ms timed) and compares their
# spikes. TIMERS=1 adds a second run of each with its own per-tick timers on.
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
NODES=$SLURM_JOB_NUM_NODES
n=$((4 * NODES))
RUN="--config $CFG --eta 1.685 --presim-ms 50 --ms 300 --vm-sigma 7.2"
SRUN="srun -N $NODES -n $n --ntasks-per-node=4 --cpu-bind=none"
cd $W
echo "== $NODES node(s), $n ranks, radius $RADIUS, net $NET, first node $(hostname)  ($(date +%H:%M:%S))"
NFILTER='^Building time|^Simulation time|^Calibrating|^Simulating|Neural activity|^\s*$|NEST GPU --|Copyright|licen|Homepage|^ *-{3,}|Host [0-9]+ :|AS IS|WARRANTY'
SFILTER='Warning|experimental|cupy._util|warnings.warn'
source /global/cfs/cdirs/m5460/envs/sna-dev.sh
echo "===== partitions  ($(date +%H:%M:%S))"
MPICH_GPU_SUPPORT_ENABLED=0 timeout -k 15 ${GEN_TIMEOUT:-600} $SRUN python -u $NC/make_brunel_partitions.py $P \
    --ranks $n --neurons 11250 --excitatory-in-degree 9000 --inhibitory-in-degree 2250 --config $CFG \
    --topology torus3d --connection-radius $RADIUS $EI 2>&1 \
  | grep --line-buffered -vE "$SFILTER" | sed "s|$P/||" | sort -V | sed -n '1p;$p'
echo "  -> exit ${PIPESTATUS[0]};  $(ls $P/np$n 2>/dev/null | wc -l) parts, $(du -sh $P/np$n 2>/dev/null | cut -f1)"
echo "===== SuperNeuroABM np$n, host-staged MPI  ($(date +%H:%M:%S))"
MPICH_GPU_SUPPORT_ENABLED=0 timeout -k 15 ${SNA_TIMEOUT:-960} $SRUN --gpus-per-task 1 \
    python -u $NC/run_partitioned_sna.py $P $RUN --out $P/spikes_sna_np${n}_vm.npz 2>&1 | grep --line-buffered -vE "$SFILTER"
echo "  -> exit ${PIPESTATUS[0]} (124 = timed out)  ($(date +%H:%M:%S))"
if [ "${TIMERS:-0}" = 1 ]; then
  echo "===== SuperNeuroABM np$n, per-tick timers on  ($(date +%H:%M:%S))"
  MPICH_GPU_SUPPORT_ENABLED=0 timeout -k 15 ${SNA_TIMEOUT:-960} $SRUN --gpus-per-task 1 \
      python -u $NC/run_partitioned_sna.py $P $RUN --tick-timers --out $P/spikes_sna_np${n}_vm_timers.npz 2>&1 \
    | grep --line-buffered -vE "$SFILTER|^\s+\[|^\[Rank|^\[TIMING"
  echo "  -> exit ${PIPESTATUS[0]} (124 = timed out)  ($(date +%H:%M:%S))"
fi
conda deactivate
source /global/cfs/cdirs/m5460/envs/nestgpu.sh
echo "===== NEST GPU np$n  ($(date +%H:%M:%S))"
timeout -k 15 ${NEST_TIMEOUT:-600} $SRUN --gpus-per-task 1 \
    python -u $NC/run_partitioned_nestgpu.py $P $RUN --out $P/spikes_nestgpu_np${n}_vm.npz 2>&1 | grep --line-buffered -vE "$NFILTER"
echo "  -> exit ${PIPESTATUS[0]} (124 = timed out)  ($(date +%H:%M:%S))"
echo "===== compare np$n"
[ -f $P/spikes_sna_np${n}_vm.npz ] && [ -f $P/spikes_nestgpu_np${n}_vm.npz ] && \
  python $NC/compare_partitioned_spikes.py $P/spikes_sna_np${n}_vm.npz $P/spikes_nestgpu_np${n}_vm.npz --burn-in-ms 50 --window-ms 50
if [ "${TIMERS:-0}" = 1 ]; then
  echo "===== NEST GPU np$n, its own timers on (seconds since the start, per host)  ($(date +%H:%M:%S))"
  timeout -k 15 ${NEST_TIMEOUT:-600} $SRUN --gpus-per-task 1 \
      python -u $NC/run_partitioned_nestgpu.py $P $RUN --kernel-timers --out $P/spikes_nestgpu_np${n}_vm_timers.npz > $P/nest_timers_np$n.txt 2>&1
  echo "  -> exit $? (124 = timed out); output in $P/nest_timers_np$n.txt  ($(date +%H:%M:%S))"
  grep -E "timed simulate|rate after" $P/nest_timers_np$n.txt
fi
echo "== done ($(date +%H:%M:%S))"
