#!/bin/bash
#SBATCH -A m5460_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 32
#SBATCH --gpus 4
#SBATCH -t 00:20:00
#SBATCH -J tick-breakdown
#SBATCH -o /pscratch/sd/z/zhangx8/partitioned/tick-breakdown-%j.out
# Where a multi-rank tick goes, in each simulator's own timers, on the hpc_benchmark network
# at 2 ranks (one node): SuperNeuroABM with host-staged MPI, SuperNeuroABM with GPU buffers
# straight to MPI (IPC, every GPU visible), NEST GPU.
W=/pscratch/sd/z/zhangx8/partitioned
P=$W/hpc_11250
NC=/global/cfs/cdirs/m5460/zhangx8/superneuroabm/scaling_analysis/nest_comparison
CFG=$NC/hpc_alpha_config.yaml
N=${N:-2}
RUN="--config $CFG --eta 1.685 --presim-ms 50 --ms 300 --vm-sigma 7.2"
cd $W
echo "== node $(hostname)  ranks $N  ($(date +%H:%M:%S))"
SFILTER='Warning|experimental|cupy._util|warnings.warn'
source /global/cfs/cdirs/m5460/envs/sna-dev.sh
echo "===== SuperNeuroABM np$N, host-staged MPI, tick timers  ($(date +%H:%M:%S))"
MPICH_GPU_SUPPORT_ENABLED=0 timeout -k 15 480 srun -n $N --gpus-per-task 1 --cpu-bind=none \
    python -u $NC/run_partitioned_sna.py $P $RUN --tick-timers --out $P/spikes_sna_np${N}_timers_host.npz 2>&1 \
  | grep --line-buffered -vE "$SFILTER|^\s+\["
echo "  -> exit ${PIPESTATUS[0]} (124 = timed out)"
echo "===== SuperNeuroABM np$N, GPU buffers to MPI (IPC), tick timers  ($(date +%H:%M:%S))"
timeout -k 15 480 srun -n $N --gpus $N --gpu-bind=none --cpu-bind=none $W/own_gpu_first.sh \
    python -u $NC/run_partitioned_sna.py $P $RUN --tick-timers --out $P/spikes_sna_np${N}_timers_gpu.npz 2>&1 \
  | grep --line-buffered -vE "$SFILTER|^\s+\["
echo "  -> exit ${PIPESTATUS[0]} (124 = timed out)"
conda deactivate
source /global/cfs/cdirs/m5460/envs/nestgpu.sh
echo "===== NEST GPU np$N, kernel timers  ($(date +%H:%M:%S))"
timeout -k 15 480 srun -n $N --gpus-per-task 1 --cpu-bind=none \
    python -u $NC/run_partitioned_nestgpu.py $P $RUN --kernel-timers --out $P/spikes_nestgpu_np${N}_timers.npz 2>&1 \
  | grep --line-buffered -E "_time:|NEST GPU, |simulate|timed|rate after|exit|Error|error"
echo "  -> exit ${PIPESTATUS[0]} (124 = timed out)"
echo "== done ($(date +%H:%M:%S))"
