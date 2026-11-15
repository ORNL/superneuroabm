#!/bin/bash
#SBATCH -A m5460_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 32
#SBATCH --gpus 4
#SBATCH -t 00:20:00
#SBATCH -J sna-mpi-variants
#SBATCH -o /pscratch/sd/z/zhangx8/partitioned/sna-mpi-variants-%j.out
# SuperNeuroABM at 2 and 4 ranks under three MPI/GPU setups, to see which ones work on
# Perlmutter. Every run has a hard timeout so a hang cannot use up the job.
W=/pscratch/sd/z/zhangx8/partitioned
P=$W/brunel_alpha_12500
NC=/global/cfs/cdirs/m5460/zhangx8/superneuroabm/scaling_analysis/nest_comparison
cd $W
echo "== node $(hostname)"
source /global/cfs/cdirs/m5460/envs/sna-dev.sh
FILTER='Warning|experimental|cupy._util|warnings.warn'
run() {  # run TAG N [extra srun args...] [wrapper]
  tag=$1; n=$2; shift 2
  echo "===== $tag np$n  ($(date +%H:%M:%S))  MPICH_GPU_SUPPORT_ENABLED=$MPICH_GPU_SUPPORT_ENABLED MPICH_GPU_IPC_ENABLED=${MPICH_GPU_IPC_ENABLED:-default}"
  timeout -k 15 150 srun -n $n --cpu-bind=none "$@" python -u $NC/run_partitioned_sna.py $P \
      --out $P/spikes_sna_np${n}_${tag}.npz 2>&1 | grep --line-buffered -vE "$FILTER"
  echo "  -> exit ${PIPESTATUS[0]} (124 = timed out)"
}
for n in 2 4; do
  # A. host-staged exchange: the route the old Frontier scripts used
  MPICH_GPU_SUPPORT_ENABLED=0 run host $n --gpus-per-task 1
  # B. GPU buffers straight to MPI, one GPU per rank, IPC off (Cray MPICH's advice under cgroups)
  MPICH_GPU_IPC_ENABLED=0 run gpu_noipc $n --gpus-per-task 1
  # C. GPU buffers straight to MPI with IPC: every GPU visible to every rank, own GPU first
  run gpu_ipc $n --gpus $n --gpu-bind=none $W/own_gpu_first.sh
done
conda deactivate
source /global/cfs/cdirs/m5460/envs/nestgpu.sh
for tag in host gpu_noipc gpu_ipc; do
  echo "===== compare $tag"
  f=""; for n in 2 4; do [ -f $P/spikes_sna_np${n}_${tag}.npz ] && f="$f $P/spikes_sna_np${n}_${tag}.npz"; done
  [ -n "$f" ] && python $NC/compare_partitioned_spikes.py $P/spikes_sna_np1.npz $f
done
echo "== done"
