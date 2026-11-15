#!/bin/bash
#SBATCH -A m5460_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 32
#SBATCH --gpus-per-task 1
#SBATCH -t 00:30:00
#SBATCH -J partitioned-brunel
#SBATCH -o /pscratch/sd/z/zhangx8/partitioned/multirank-%j.out
P=/pscratch/sd/z/zhangx8/partitioned/brunel_alpha_12500
NC=/global/cfs/cdirs/m5460/zhangx8/superneuroabm/scaling_analysis/nest_comparison
cd /pscratch/sd/z/zhangx8/partitioned
echo "== node $(hostname)"
source /global/cfs/cdirs/m5460/envs/sna-dev.sh
for n in 1 2 4; do
  echo "===== SuperNeuroABM np$n"
  srun -n $n --gpus-per-task 1 --cpu-bind=none python $NC/run_partitioned_sna.py $P 2>&1 | grep -vE 'Warning|experimental|cupy._util'
done
conda deactivate
source /global/cfs/cdirs/m5460/envs/nestgpu.sh
for n in 1 2 4; do
  echo "===== NEST GPU np$n"
  srun -n $n --gpus-per-task 1 --cpu-bind=none python $NC/run_partitioned_nestgpu.py $P 2>&1 | grep -vE '^Building time|^Simulation time|^Calibrating|^Simulating|Neural activity|^\s*$|NEST GPU --|Copyright|licen|Homepage|^ *-{3,}'
done
echo "===== compare"
python $NC/compare_partitioned_spikes.py $P/spikes_sna_np1.npz $P/spikes_sna_np2.npz $P/spikes_sna_np4.npz \
    $P/spikes_nestgpu_np1.npz $P/spikes_nestgpu_np2.npz $P/spikes_nestgpu_np4.npz
echo "== done"
