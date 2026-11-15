#!/bin/bash
#SBATCH -A m5460_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 32
#SBATCH --gpus 4
#SBATCH -t 00:15:00
#SBATCH -J regress-12500
#SBATCH -o /pscratch/sd/z/zhangx8/partitioned/regress-12500-%j.out
# Regression of both runners on the 12,500-neuron network after a change: NEST GPU at
# 1, 2, 4 ranks must reproduce spikes_nestgpu_np1.npz exactly; SuperNeuroABM at 1 and 2
# ranks must reproduce spikes_sna_np1.npz for the first few hundred ms.
W=/pscratch/sd/z/zhangx8/partitioned
P=$W/brunel_alpha_12500
NC=/global/cfs/cdirs/m5460/zhangx8/superneuroabm/scaling_analysis/nest_comparison
TAG=${TAG:-regress}
cd $W
echo "== node $(hostname)  tag $TAG"
NFILTER='^Building time|^Simulation time|^Calibrating|^Simulating|Neural activity|^\s*$|NEST GPU --|Copyright|licen|Homepage|^ *-{3,}|Host [0-9]+ :|AS IS|WARRANTY'
SFILTER='Warning|experimental|cupy._util|warnings.warn'
source /global/cfs/cdirs/m5460/envs/sna-dev.sh
for n in 1 2; do
  echo "===== SuperNeuroABM np$n  ($(date +%H:%M:%S))"
  MPICH_GPU_SUPPORT_ENABLED=0 timeout -k 15 240 srun -n $n --gpus-per-task 1 --cpu-bind=none \
      python -u $NC/run_partitioned_sna.py $P --out $P/spikes_sna_np${n}_${TAG}.npz 2>&1 | grep --line-buffered -vE "$SFILTER"
  echo "  -> exit ${PIPESTATUS[0]} (124 = timed out)"
done
conda deactivate
source /global/cfs/cdirs/m5460/envs/nestgpu.sh
for n in 1 2 4; do
  echo "===== NEST GPU np$n  ($(date +%H:%M:%S))"
  timeout -k 15 300 srun -n $n --gpus-per-task 1 --cpu-bind=none python -u $NC/run_partitioned_nestgpu.py $P \
      --out $P/spikes_nestgpu_np${n}_${TAG}.npz 2>&1 | grep --line-buffered -vE "$NFILTER"
  echo "  -> exit ${PIPESTATUS[0]} (124 = timed out)"
done
echo "===== compare NEST GPU"
f=""; for n in 1 2 4; do [ -f $P/spikes_nestgpu_np${n}_${TAG}.npz ] && f="$f $P/spikes_nestgpu_np${n}_${TAG}.npz"; done
[ -n "$f" ] && python $NC/compare_partitioned_spikes.py $P/spikes_nestgpu_np1.npz $f | grep -A2 "spikes_nestgpu_np1 vs"
echo "===== compare SuperNeuroABM"
f=""; for n in 1 2; do [ -f $P/spikes_sna_np${n}_${TAG}.npz ] && f="$f $P/spikes_sna_np${n}_${TAG}.npz"; done
[ -n "$f" ] && python $NC/compare_partitioned_spikes.py $P/spikes_sna_np1.npz $f | grep -A2 "spikes_sna_np1 vs"
echo "== done"
