#!/bin/bash
#SBATCH -A m5460_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 32
#SBATCH --gpus 4
#SBATCH -t 00:25:00
#SBATCH -J nest-mirror-check
#SBATCH -o /pscratch/sd/z/zhangx8/partitioned/nest-mirror-check-%j.out
# NEST GPU at 1, 2 and 4 ranks (mirrored RemoteConnect calls, 31-bit node index), compared
# with the earlier 1-rank run spikes_nestgpu_np1.npz.
W=/pscratch/sd/z/zhangx8/partitioned
P=$W/brunel_alpha_12500
NC=/global/cfs/cdirs/m5460/zhangx8/superneuroabm/scaling_analysis/nest_comparison
TAG=${TAG:-bits31}
cd $W
echo "== node $(hostname)  tag $TAG"
source /global/cfs/cdirs/m5460/envs/nestgpu.sh
FILTER='^Building time|^Simulation time|^Calibrating|^Simulating|Neural activity|^\s*$|NEST GPU --|Copyright|licen|Homepage|^ *-{3,}|Host [0-9]+ :|AS IS|WARRANTY'
for n in 1 2 4; do
  echo "===== NEST GPU np$n  ($(date +%H:%M:%S))"
  timeout -k 15 420 srun -n $n --gpus-per-task 1 --cpu-bind=none python -u $NC/run_partitioned_nestgpu.py $P \
      --out $P/spikes_nestgpu_np${n}_${TAG}.npz 2>&1 | grep --line-buffered -vE "$FILTER"
  echo "  -> exit ${PIPESTATUS[0]} (124 = timed out)"
done
echo "===== compare"
f=""; for n in 1 2 4; do [ -f $P/spikes_nestgpu_np${n}_${TAG}.npz ] && f="$f $P/spikes_nestgpu_np${n}_${TAG}.npz"; done
[ -n "$f" ] && python $NC/compare_partitioned_spikes.py $P/spikes_nestgpu_np1.npz $f
echo "== done"
