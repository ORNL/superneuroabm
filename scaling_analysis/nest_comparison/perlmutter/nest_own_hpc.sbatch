#!/bin/bash
#SBATCH -A m5460_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 32
#SBATCH --gpus 4
#SBATCH -t 00:20:00
#SBATCH -J nest-own-hpc
#SBATCH -o /pscratch/sd/z/zhangx8/partitioned/nest-own-hpc-%j.out
# NEST GPU's own hpc_benchmark.py, unmodified (rule-based distributed connect, Poisson
# generator, random initial V_m), at scale 1 on RANKS processes: the reference for the
# firing rate and NEST GPU's best-case timing.  RANKS="1 2 4" sbatch nest_own_hpc.sbatch
W=/pscratch/sd/z/zhangx8/partitioned/nest_own_hpc
HPC=/global/cfs/cdirs/m5460/zhangx8/software/nest-gpu/src/python/examples/hpc_benchmark/hpc_benchmark.py
RANKS=${RANKS:-1 2 4}
SCALE=${SCALE:-1}
mkdir -p $W && cd $W
echo "== node $(hostname)  ranks $RANKS  scale $SCALE  ($(date +%H:%M:%S))"
source /global/cfs/cdirs/m5460/envs/nestgpu.sh
for n in $RANKS; do
  D=$W/scale${SCALE}_np$n; mkdir -p $D
  echo "===== NEST GPU hpc_benchmark.py np$n  ($(date +%H:%M:%S))"
  timeout -k 15 300 srun -n $n --gpus-per-task 1 --cpu-bind=none python -u $HPC --path $D --scale $SCALE \
      --simtime 250 --verbosity 0 > $D/stdout.txt 2>&1
  echo "  -> exit $? (124 = timed out)"
  python - $D $n <<'EOF'
import json, sys, glob
d, n = sys.argv[1], int(sys.argv[2])
logs = [json.load(open(f)) for f in sorted(glob.glob(d + "/log_*.json"))]
if len(logs) != n:
    print(f"  {len(logs)} of {n} rank logs found; last lines of stdout:")
    print("".join(open(d + "/stdout.txt").readlines()[-12:]))
else:
    for key in ("time_construct", "time_calibrate", "time_presimulate", "time_simulate", "time_total"):
        v = [l["timers"][key] / 1e9 for l in logs]
        print(f"  {key:18s} max {max(v):8.3f} s   min {min(v):8.3f} s")
    if "stats" in logs[0]:
        e = [l["stats"]["excitatory_firing_rate"] for l in logs]; i = [l["stats"]["inhibitory_firing_rate"] for l in logs]
        print(f"  firing rate: E {sum(e)/len(e):.3f} Hz  I {sum(i)/len(i):.3f} Hz  (mean over ranks)")
    g = logs[0]["gpu_mem"]; print("  gpu_mem (rank 0):", {k: round(v / 2**30, 2) if v > 1e6 else v for k, v in g.items()})
EOF
done
echo "== done ($(date +%H:%M:%S))"
