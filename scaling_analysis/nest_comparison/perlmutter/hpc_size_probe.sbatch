#!/bin/bash
#SBATCH -A m5460_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 32
#SBATCH --gpus 4
#SBATCH -t 00:25:00
#SBATCH -J hpc-size-probe
#SBATCH -o /pscratch/sd/z/zhangx8/partitioned/hpc-size-probe-%j.out
# Does SuperNeuroABM hold NEST GPU's hpc_benchmark at scale 1 on one A100 (40 GB)?
# 11,250 neurons, in-degree 11,250 (126.6 M synapses), with the existing in-memory driver.
# Reports the timers, peak host memory and peak GPU memory.
W=/pscratch/sd/z/zhangx8/partitioned/hpc_probe
NC=/global/cfs/cdirs/m5460/zhangx8/superneuroabm/scaling_analysis/nest_comparison
mkdir -p $W && cd $W
echo "== node $(hostname)  $(date +%H:%M:%S)"
free -g | sed -n 1,2p
source /global/cfs/cdirs/m5460/envs/sna-dev.sh
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 2 > gpu_mem_$SLURM_JOB_ID.log &
LOGGER=$!
timeout -k 15 1200 srun -n 1 --gpus 1 --cpu-bind=none /usr/bin/time -v python -u $NC/hpc_benchmark_sna.py \
    --neurons-per-rank ${NEURONS:-11250} --in-degree ${INDEGREE:-11250} --presim-ms 50 --sim-ms 250 2>&1 \
  | grep --line-buffered -vE 'Warning|experimental|cupy._util|warnings.warn|Page size|Exit status|Signals|Swaps|File system|Socket|Average|Voluntary|Involuntary|Minor|Major|Percent of CPU'
echo "  -> exit ${PIPESTATUS[0]} (124 = timed out)"
kill $LOGGER
echo "peak GPU memory per device (MiB):"
awk -F', ' '{ if ($2 > m[$1]) m[$1] = $2 } END { for (i in m) print "  gpu " i ": " m[i] }' gpu_mem_$SLURM_JOB_ID.log | sort
echo "== done $(date +%H:%M:%S)"
