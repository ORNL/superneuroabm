#!/bin/bash
#SBATCH -A m5460_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-node=4
#SBATCH -t 00:15:00
#SBATCH -J torus-first-tick
#SBATCH -o /pscratch/sd/z/zhangx8/partitioned/torus-first-tick-%j.out
# Where SuperNeuroABM's first tick goes on the torus, by phase (SAGESim's timers on for that
# tick: run_partitioned_sna.py --first-tick-timers). SuperNeuroABM only, on parts that
# torus_scale.sbatch has written, 10 ms of network time after the build:
#   sbatch -N 16 --export=ALL,DIR=torus_mixed_11250_r17 torus_first_tick.sbatch
# The first tick took 80 s on 64 ranks, 309 s on 128 and over 800 s on 256 with the same
# 26 peers and 124,000 ghost somas per rank (jobs 61345640, 61345642, 61345643).
W=/pscratch/sd/z/zhangx8/partitioned
P=$W/${DIR:?DIR=<parts directory under the work directory>}
NC=/global/cfs/cdirs/m5460/zhangx8/superneuroabm/scaling_analysis/nest_comparison
CFG=$NC/hpc_alpha_config.yaml
NODES=$SLURM_JOB_NUM_NODES
n=$((4 * NODES))
cd $W
echo "== $NODES node(s), $n ranks, $DIR, first node $(hostname)  ($(date +%H:%M:%S))"
[ -f $P/np$n/partition_$((n - 1)).npz ] || { echo "no parts for $n ranks in $P"; exit 2; }
source /global/cfs/cdirs/m5460/envs/sna-dev.sh
MPICH_GPU_SUPPORT_ENABLED=0 timeout -k 15 ${SNA_TIMEOUT:-780} srun -N $NODES -n $n --ntasks-per-node=4 --cpu-bind=none --gpus-per-task 1 \
    python -u $NC/run_partitioned_sna.py $P --config $CFG --eta 1.685 --presim-ms 50 --ms 60 --vm-sigma 7.2 \
    --first-tick-timers --out $P/spikes_sna_np${n}_first_tick.npz 2>&1 \
  | grep --line-buffered -vE "Warning|experimental|cupy._util|warnings.warn|^\[Rank"
echo "  -> exit ${PIPESTATUS[0]} (124 = timed out)  ($(date +%H:%M:%S))"
