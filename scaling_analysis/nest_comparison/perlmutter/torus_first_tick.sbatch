#!/bin/bash
#SBATCH -A m5460_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-node=4
#SBATCH -t 00:30:00
#SBATCH -J torus-first-tick
#SBATCH -o /pscratch/sd/z/zhangx8/partitioned/torus-first-tick-%j.out
# Where SuperNeuroABM's first tick goes on the torus, by phase (SAGESim's timers on for that
# tick: run_partitioned_sna.py --first-tick-timers). SuperNeuroABM only, on parts that
# torus_scale.sbatch has written, 10 ms of network time after the build:
#   sbatch -N 16 --export=ALL,DIR=torus_mixed_11250_r17 torus_first_tick.sbatch
# The first tick took 80 s on 64 ranks, 309 to 665 s on 128 and over 800 s on 256 with the
# same 26 peers and 124,000 ghost somas per rank (jobs 61345640, 61345642, 61346213, 61345643).
# SAGESim's timers print only when the tick ends, so every rank also writes its Python stack
# every DUMP_EVERY seconds (default 20) and summarize_stack_dumps.py says where the ranks were.
# Jobs 61350810 / 61350811: one node's four ranks crawl through the wake-list build while all
# others wait for them. Every rank also samples its node's memory (memory_sampler.py), the
# dumps and samples of a run are kept in <parts>/stacks_np<N>_<variant>/:
#   sbatch -N 32 -t 00:24:00 --export=ALL,DIR=torus_mixed_11250_r17,VARIANTS="dropcache plain",SNA_TIMEOUT=600 torus_first_tick.sbatch
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
run() {   # label, extra options of the runner
  rm -rf $P/stacks_np$n
  echo "===== $1  ($(date +%H:%M:%S))"
  MPICH_GPU_SUPPORT_ENABLED=0 timeout -k 15 ${SNA_TIMEOUT:-1560} srun -N $NODES -n $n --ntasks-per-node=4 --cpu-bind=none --gpus-per-task 1 \
      python -u $NC/run_partitioned_sna.py $P --config $CFG --eta 1.685 --presim-ms 50 --ms 60 --vm-sigma 7.2 \
      --first-tick-timers --stack-dump-every ${DUMP_EVERY:-20} $2 --out $P/spikes_sna_np${n}_first_tick.npz 2>&1 \
    | grep --line-buffered -vE "Warning|experimental|cupy._util|warnings.warn|^\[Rank"
  echo "  -> exit ${PIPESTATUS[0]} (124 = timed out)  ($(date +%H:%M:%S))"
  rm -rf $P/stacks_np${n}_$1
  mv $P/stacks_np$n $P/stacks_np${n}_$1
  echo "===== $1: where the ranks were during the first tick, and the nodes' memory"
  python $NC/summarize_stack_dumps.py $P/stacks_np${n}_$1
}
# VARIANTS: the runs of this job, in order, on the same nodes. "plain" is the run as it is;
# "dropcache" drops the part files from the page cache after the load (--drop-file-cache).
for v in ${VARIANTS:-plain}; do
  case $v in
    plain)     run plain "";;
    dropcache) run dropcache "--drop-file-cache";;
    *) echo "VARIANTS: $v?";;
  esac
done
