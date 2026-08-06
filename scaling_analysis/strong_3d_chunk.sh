#!/bin/bash
#SBATCH -A lrn088
#SBATCH -p batch
#SBATCH -J strong_3d_chunk
#SBATCH -o /lustre/orion/lrn088/proj-shared/objective3/xxz/superneuroabm/scaling_analysis/outputs/strong_3d_chunk_%j.out

# Parameterized STRONG-scaling chunk runner (the twin of weak_3d_chunk.sh).
# ALL timing rows append to ONE shared CSV per (total-neurons, K). Queue/walltime/node-count
# are set on the sbatch COMMAND LINE (-q, -t, -N), so the same script serves every chunk:
#
#   chunk A (small P) : sbatch -N 32  -t 01:00:00 strong_3d_chunk.sh "204800" 8 "16 32 64 128 256"
#   chunk B (large P) : sbatch -N 256 -t 01:00:00 strong_3d_chunk.sh "204800" 8 "512 1024 2048"
#   single refill     : sbatch -N 4   -t 00:45:00 strong_3d_chunk.sh "204800" 8 "32" 5
#
# The partition is pinned to `batch` in the SBATCH header; never add -q debug (the two cannot
# be combined). Chunks are still split by node count so a 2-node point does not idle a
# 256-node allocation, but every chunk now waits in the batch queue.
#
# Args:  $1 = NLIST (space-separated TOTAL-neuron counts, each held fixed across its sweep)
#        $2 = RANKS_PER_NODE (8 for this campaign)
#        $3 = WLIST (space-separated worker counts)
#        $4 = MAX_ATTEMPTS per case (default 3)
# Env:   IN_DEGREE (default 1000), TICKS (default 100), KEEP_PARTITIONS (default 0)
#        -- override without editing the script.
#
# -N on the sbatch line must cover the LARGEST srun in WLIST (ceil(max(WLIST)/RANKS_PER_NODE)).
#
# MAX_ATTEMPTS exists because the cupy.jit transpile crash ("'Assign' object has no attribute
# 'name'") is NONDETERMINISTIC at roughly 1-in-6 and is NOT driven by PYTHONHASHSEED (proven by
# job 5069521: 5 PASS / 1 FAIL with both random and fixed seed). A retry with a fresh cache is
# the remedy, not a code fix.

NLIST="${1:?usage: strong_3d_chunk.sh \"<total_neurons...>\" <ranks_per_node> \"<workers...>\" [max_attempts]}"
RPN="${2:?ranks_per_node required}"
WLIST="${3:?worker list required}"
MAX_ATTEMPTS="${4:-3}"

unset SLURM_EXPORT_ENV

module load cpe/26.03
module load PrgEnv-gnu
module load miniforge3
module load rocm/7.2.0
module load craype-accel-amd-gfx90a

source activate /lustre/orion/proj-shared/lrn088/objective3/envs/superneuroabm_env_cupy14

export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

WORK_DIR=/lustre/orion/lrn088/proj-shared/objective3/xxz/superneuroabm/scaling_analysis
export CUPY_CACHE_DIR=${WORK_DIR}/outputs/cupy-cache
export CUPY_CACHE_SAVE_CUDA_SOURCE=1

cd ${WORK_DIR}
mkdir -p outputs outputs/ticks partitions
lfs setstripe -c 8 partitions 2>/dev/null || true

# Held identical to the weak-scaling campaign so the two curves describe one network family.
# The P=16 point (12,800 neurons/GPU) is the cross-check against the weak K=1000 curve's
# 12,500 neurons/GPU point.
IN_DEGREE=${IN_DEGREE:-1000}
# 100, not the weak campaign's 30. The steady-state metric is per-tick, so it is tick-count
# independent, and tick 1 (ghost discovery + GPU buffer build) dominates wall time so
# heavily -- ~211 s of a 212 s simulate at P=16 -- that tripling the sampled ticks costs
# under a second per run. Cheap samples matter here because the step-time distribution has
# a heavy right tail and the metric is a median.
TICKS=${TICKS:-100}
UPDATE_TICKS=1
G=5.0
J_E=0.02581
DELAY=1.5
FIRING_RATE=10.0
CONNECTION_RADIUS=          # empty = auto (smallest ball holding >= 2K); FIXED across the sweep

EXTRA_FLAGS="--diagnostics"
[ -n "$CONNECTION_RADIUS" ] && EXTRA_FLAGS="$EXTRA_FLAGS --connection-radius $CONNECTION_RADIUS"

echo "######################################################################"
echo "strong_3d_chunk: N={$NLIST}  ranks/node=$RPN  workers={$WLIST}  K=$IN_DEGREE  ticks=$TICKS"
echo "job $SLURM_JOB_ID   $(date)"
echo "######################################################################"

# Per-point partition cleanup, ON by default (KEEP_PARTITIONS=1 to disable). Without it the
# full sweep leaves ~13 TB behind; with it, peak is one configuration and the end state is ~0.
# Deleted only on SUCCESS -- a failed point keeps its partitions so a retry does not pay to
# regenerate them and so the failure can be inspected.
KEEP_PARTITIONS=${KEEP_PARTITIONS:-0}
cleanup_partitions() {   # $1 = partition dir
    if [ "$KEEP_PARTITIONS" = "1" ]; then
        echo "    KEEP_PARTITIONS=1 -- leaving $1"
    elif [ -d "$1" ]; then
        local sz; sz=$(du -sh "$1" 2>/dev/null | cut -f1)
        rm -rf "$1" && echo "    reclaimed $sz from $1"
    fi
}

FAILED_ANY=0
for TOTAL_NEURONS in $NLIST; do

  # THE single shared CSV for this problem size. Every case appends one row (rank-0 append,
  # header-once). Never overwritten -- it is append-only provenance for consolidate_strong_3d.py.
  SHARED_CSV="outputs/strong_3d_curve_N${TOTAL_NEURONS}_K${IN_DEGREE}.csv"

  # Sizing table for the whole sweep up front: tile shape, global grid and predicted peer count
  # change at EVERY point under strong scaling, so the log should carry them next to the timings.
  # Informational only -- an invalid point is skipped below rather than aborting the chunk.
  echo ""
  python -u strong_scaling.py --dry-run $WLIST \
      --total-neurons $TOTAL_NEURONS --in-degree $IN_DEGREE --ranks-per-node $RPN \
      $([ -n "$CONNECTION_RADIUS" ] && echo "--connection-radius $CONNECTION_RADIUS")

  for NWORKERS in $WLIST; do
    # Strong scaling's defining constraint: N is fixed, so P must divide it exactly.
    if (( TOTAL_NEURONS % NWORKERS != 0 )); then
        echo ""
        echo "SKIP: $NWORKERS workers does not evenly divide $TOTAL_NEURONS neurons"
        FAILED_ANY=1
        continue
    fi
    NNODES=$(( (NWORKERS + RPN - 1) / RPN ))
    echo ""
    echo "----------------------------------------------------------------------"
    echo "torus3d N=$TOTAL_NEURONS K=$IN_DEGREE: $NWORKERS GPUs ($NNODES nodes, $RPN/node), \
$((TOTAL_NEURONS / NWORKERS)) neurons/GPU"
    echo "----------------------------------------------------------------------"
    for ATTEMPT in $(seq 1 $MAX_ATTEMPTS); do
      # Isolated cupy JIT cache PER ATTEMPT: the shared outputs/cupy-cache triggers a
      # cupy14/py3.14 transpile crash ('Assign' has no attribute 'name') on stale/racy
      # entries, and even a fresh cache hits it ~1/6 of the time. Each attempt therefore
      # gets its own cache dir so the retry is a genuinely independent transpile.
      export CUPY_CACHE_DIR="${WORK_DIR}/outputs/sc_${SLURM_JOB_ID}_N${TOTAL_NEURONS}_w${NWORKERS}_a${ATTEMPT}"
      rm -rf "$CUPY_CACHE_DIR"; mkdir -p "$CUPY_CACHE_DIR"
      # Per-attempt log so a successful retry does not clobber the failing attempt's evidence.
      CASE_LOG="outputs/scase_${SLURM_JOB_ID}_N${TOTAL_NEURONS}_w${NWORKERS}_a${ATTEMPT}.log"
      set +e
      srun -N$NNODES -n$NWORKERS -c7 --ntasks-per-gpu=1 --gpu-bind=closest \
          python -u strong_scaling.py \
          --topology torus3d \
          --total-neurons $TOTAL_NEURONS \
          --ticks $TICKS \
          --update-ticks $UPDATE_TICKS \
          --in-degree $IN_DEGREE \
          --g $G \
          --J-E $J_E \
          --delay $DELAY \
          --firing-rate $FIRING_RATE \
          $EXTRA_FLAGS \
          --csv $SHARED_CSV \
          > "$CASE_LOG" 2>&1
      EXIT_CODE=$?
      set -e
      if [ $EXIT_CODE -eq 0 ]; then
          echo "Full log: $CASE_LOG  (attempt $ATTEMPT/$MAX_ATTEMPTS)"
          grep -E "Wiring:|predicted peers|peers |ghost somas|Per-tick timing|tick 1:|tick 2:|SUCCESS" "$CASE_LOG"
          # Must match strong_scaling.py's partition_dir naming exactly.
          cleanup_partitions "partitions/strong_${TOTAL_NEURONS}n_${NWORKERS}w_K${IN_DEGREE}_torus3d_rauto"
          break
      fi
      if [ $ATTEMPT -lt $MAX_ATTEMPTS ]; then
          echo "RETRY $ATTEMPT/$MAX_ATTEMPTS after exit $EXIT_CODE (N=$TOTAL_NEURONS, $NWORKERS GPUs) -- log: $CASE_LOG"
          grep -m1 "'Assign' object has no attribute" "$CASE_LOG" || tail -10 "$CASE_LOG"
      else
          echo "FAILED (N=$TOTAL_NEURONS, $NWORKERS GPUs, exit $EXIT_CODE) after $MAX_ATTEMPTS attempts -- full log: $CASE_LOG"
          tail -40 "$CASE_LOG"
          FAILED_ANY=1
      fi
    done
  done

  echo ""
  echo "Shared CSV for N=$TOTAL_NEURONS now:"; cat $SHARED_CSV 2>/dev/null
done

echo ""
echo "######################################################################"
echo "CHUNK DONE (N={$NLIST}, workers={$WLIST}). $(date)"
echo "######################################################################"

if [ "$FAILED_ANY" -ne 0 ]; then
    echo "ONE OR MORE CASES FAILED OR WERE SKIPPED -- see per-case logs (outputs/scase_${SLURM_JOB_ID}_*.log)."
    exit 1
fi
