#!/bin/bash
#SBATCH -A lrn088
#SBATCH -p batch
#SBATCH -J weak_3d_chunk
#SBATCH -o /lustre/orion/lrn088/proj-shared/objective3/xxz/superneuroabm/scaling_analysis/outputs/weak_3d_chunk_%j.out

# Parameterized weak-scaling chunk runner. ALL timing rows append to ONE shared CSV.
# Queue/walltime/node-count are set on the sbatch COMMAND LINE (-q, -t, -N), so the same script
# serves both Group-B paths:
#   debug chunk : sbatch -q debug    -N <maxnodes> -t 02:00:00 weak_3d_chunk.sh "4000"      4 "1 2"
#   extended    : sbatch -q extended -N 16         -t 24:00:00 weak_3d_chunk.sh "4000 2000" 4 "1 2 4 8 16 32 64"
#   flaky refill: sbatch -q debug    -N 1          -t 00:45:00 weak_3d_chunk.sh "1000"      8 "2" 5
#
# Args:  $1 = KLIST (space-separated in-degrees)   $2 = RANKS_PER_NODE (4 or 8)   $3 = WLIST (space-separated worker counts)
#        $4 = MAX_ATTEMPTS per case (default 3)
#
# MAX_ATTEMPTS exists because the cupy.jit transpile crash ("'Assign' object has no attribute
# 'name'") is NONDETERMINISTIC at roughly 1-in-6 and is NOT driven by PYTHONHASHSEED (proven by
# job 5069521: 5 PASS / 1 FAIL with both random and fixed seed). A retry with a fresh cache is
# the remedy, not a code fix.

KLIST="${1:?usage: weak_3d_chunk.sh \"<K...>\" <ranks_per_node> \"<workers...>\" [max_attempts]}"
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

NEURONS_PER_WORKER=${NEURONS_PER_WORKER:-12500}
CONNECTION_RADIUS=
TICKS=${TICKS:-100}
UPDATE_TICKS=1
G=5.0
J_E=0.02581
DELAY=1.5
FIRING_RATE=10.0

# THE single shared CSV. Every case appends one row (rank-0 append, header-once). Never overwritten.
SHARED_CSV="outputs/weak_3d_curve_npp${NEURONS_PER_WORKER}.csv"

EXTRA_FLAGS="--diagnostics"
[ -n "$CONNECTION_RADIUS" ] && EXTRA_FLAGS="$EXTRA_FLAGS --connection-radius $CONNECTION_RADIUS"

echo "######################################################################"
echo "weak_3d_chunk: K={$KLIST}  ranks/node=$RPN  workers={$WLIST}  ticks=$TICKS"
echo "CSV: $SHARED_CSV   job $SLURM_JOB_ID   $(date)"
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
for IN_DEGREE in $KLIST; do
  for NWORKERS in $WLIST; do
    NNODES=$(( (NWORKERS + RPN - 1) / RPN ))
    echo ""
    echo "----------------------------------------------------------------------"
    echo "torus3d K=$IN_DEGREE: $NWORKERS GPUs ($NNODES nodes, $RPN/node), $((NWORKERS * NEURONS_PER_WORKER)) neurons"
    echo "----------------------------------------------------------------------"
    for ATTEMPT in $(seq 1 $MAX_ATTEMPTS); do
      # Isolated cupy JIT cache PER ATTEMPT: the shared outputs/cupy-cache triggers a
      # cupy14/py3.14 transpile crash ('Assign' has no attribute 'name') on stale/racy
      # entries, and even a fresh cache hits it ~1/6 of the time. Each attempt therefore
      # gets its own cache dir so the retry is a genuinely independent transpile.
      export CUPY_CACHE_DIR="${WORK_DIR}/outputs/cc_${SLURM_JOB_ID}_K${IN_DEGREE}_w${NWORKERS}_a${ATTEMPT}"
      rm -rf "$CUPY_CACHE_DIR"; mkdir -p "$CUPY_CACHE_DIR"
      # Per-attempt log so a successful retry does not clobber the failing attempt's evidence.
      CASE_LOG="outputs/case_${SLURM_JOB_ID}_K${IN_DEGREE}_w${NWORKERS}_a${ATTEMPT}.log"
      set +e
      srun -N$NNODES -n$NWORKERS -c7 --ntasks-per-gpu=1 --gpu-bind=closest \
          python -u weak_scaling.py \
          --topology torus3d \
          --neurons-per-worker $NEURONS_PER_WORKER \
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
          grep -E "Wiring:|peers |ghost somas|Per-tick timing|tick 1:|tick 2:|SUCCESS" "$CASE_LOG"
          # Must match weak_scaling.py's partition_dir naming exactly.
          cleanup_partitions "partitions/${NWORKERS}w_${NEURONS_PER_WORKER}n_K${IN_DEGREE}_torus3d_rauto"
          break
      fi
      if [ $ATTEMPT -lt $MAX_ATTEMPTS ]; then
          echo "RETRY $ATTEMPT/$MAX_ATTEMPTS after exit $EXIT_CODE (K=$IN_DEGREE, $NWORKERS GPUs) -- log: $CASE_LOG"
          grep -m1 "'Assign' object has no attribute" "$CASE_LOG" || tail -10 "$CASE_LOG"
      else
          echo "FAILED (K=$IN_DEGREE, $NWORKERS GPUs, exit $EXIT_CODE) after $MAX_ATTEMPTS attempts -- full log: $CASE_LOG"
          tail -40 "$CASE_LOG"
          FAILED_ANY=1
      fi
    done
  done
done

echo ""
echo "######################################################################"
echo "CHUNK DONE (K={$KLIST}, workers={$WLIST}). $(date)"
echo "Shared CSV now:"; cat $SHARED_CSV 2>/dev/null
echo "######################################################################"

if [ "$FAILED_ANY" -ne 0 ]; then
    echo "ONE OR MORE CASES FAILED -- see per-case logs (outputs/case_${SLURM_JOB_ID}_*.log)."
    exit 1
fi
