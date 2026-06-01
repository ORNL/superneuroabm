#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J strong_1to20_seq
#SBATCH -o /lustre/orion/lrn088/proj-shared/objective3/xxz/superneuroabm/scaling_analysis/outputs/strong_1to20_seq_%j.out
#SBATCH -t 02:00:00
#SBATCH -q debug
#SBATCH -N 20

# Strong scaling test (Brunel): fixed total neurons, 1-20 nodes sequential.

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
export ROCPROF_OUTPUT_DIR=${WORK_DIR}/outputs/rocprof_${SLURM_JOB_ID}

cd ${WORK_DIR}
mkdir -p outputs outputs/cupy-cache

# Configuration (Brunel balanced network)
# Total neurons is held CONSTANT; only the worker count grows.
# Must be divisible by every worker count tested (1..160 GPUs => use a highly
# divisible N). 80640 = 2^7 * 3^2 * 5 * 7 * 2 is divisible by all of 1..20 nodes
# (8..160 GPUs) for the counts that matter; adjust if you change the node range.
TOTAL_NEURONS=80640
TICKS=50
UPDATE_TICKS=1
IN_DEGREE=1000      # fixed in-degree K per neuron (Brunel)
G=5.0               # |J_I|/J_E (inhibition-dominated)
J_E=14.0            # excitatory weight; J_I = -G*J_E
DELAY=1.5           # synaptic delay (ms)
FIRING_RATE=10.0    # external Poisson drive (Hz)

SHARED_CSV="outputs/strong_1to20_seq_${SLURM_JOB_ID}.csv"

echo "======================================================================"
echo "Strong Scaling Test (Brunel) - 1 to 20 Nodes Sequential"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Total neurons (fixed): $TOTAL_NEURONS"
echo "In-degree K: $IN_DEGREE"
echo "Results file: $SHARED_CSV"
echo "======================================================================"

for NNODES in {1..20}; do
    NWORKERS=$((NNODES * 8))

    if (( TOTAL_NEURONS % NWORKERS != 0 )); then
        echo "SKIP: $NWORKERS GPUs does not evenly divide $TOTAL_NEURONS neurons"
        continue
    fi

    echo ""
    echo "======================================================================"
    echo "Testing: $NNODES nodes, $NWORKERS GPUs | $((TOTAL_NEURONS / NWORKERS)) neurons/GPU"
    echo "======================================================================"
    echo "Starting at: $(date)"

    set +e
    OUTPUT=$(srun -N$NNODES -n$NWORKERS -c7 --ntasks-per-gpu=1 --gpu-bind=closest \
        python strong_scaling.py \
        --total-neurons $TOTAL_NEURONS \
        --ticks $TICKS \
        --update-ticks $UPDATE_TICKS \
        --in-degree $IN_DEGREE \
        --g $G \
        --J-E $J_E \
        --delay $DELAY \
        --firing-rate $FIRING_RATE \
        --csv $SHARED_CSV \
        2>&1)
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Test failed for $NNODES nodes (exit code: $EXIT_CODE)"
        echo "$OUTPUT"
        continue
    fi

    echo "$OUTPUT" | grep -E "(STRONG SCALING|Total neurons|Neurons per worker|Simulation time|SUCCESS|ERROR|TIMING|Rank|Metric|Straggler|MPI Traffic)"
    echo "Completed: $NNODES nodes"
done

echo ""
echo "======================================================================"
echo "ALL STRONG SCALING TESTS COMPLETED (1-20 nodes)!"
echo "======================================================================"
echo "Results file: $SHARED_CSV"
cat $SHARED_CSV
