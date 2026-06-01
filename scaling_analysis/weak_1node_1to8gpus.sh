#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J weak_1node_1to8gpus
#SBATCH -o /lustre/orion/lrn088/proj-shared/objective3/xxz/superneuroabm/scaling_analysis/outputs/weak_1node_1to8gpus_%j.out
#SBATCH -t 02:00:00
#SBATCH -q debug
#SBATCH -N 1

# Weak scaling test: 1 node, 1-8 GPUs sequential
# This single job tests all configurations from 1 to 8 GPUs

unset SLURM_EXPORT_ENV

module load cpe/26.03
module load PrgEnv-gnu
module load miniforge3
module load rocm/7.2.0
module load craype-accel-amd-gfx90a

source activate /lustre/orion/proj-shared/lrn088/objective3/envs/superneuroabm_env_cupy14

export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

# CuPy and profiling output
WORK_DIR=/lustre/orion/lrn088/proj-shared/objective3/xxz/superneuroabm/scaling_analysis
export CUPY_CACHE_DIR=${WORK_DIR}/outputs/cupy-cache
export CUPY_CACHE_SAVE_CUDA_SOURCE=1
export ROCPROF_OUTPUT_DIR=${WORK_DIR}/outputs/rocprof_${SLURM_JOB_ID}

cd ${WORK_DIR}
mkdir -p outputs outputs/cupy-cache

# Configuration (Brunel balanced network)
NEURONS_PER_WORKER=5000
TICKS=50
UPDATE_TICKS=1
G=5.0               # |J_I|/J_E (inhibition-dominated)
J_E=14.0            # excitatory weight; J_I = -G*J_E
DELAY=1.5           # synaptic delay (ms)
FIRING_RATE=10.0    # external Poisson drive (Hz)
IN_DEGREE_ARRAY=(1000)  # fixed in-degree K per neuron; add values to sweep

# Create shared CSV file for all timing results (will be appended to by Python script)
SHARED_CSV="outputs/weak_1node_1to8gpus_${SLURM_JOB_ID}.csv"

echo "======================================================================"
echo "Weak Scaling Test (Brunel) - 1 Node, 1 to 8 GPUs Sequential"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated nodes: 1"
echo "Testing: 1, 2, 3, 4, 5, 6, 7, 8 GPUs"
echo "In-degree K to test: ${IN_DEGREE_ARRAY[@]}"
echo "Results file: $SHARED_CSV"
echo "======================================================================"

# Loop through in-degree values
for IN_DEGREE in "${IN_DEGREE_ARRAY[@]}"; do
    echo ""
    echo "##################################################################"
    echo "## TESTING: IN_DEGREE K = $IN_DEGREE"
    echo "##################################################################"
    echo ""

    # Loop through GPU counts from 1 to 8
    for NGPUS in {1..8}; do
        NNODES=1
        NWORKERS=$NGPUS
        TOTAL_NEURONS=$((NWORKERS * NEURONS_PER_WORKER))

        echo ""
        echo "======================================================================"
        echo "Testing: $NNODES node, $NWORKERS GPUs, $TOTAL_NEURONS neurons | K: $IN_DEGREE"
        echo "======================================================================"
        echo "Starting at: $(date)"

        # Run test - Python script will append to shared CSV
        set +e
        OUTPUT=$(srun -N$NNODES -n$NWORKERS -c7 --ntasks-per-gpu=1 --gpu-bind=closest \
            python weak_scaling.py \
            --neurons-per-worker $NEURONS_PER_WORKER \
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

        # Check if test failed
        if [ $EXIT_CODE -ne 0 ]; then
            echo "=========================================="
            echo "ERROR: Test failed for $NWORKERS GPUs, K=$IN_DEGREE"
            echo "Exit code: $EXIT_CODE"
            echo "=========================================="
            echo ""
            echo "FULL ERROR OUTPUT:"
            echo "------------------------------------------"
            echo "$OUTPUT"
            echo "------------------------------------------"
            echo ""
            echo "Continuing to next configuration..."
            continue
        fi

        # Display filtered output (including verbose timing)
        echo "$OUTPUT" | grep -E "(WEAK SCALING|Total neurons|Simulation time|SUCCESS|ERROR|agents \(|TIMING|Rank|Metric|Straggler|MPI Traffic|Grid Barriers)"

        echo "Completed: $NWORKERS GPUs, K=$IN_DEGREE"
        echo ""
    done

    echo ""
    echo "## Completed: IN_DEGREE K = $IN_DEGREE"
    echo ""
done

echo ""
echo "======================================================================"
echo "ALL TESTS COMPLETED (1-8 GPUs)!"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Results file: $SHARED_CSV"
echo "======================================================================"
echo ""
echo "Summary of results:"
cat $SHARED_CSV
