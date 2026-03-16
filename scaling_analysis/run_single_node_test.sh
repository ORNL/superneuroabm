#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J weak_test
#SBATCH -o /lustre/orion/lrn088/proj-shared/objective3/xxz/superneuroabm/scaling_analysis/outputs/weak_single_%j.out
#SBATCH -t 00:30:00
#SBATCH -q debug
#SBATCH -N 1

# Run single weak scaling test with FULL output (no suppression/filtering)
#
# Usage:
#   sbatch run_single_node_test.sh                    # Run with 1 node (default)
#   sbatch -N 9 run_single_node_test.sh              # Run with 9 nodes
#   sbatch -N 12 -t 01:00:00 run_single_node_test.sh # Run with 12 nodes, 1 hour

unset SLURM_EXPORT_ENV

# Load modules
module load PrgEnv-gnu/8.6.0
module load cray-hdf5-parallel/1.12.2.11
module load miniforge3/23.11.0-0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a

# Activate environment
source activate /lustre/orion/proj-shared/lrn088/objective3/envs/superneuroabm_env_xxz

# Go to scaling_analysis directory
cd /lustre/orion/lrn088/proj-shared/objective3/xxz/superneuroabm/scaling_analysis

# Configuration
NEURONS_PER_WORKER=5000
TICKS=50
UPDATE_TICKS=1
INTRA_DEGREE=10
NUM_NEIGHBOR_CLUSTERS=1
CROSS_CLUSTER_EDGES=5000

NNODES=${SLURM_JOB_NUM_NODES}
NWORKERS=$((NNODES * 8))
TOTAL_NEURONS=$((NWORKERS * NEURONS_PER_WORKER))

echo "======================================================================"
echo "SINGLE NODE WEAK SCALING TEST - FULL OUTPUT"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $NNODES"
echo "Workers (GPUs): $NWORKERS"
echo "Total neurons: $TOTAL_NEURONS"
echo "Queue: debug"
echo "======================================================================"
echo "Configuration:"
echo "  Neurons per worker: $NEURONS_PER_WORKER (constant)"
echo "  Intra-cluster degree: $INTRA_DEGREE edges/neuron"
echo "  Cross-cluster edges: $CROSS_CLUSTER_EDGES per worker"
echo "  Neighbor clusters: $NUM_NEIGHBOR_CLUSTERS (directed ring)"
echo "  Simulation ticks: $TICKS"
echo "======================================================================"
echo ""
echo "Starting test at: $(date)"
echo ""

# Run test with FULL output (no filtering, no suppression)
# All Python output including verbose timing will be displayed
set +e

srun -N$NNODES -n$NWORKERS -c7 --ntasks-per-gpu=1 --gpu-bind=closest \
    python weak_scaling_const_comm.py \
    --neurons-per-worker $NEURONS_PER_WORKER \
    --ticks $TICKS \
    --update-ticks $UPDATE_TICKS \
    --intra-cluster-degree $INTRA_DEGREE \
    --cross-cluster-edges $CROSS_CLUSTER_EDGES \
    --num-neighbor-clusters $NUM_NEIGHBOR_CLUSTERS \
    --csv outputs/timing_results.csv

EXIT_CODE=$?

set -e

echo ""
echo "======================================================================"
echo "Test completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Test SUCCEEDED"
    echo ""
    echo "Check output above for detailed timing breakdowns including:"
    echo "  - Network generation time"
    echo "  - Model setup time"
    echo "  - Simulation time per tick"
    echo "  - MPI communication overhead"
    echo "  - GPU computation time"
    echo "  - Straggler analysis"
else
    echo ""
    echo "✗ Test FAILED with exit code $EXIT_CODE"
    echo ""
    echo "Full output shown above. Check for:"
    echo "  - Python tracebacks"
    echo "  - MPI errors"
    echo "  - GPU errors"
    echo "  - Memory errors"
fi

echo ""
echo "======================================================================"
echo "Output file: outputs/weak_single_${SLURM_JOB_ID}.out"
echo "======================================================================"
