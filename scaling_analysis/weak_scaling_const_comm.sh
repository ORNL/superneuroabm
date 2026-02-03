#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J weak_scaling_constcomm
#SBATCH -o /lustre/orion/lrn088/proj-shared/objective3/xxz/superneuroabm/tests/output/weak_scaling_constcomm_%j.out
#SBATCH -t 03:00:00
#SBATCH -p extended
#SBATCH -N 10
#SBATCH --contiguous

# Weak scaling test on Frontier - CONSTANT COMMUNICATION MODEL
# Cross-worker edges per worker remains constant as network scales
# Uses weak_scaling_const_comm.py
# Adjust -N and --gpus based on your needs

unset SLURM_EXPORT_ENV

# Load modules
module load PrgEnv-gnu/8.6.0
module load cray-hdf5-parallel/1.12.2.11
module load miniforge3/23.11.0-0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
module load metis/5.1.0

# Set up METIS library path
export LD_LIBRARY_PATH=/lustre/orion/lrn088/proj-shared/objective3/xxz/SAGESim/local_lib:$LD_LIBRARY_PATH

# Activate environment
source activate /lustre/orion/proj-shared/lrn088/objective3/envs/superneuroabm_cupy13.6_env_xxz

# Go to test directory
cd /lustre/orion/lrn088/proj-shared/objective3/xxz/superneuroabm/tests

# Create output directory
mkdir -p output

# Configuration - FULL WEAK SCALING TEST (1-10 nodes, 8-80 GPUs)
NEURONS_PER_WORKER=5000     # Neurons per worker (constant)
TICKS=10                    # Quick test
UPDATE_TICKS=5              # Sync every 5 ticks
INTRA_DEGREE=10             # Constant degree per neuron (O(n) edges!)
CROSS_CLUSTER_EDGES=100     # Constant cross-cluster edges per worker
NUM_NEIGHBOR_CLUSTERS=1     # Directed ring topology (1 = TRUE weak scaling)

echo "======================================================================"
echo "Weak Scaling Test - PROPER WEAK SCALING (Directed Ring)"
echo "======================================================================"
echo "Goal: Demonstrate CONSTANT per-worker workload as network scales"
echo "      - Constant neurons per worker"
echo "      - Constant edges per worker (O(n), not O(n²)!)"
echo "      - Constant communication per worker"
echo "      - Constant contextualization overhead per worker"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs: $SLURM_GPUS"
echo "Neurons per worker: $NEURONS_PER_WORKER (constant)"
echo "Intra-cluster degree: $INTRA_DEGREE edges/neuron (constant → O(n) scaling!)"
echo "Neighbor clusters: $NUM_NEIGHBOR_CLUSTERS (directed ring)"
echo "Cross-cluster edges per neighbor: $CROSS_CLUSTER_EDGES (constant)"
echo "Expected edges per worker: ~$((NEURONS_PER_WORKER * INTRA_DEGREE + CROSS_CLUSTER_EDGES))"
echo "Simulation ticks: $TICKS"
echo "======================================================================"

# Test with 1 to 10 nodes (8 to 80 GPUs)
# Each node has 8 GPUs on Frontier
for NNODES in 1 2 3 4 5 6 7 8 9 10; do
    NWORKERS=$((NNODES * 8))
    TOTAL_NEURONS=$((NWORKERS * NEURONS_PER_WORKER))
    echo ""
    echo "======================================================================"
    echo "Test: $NNODES node(s), $NWORKERS GPUs -> $((TOTAL_NEURONS / 1000))K neurons"
    echo "======================================================================"

    # Run with output filtered to show only important info
    srun -N$NNODES -n$NWORKERS -c7 --ntasks-per-gpu=1 --gpu-bind=closest \
        python weak_scaling_const_comm.py \
        --neurons-per-worker $NEURONS_PER_WORKER \
        --ticks $TICKS \
        --update-ticks $UPDATE_TICKS \
        --intra-cluster-degree $INTRA_DEGREE \
        --cross-cluster-edges $CROSS_CLUSTER_EDGES \
        --num-neighbor-clusters $NUM_NEIGHBOR_CLUSTERS \
        2>&1 | grep -E "(WEAK SCALING|Network Size|Simulation time|SUCCESS|ERROR|Edge cut|agents \()"

    echo ""
done

echo ""
echo "======================================================================"
echo "All tests completed!"
echo "======================================================================"
