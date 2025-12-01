#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J weak_scaling_constcomm
#SBATCH -o /lustre/orion/lrn088/proj-shared/objective3/xxz/superneuroabm/tests/output/weak_scaling_constcomm_%j.out
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --gpus=4

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

# Configuration
# Memory limit found: 20K neurons max, failed at 30K
# Using 50% of limit for safety = 10K neurons per worker
NEURONS_PER_WORKER=10000    # 50% of single-GPU limit (20K max)
TICKS=10                    # Small number for quick comparison
UPDATE_TICKS=5              # Sync every 5 ticks
INTRA_PROB=0.01             # Intra-cluster connection probability (1%)
CROSS_CLUSTER_EDGES=2000    # Total outgoing edges per worker (constant)

echo "======================================================================"
echo "Weak Scaling Test - Constant Communication Model"
echo "======================================================================"
echo "Goal: Demonstrate that MORE WORKERS enable LARGER NETWORKS"
echo "      with CONSTANT cross-worker communication overhead"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs: $SLURM_GPUS"
echo "Neurons per worker: $NEURONS_PER_WORKER (constant)"
echo "Simulation ticks: $TICKS"
echo "Intra-cluster prob: $INTRA_PROB"
echo "Cross-cluster edges per worker: $CROSS_CLUSTER_EDGES (constant)"
echo "======================================================================"

# Compare 2 vs 4 workers (both have MPI overhead)
# Avoid comparing against 1 worker since it bypasses contextualization

# Test with progressively more workers
for NWORKERS in 2 4; do
    TOTAL_NEURONS=$((NWORKERS * NEURONS_PER_WORKER))
    echo ""
    echo "======================================================================"
    echo "Test: $NWORKERS worker(s) -> ${TOTAL_NEURONS} neurons total"
    echo "======================================================================"

    # Use srun for all worker counts (more stable)
    srun -n$NWORKERS -c7 --gpus-per-task=1 --gpu-bind=closest \
        python weak_scaling_const_comm.py \
        --neurons-per-worker $NEURONS_PER_WORKER \
        --ticks $TICKS \
        --update-ticks $UPDATE_TICKS \
        --intra-cluster-prob $INTRA_PROB \
        --cross-cluster-edges $CROSS_CLUSTER_EDGES

    echo ""
done

echo ""
echo "======================================================================"
echo "All tests completed!"
echo "======================================================================"
