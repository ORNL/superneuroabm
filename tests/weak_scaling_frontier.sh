#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J weak_scaling
#SBATCH -o /lustre/orion/lrn088/proj-shared/objective3/xxz/superneuroabm/tests/output/weak_scaling_%j.out
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --gpus=8

# Weak scaling test on Frontier
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
NEURONS_PER_WORKER=10000  # Large per-worker to demonstrate memory limits
TICKS=10                  # Small number for quick comparison
UPDATE_TICKS=5            # Sync every 5 ticks

echo "======================================================================"
echo "Memory Scaling Test on Frontier"
echo "======================================================================"
echo "Goal: Demonstrate that MORE WORKERS enable LARGER NETWORKS"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs: $SLURM_GPUS"
echo "Neurons per worker: $NEURONS_PER_WORKER (constant)"
echo "Simulation ticks: $TICKS"
echo "======================================================================"

# Start with smallest configuration that might hit memory limit
# Then show that adding workers enables larger networks

# Test with progressively more workers
for NWORKERS in 1 2; do
    TOTAL_NEURONS=$((NWORKERS * NEURONS_PER_WORKER))
    echo ""
    echo "======================================================================"
    echo "Test: $NWORKERS worker(s) -> ${TOTAL_NEURONS} neurons total"
    echo "======================================================================"

    if [ $NWORKERS -eq 1 ]; then
        # Single worker
        python weak_scaling_lif.py \
            --neurons-per-worker $NEURONS_PER_WORKER \
            --ticks $TICKS \
            --update-ticks $UPDATE_TICKS
    else
        # Multiple workers - use srun
        srun -n$NWORKERS -c7 --gpus-per-task=1 --gpu-bind=closest \
            python weak_scaling_lif.py \
            --neurons-per-worker $NEURONS_PER_WORKER \
            --ticks $TICKS \
            --update-ticks $UPDATE_TICKS
    fi

    echo ""
done

echo ""
echo "======================================================================"
echo "All tests completed!"
echo "======================================================================"
