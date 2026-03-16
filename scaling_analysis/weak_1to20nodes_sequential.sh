#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J weak_1to20_seq
#SBATCH -o /lustre/orion/lrn088/proj-shared/objective3/xxz/superneuroabm/scaling_analysis/outputs/weak_1to20_seq_%j.out
#SBATCH -t 02:00:00
#SBATCH -q debug
#SBATCH -N 20

# Weak scaling test: 1-20 nodes sequential
# This single job tests all configurations from 1 to 20 nodes

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

# Create output directory
mkdir -p outputs

# Configuration
NEURONS_PER_WORKER=5000
TICKS=50
UPDATE_TICKS=1
INTRA_DEGREE=10
NUM_NEIGHBOR_CLUSTERS=1
CROSS_CLUSTER_EDGES_ARRAY=(5000)  # Test only highest edge density for production runs

# Create shared CSV file for all timing results (will be appended to by Python script)
SHARED_CSV="outputs/weak_1to20_seq_${SLURM_JOB_ID}.csv"

echo "======================================================================"
echo "Weak Scaling Test - 1 to 20 Nodes Sequential"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated nodes: 20"
echo "Testing: 1, 2, 3, ..., 20 nodes"
echo "Cross-cluster edges to test: ${CROSS_CLUSTER_EDGES_ARRAY[@]}"
echo "Results file: $SHARED_CSV"
echo "======================================================================"

# Loop through cross-cluster edge densities
for CROSS_CLUSTER_EDGES in "${CROSS_CLUSTER_EDGES_ARRAY[@]}"; do
    # Calculate cross-cluster percentage
    TOTAL_EDGES_PER_WORKER=$((NEURONS_PER_WORKER * INTRA_DEGREE + CROSS_CLUSTER_EDGES))
    CROSS_PERCENT=$(echo "scale=2; 100 * $CROSS_CLUSTER_EDGES / $TOTAL_EDGES_PER_WORKER" | bc)

    echo ""
    echo "##################################################################"
    echo "## TESTING: CROSS_CLUSTER_EDGES = $CROSS_CLUSTER_EDGES (~${CROSS_PERCENT}% of edges)"
    echo "##################################################################"
    echo ""

    # Loop through node counts from 1 to 20
    for NNODES in {1..20}; do
        NWORKERS=$((NNODES * 8))
        TOTAL_NEURONS=$((NWORKERS * NEURONS_PER_WORKER))

        echo ""
        echo "======================================================================"
        echo "Testing: $NNODES nodes, $NWORKERS GPUs, $TOTAL_NEURONS neurons | Cross-edges: $CROSS_CLUSTER_EDGES"
        echo "======================================================================"
        echo "Starting at: $(date)"

        # Run test - Python script will append to shared CSV
        set +e
        OUTPUT=$(srun -N$NNODES -n$NWORKERS -c7 --ntasks-per-gpu=1 --gpu-bind=closest \
            python weak_scaling_const_comm.py \
            --neurons-per-worker $NEURONS_PER_WORKER \
            --ticks $TICKS \
            --update-ticks $UPDATE_TICKS \
            --intra-cluster-degree $INTRA_DEGREE \
            --cross-cluster-edges $CROSS_CLUSTER_EDGES \
            --num-neighbor-clusters $NUM_NEIGHBOR_CLUSTERS \
            --csv $SHARED_CSV \
            2>&1)
        EXIT_CODE=$?
        set -e

        # Check if test failed
        if [ $EXIT_CODE -ne 0 ]; then
            echo "=========================================="
            echo "ERROR: Test failed for $NNODES nodes, $CROSS_CLUSTER_EDGES edges"
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
        echo "$OUTPUT" | grep -E "(WEAK SCALING|Network Size|Simulation time|SUCCESS|ERROR|Edge cut|agents \(|TIMING|Rank|Metric|Straggler|MPI Traffic|Grid Barriers)"

        echo "Completed: $NNODES nodes, $CROSS_CLUSTER_EDGES edges"
        echo ""
    done

    echo ""
    echo "## Completed: CROSS_CLUSTER_EDGES = $CROSS_CLUSTER_EDGES"
    echo ""
done

echo ""
echo "======================================================================"
echo "ALL TESTS COMPLETED (1-20 nodes)!"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Results file: $SHARED_CSV"
echo "======================================================================"
echo ""
echo "Summary of results:"
cat $SHARED_CSV
