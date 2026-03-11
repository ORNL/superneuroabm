#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J weak_1to10_multiedge
#SBATCH -o /lustre/orion/lrn088/proj-shared/objective3/xxz/superneuroabm/scaling_analysis/outputs/weak_1to10_multiedge_%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 10

# Weak scaling test: 1-10 nodes, multiple cross-cluster edge densities
# This single job tests all configurations from 1 to 10 nodes with 5 edge densities

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
CROSS_CLUSTER_EDGES_ARRAY=(1000 2000 3000 4000 5000)

# Create results CSV file
RESULTS_FILE="outputs/weak_1to10_multiedge_${SLURM_JOB_ID}.csv"
echo "timestamp,cross_cluster_edges,cross_cluster_percent,num_nodes,num_workers,total_neurons,total_edges,simulation_ticks,simulation_time_sec,status" > $RESULTS_FILE

echo "======================================================================"
echo "Weak Scaling Test - 1 to 10 Nodes (Multiple Edge Densities)"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated nodes: 10"
echo "Testing: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 nodes"
echo "Cross-cluster edges to test: ${CROSS_CLUSTER_EDGES_ARRAY[@]}"
echo "Results file: $RESULTS_FILE"
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

    # Loop through node counts
    for NNODES in 1 2 3 4 5 6 7 8 9 10; do
        NWORKERS=$((NNODES * 8))
        TOTAL_NEURONS=$((NWORKERS * NEURONS_PER_WORKER))
        TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)

        echo ""
        echo "======================================================================"
        echo "Testing: $NNODES nodes, $NWORKERS GPUs, $TOTAL_NEURONS neurons | Cross-edges: $CROSS_CLUSTER_EDGES"
        echo "======================================================================"

        # Run test
        OUTPUT=$(srun -N$NNODES -n$NWORKERS -c7 --ntasks-per-gpu=1 --gpu-bind=closest \
            python weak_scaling_const_comm.py \
            --neurons-per-worker $NEURONS_PER_WORKER \
            --ticks $TICKS \
            --update-ticks $UPDATE_TICKS \
            --intra-cluster-degree $INTRA_DEGREE \
            --cross-cluster-edges $CROSS_CLUSTER_EDGES \
            --num-neighbor-clusters $NUM_NEIGHBOR_CLUSTERS \
            2>&1) || {
            echo "ERROR: Test failed for $NNODES nodes, $CROSS_CLUSTER_EDGES edges"
            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" "$TIMESTAMP" "$CROSS_CLUSTER_EDGES" "$CROSS_PERCENT" "$NNODES" "$NWORKERS" "$TOTAL_NEURONS" "NA" "$TICKS" "NA" "FAILED" >> $RESULTS_FILE
            echo "Continuing to next configuration..."
            continue
        }

        # Display filtered output
        echo "$OUTPUT" | grep -E "(WEAK SCALING|Network Size|Simulation time|SUCCESS|ERROR|Edge cut)"

        # Extract results
        SIM_TIME=$(echo "$OUTPUT" | grep "Simulation time:" | sed -E 's/.*Simulation time: ([0-9.]+)s.*/\1/' | head -1)
        TOTAL_EDGES=$(echo "$OUTPUT" | grep "Total edges:" | sed -E 's/.*Total edges: ([0-9,]+).*/\1/' | tr -d ',' | head -1)

        if [ -z "$SIM_TIME" ]; then
            SIM_TIME="NA"
        fi
        if [ -z "$TOTAL_EDGES" ]; then
            TOTAL_EDGES="NA"
        fi

        if echo "$OUTPUT" | grep -q "SUCCESS"; then
            STATUS="SUCCESS"
        else
            STATUS="FAILED"
        fi

        # Save results
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" "$TIMESTAMP" "$CROSS_CLUSTER_EDGES" "$CROSS_PERCENT" "$NNODES" "$NWORKERS" "$TOTAL_NEURONS" "$TOTAL_EDGES" "$TICKS" "$SIM_TIME" "$STATUS" >> $RESULTS_FILE
        echo "Results saved to $RESULTS_FILE"

        echo "Completed: $NNODES nodes, $CROSS_CLUSTER_EDGES edges"
        echo ""
    done

    echo ""
    echo "## Completed: CROSS_CLUSTER_EDGES = $CROSS_CLUSTER_EDGES"
    echo ""
done

echo ""
echo "======================================================================"
echo "ALL TESTS COMPLETED (1-10 nodes)!"
echo "======================================================================"
echo "Final results:"
cat $RESULTS_FILE
