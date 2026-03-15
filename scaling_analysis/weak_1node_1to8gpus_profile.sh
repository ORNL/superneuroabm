#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J weak_1node_profile
#SBATCH -o /lustre/orion/lrn088/proj-shared/objective3/xxz/superneuroabm/scaling_analysis/outputs/weak_1node_profile_%j.out
#SBATCH -t 02:00:00
#SBATCH -q debug
#SBATCH -N 1

# Weak scaling test on Frontier - 1 node, 1-8 GPUs WITH ROCPROF PROFILING
# Per-rank GPU profiling enabled to analyze NCCL communication and per-worker performance
# Cross-worker edges per worker remains constant as network scales

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

# Enable GPU-aware MPI for direct GPU-to-GPU communication via GPU-Direct RDMA
# SAGESim automatically detects this and uses optimized GPU communication paths
export MPICH_GPU_SUPPORT_ENABLED=1

# Go to scaling_analysis directory (FIX: was incorrectly pointing to tests/)
cd /lustre/orion/lrn088/proj-shared/objective3/xxz/superneuroabm/scaling_analysis

# Create outputs directory
mkdir -p outputs

# Configuration - FULL WEAK SCALING TEST (1-10 nodes, 8-80 GPUs)
NEURONS_PER_WORKER=5000     # Neurons per worker (constant)
TICKS=50                    # Simulation ticks
UPDATE_TICKS=1              # Sync every tick
INTRA_DEGREE=10             # Constant degree per neuron (O(n) edges!)
NUM_NEIGHBOR_CLUSTERS=1     # Directed ring topology (1 = TRUE weak scaling)

# Test multiple cross-cluster edge densities
CROSS_CLUSTER_EDGES_ARRAY=(1000 2000 3000 4000 5000)

# Create results CSV file with header
RESULTS_FILE="outputs/weak_scaling_profile_${SLURM_JOB_ID}.csv"
echo "timestamp,cross_cluster_edges,cross_cluster_percent,num_nodes,num_workers,total_neurons,total_edges,simulation_ticks,simulation_time_sec,status" > $RESULTS_FILE
echo "Results will be saved to: $RESULTS_FILE"

# Create profiles directory
mkdir -p outputs/profiles

echo "======================================================================"
echo "Weak Scaling Test - WITH GPU PROFILING (Directed Ring)"
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
echo "Cross-cluster edge densities to test: ${CROSS_CLUSTER_EDGES_ARRAY[@]}"
echo "Simulation ticks: $TICKS"
echo "GPU-aware MPI: $([ $MPICH_GPU_SUPPORT_ENABLED -eq 1 ] && echo 'ENABLED (GPU-Direct RDMA)' || echo 'DISABLED')"
echo "PROFILING: rocprofv3 enabled (per-rank profiling)"
echo "Profile output: outputs/profiles/<config>/rank_N.csv"
echo "======================================================================"

# Test multiple cross-cluster edge densities
for CROSS_CLUSTER_EDGES in "${CROSS_CLUSTER_EDGES_ARRAY[@]}"; do
    # Calculate cross-cluster percentage
    TOTAL_EDGES_PER_WORKER=$((NEURONS_PER_WORKER * INTRA_DEGREE + CROSS_CLUSTER_EDGES))
    CROSS_PERCENT=$(echo "scale=2; 100 * $CROSS_CLUSTER_EDGES / $TOTAL_EDGES_PER_WORKER" | bc)

    echo ""
    echo "##################################################################"
    echo "## TESTING: CROSS_CLUSTER_EDGES = $CROSS_CLUSTER_EDGES (~${CROSS_PERCENT}% of edges)"
    echo "##################################################################"
    echo "Expected edges per worker: ~$TOTAL_EDGES_PER_WORKER"
    echo "  - Intra-cluster: ~$((NEURONS_PER_WORKER * INTRA_DEGREE))"
    echo "  - Cross-cluster: $CROSS_CLUSTER_EDGES"
    echo ""

    # Test with 1 node, 1-8 GPUs
    for NGPUS in 1 2 3 4 5 6 7 8; do
        NNODES=1
        NWORKERS=$NGPUS
        TOTAL_NEURONS=$((NWORKERS * NEURONS_PER_WORKER))
        TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)

        echo ""
        echo "======================================================================"
        echo "Test: $NNODES node(s), $NWORKERS GPUs -> $((TOTAL_NEURONS / 1000))K neurons | Cross-edges: $CROSS_CLUSTER_EDGES"
        echo "======================================================================"

        # Create profile directory for this configuration
        PROFILE_DIR="outputs/profiles/${NWORKERS}gpu_${CROSS_CLUSTER_EDGES}e"
        mkdir -p $PROFILE_DIR
        echo "Profile directory: $PROFILE_DIR"

        # Run test with rocprofv3 per-rank profiling
        OUTPUT=$(srun -N$NNODES -n$NWORKERS -c7 --ntasks-per-gpu=1 --gpu-bind=closest \
            /opt/rocm-6.4.1/bin/rocprofv3 \
            --hip-trace --kernel-trace --memory-copy-trace \
            --output-directory ${PROFILE_DIR} \
            --output-file rank_%p \
            --output-format csv \
            -- python weak_scaling_const_comm.py \
            --neurons-per-worker $NEURONS_PER_WORKER \
            --ticks $TICKS \
            --update-ticks $UPDATE_TICKS \
            --intra-cluster-degree $INTRA_DEGREE \
            --cross-cluster-edges $CROSS_CLUSTER_EDGES \
            --num-neighbor-clusters $NUM_NEIGHBOR_CLUSTERS \
            2>&1) || {
            echo "ERROR: Test failed for NNODES=$NNODES, CROSS_CLUSTER_EDGES=$CROSS_CLUSTER_EDGES"
            echo "Full error output:"
            echo "$OUTPUT"
            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" "$TIMESTAMP" "$CROSS_CLUSTER_EDGES" "$CROSS_PERCENT" "$NNODES" "$NWORKERS" "$TOTAL_NEURONS" "NA" "$TICKS" "NA" "FAILED" >> $RESULTS_FILE
            echo "Continuing to next configuration..."
            continue
        }

        # Display filtered output
        echo "$OUTPUT" | grep -E "(WEAK SCALING|Network Size|Simulation time|SUCCESS|ERROR|Edge cut|agents \()"

        # Extract simulation time and total edges from output
        SIM_TIME=$(echo "$OUTPUT" | grep "Simulation time:" | sed -E 's/.*Simulation time: ([0-9.]+)s.*/\1/' | head -1)
        TOTAL_EDGES=$(echo "$OUTPUT" | grep "Total edges:" | sed -E 's/.*Total edges: ([0-9,]+).*/\1/' | tr -d ',' | head -1)

        # Handle empty extractions
        if [ -z "$SIM_TIME" ]; then
            SIM_TIME="NA"
        fi
        if [ -z "$TOTAL_EDGES" ]; then
            TOTAL_EDGES="NA"
        fi

        # Check if test succeeded
        if echo "$OUTPUT" | grep -q "SUCCESS"; then
            STATUS="SUCCESS"
        else
            STATUS="FAILED"
        fi

        # Save results to CSV (append after each test) - use printf to avoid newline issues
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" "$TIMESTAMP" "$CROSS_CLUSTER_EDGES" "$CROSS_PERCENT" "$NNODES" "$NWORKERS" "$TOTAL_NEURONS" "$TOTAL_EDGES" "$TICKS" "$SIM_TIME" "$STATUS" >> $RESULTS_FILE
        echo "Results saved to $RESULTS_FILE"

        # Count and report profile files
        PROFILE_COUNT=$(ls -1 ${PROFILE_DIR}/rank_*.csv 2>/dev/null | wc -l)
        if [ $PROFILE_COUNT -gt 0 ]; then
            echo "Generated $PROFILE_COUNT profile files in $PROFILE_DIR"
        fi

        echo ""
    done

    echo ""
    echo "## Completed: CROSS_CLUSTER_EDGES = $CROSS_CLUSTER_EDGES"
    echo ""
done

echo ""
echo "======================================================================"
echo "All tests completed!"
echo "======================================================================"
echo ""
echo "Profile files location: outputs/profiles/"
echo "Summary of profile directories:"
ls -lh outputs/profiles/ | tail -n +2
echo ""
echo "To analyze profiles:"
echo "  - CSV files: outputs/profiles/<config>/rank_N.csv"
echo "  - JSON traces: outputs/profiles/<config>/rank_N.json"
echo "======================================================================"
