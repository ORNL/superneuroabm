#!/bin/bash
# General-purpose script to submit M independent weak scaling jobs in parallel
# Usage: ./submit_weak_scaling_parallel.sh --start-nodes N --end-nodes M --step S --edges E [options]
#
# Example: ./submit_weak_scaling_parallel.sh --start-nodes 1 --end-nodes 200 --step 10 --edges 5000

# Default values
START_NODES=10
END_NODES=200
STEP=10
CROSS_CLUSTER_EDGES_ARRAY=(1000 2000 3000 4000 5000)  # Test multiple edge densities
NEURONS_PER_WORKER=5000
TICKS=50
UPDATE_TICKS=1
INTRA_DEGREE=10
NUM_NEIGHBOR_CLUSTERS=1  # Fixed at 1 for proper weak scaling (directed ring topology)

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --start-nodes)
            START_NODES="$2"
            shift 2
            ;;
        --end-nodes)
            END_NODES="$2"
            shift 2
            ;;
        --step)
            STEP="$2"
            shift 2
            ;;
        --neurons-per-worker)
            NEURONS_PER_WORKER="$2"
            shift 2
            ;;
        --ticks)
            TICKS="$2"
            shift 2
            ;;
        --update-ticks)
            UPDATE_TICKS="$2"
            shift 2
            ;;
        --intra-degree)
            INTRA_DEGREE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Submit multiple independent SLURM jobs for weak scaling tests."
            echo ""
            echo "Required options:"
            echo "  --start-nodes N          Starting number of nodes (default: 10)"
            echo "  --end-nodes M            Ending number of nodes (default: 200)"
            echo "  --step S                 Step size between node counts (default: 10)"
            echo ""
            echo "Optional configuration:"
            echo "  --neurons-per-worker N   Neurons per worker (default: 5000)"
            echo "  --ticks N                Simulation ticks (default: 50)"
            echo "  --update-ticks N         Update interval (default: 1)"
            echo "  --intra-degree N         Intra-cluster degree (default: 10)"
            echo ""
            echo "Examples:"
            echo "  $0 --start-nodes 1 --end-nodes 200 --step 10"
            echo "  $0 --start-nodes 10 --end-nodes 100 --step 5 --ticks 100"
            echo "  $0 --start-nodes 1 --end-nodes 10 --step 1"
            echo ""
            echo "Note: Each job tests multiple cross-cluster edge densities: ${CROSS_CLUSTER_EDGES_ARRAY[@]}"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate inputs
if [ $START_NODES -le 0 ] || [ $END_NODES -le 0 ] || [ $STEP -le 0 ]; then
    echo "ERROR: Node counts and step must be positive integers"
    exit 1
fi

if [ $START_NODES -gt $END_NODES ]; then
    echo "ERROR: Start nodes ($START_NODES) cannot be greater than end nodes ($END_NODES)"
    exit 1
fi

# Change to scaling_analysis directory
cd /lustre/orion/lrn088/proj-shared/objective3/xxz/superneuroabm/scaling_analysis

# Calculate number of jobs
NUM_JOBS=$(( (END_NODES - START_NODES) / STEP + 1 ))

echo "======================================================================"
echo "Submitting Independent Weak Scaling Jobs"
echo "======================================================================"
echo "Configuration:"
echo "  Node range: $START_NODES to $END_NODES (step: $STEP)"
echo "  Number of jobs: $NUM_JOBS"
echo "  Cross-cluster edges to test: ${CROSS_CLUSTER_EDGES_ARRAY[@]}"
echo "  Neurons per worker: $NEURONS_PER_WORKER"
echo "  Simulation ticks: $TICKS"
echo "======================================================================"
echo ""

JOB_COUNT=0
FAILED_COUNT=0

# Loop through node counts
for ((NODES=START_NODES; NODES<=END_NODES; NODES+=STEP)); do
    # Set wall time based on node count tiers
    if [ $NODES -le 91 ]; then
        WALLTIME="02:00:00"
    elif [ $NODES -le 183 ]; then
        WALLTIME="06:00:00"
    else
        WALLTIME="12:00:00"
    fi

    JOBNAME="weak_${NODES}n_multiedge"

    echo "Submitting job for $NODES nodes (walltime: $WALLTIME)..."

    # Create temporary job script
    cat > /tmp/job_${NODES}_multiedge.sh <<EOF
#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J ${JOBNAME}
#SBATCH -o /lustre/orion/lrn088/proj-shared/objective3/xxz/superneuroabm/scaling_analysis/outputs/weak_${NODES}n_multiedge_%j.out
#SBATCH -t ${WALLTIME}
#SBATCH -p batch
#SBATCH -N ${NODES}

# Weak scaling test - ${NODES} nodes, multiple cross-cluster edge densities
unset SLURM_EXPORT_ENV

# Load modules
module load PrgEnv-gnu/8.6.0
module load cray-hdf5-parallel/1.12.2.11
module load miniforge3/23.11.0-0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
module load metis/5.1.0

# Set up METIS library path
export LD_LIBRARY_PATH=/lustre/orion/lrn088/proj-shared/objective3/xxz/SAGESim/local_lib:\$LD_LIBRARY_PATH

# Activate environment
source activate /lustre/orion/proj-shared/lrn088/objective3/envs/superneuroabm_env_xxz

# Go to scaling_analysis directory
cd /lustre/orion/lrn088/proj-shared/objective3/xxz/superneuroabm/scaling_analysis

# Create output directory
mkdir -p outputs

# Configuration
NEURONS_PER_WORKER=${NEURONS_PER_WORKER}
TICKS=${TICKS}
UPDATE_TICKS=${UPDATE_TICKS}
INTRA_DEGREE=${INTRA_DEGREE}
NUM_NEIGHBOR_CLUSTERS=${NUM_NEIGHBOR_CLUSTERS}
CROSS_CLUSTER_EDGES_ARRAY=(${CROSS_CLUSTER_EDGES_ARRAY[@]})

# Create results CSV file
RESULTS_FILE="outputs/weak_${NODES}n_multiedge_\${SLURM_JOB_ID}.csv"
echo "timestamp,cross_cluster_edges,cross_cluster_percent,num_nodes,num_workers,total_neurons,total_edges,simulation_ticks,simulation_time_sec,status" > \$RESULTS_FILE

echo "======================================================================"
echo "Weak Scaling Test - ${NODES} Nodes (Multiple Edge Densities)"
echo "======================================================================"
echo "Job ID: \$SLURM_JOB_ID"
echo "Nodes: ${NODES}"
echo "GPUs: $((${NODES} * 8))"
echo "Total neurons: $((${NODES} * 8 * ${NEURONS_PER_WORKER}))"
echo "Cross-cluster edges to test: \${CROSS_CLUSTER_EDGES_ARRAY[@]}"
echo "Results file: \$RESULTS_FILE"
echo "======================================================================"

NNODES=${NODES}
NWORKERS=\$((NNODES * 8))
TOTAL_NEURONS=\$((NWORKERS * NEURONS_PER_WORKER))

# Loop through cross-cluster edge densities
for CROSS_CLUSTER_EDGES in "\${CROSS_CLUSTER_EDGES_ARRAY[@]}"; do
    TIMESTAMP=\$(date +%Y-%m-%d_%H:%M:%S)
    TOTAL_EDGES_PER_WORKER=\$((NEURONS_PER_WORKER * INTRA_DEGREE + CROSS_CLUSTER_EDGES))
    CROSS_PERCENT=\$(echo "scale=2; 100 * \$CROSS_CLUSTER_EDGES / \$TOTAL_EDGES_PER_WORKER" | bc)

    echo ""
    echo "##################################################################"
    echo "## TESTING: CROSS_CLUSTER_EDGES = \$CROSS_CLUSTER_EDGES (~\${CROSS_PERCENT}% of edges)"
    echo "##################################################################"
    echo ""

    # Run test
    OUTPUT=\$(srun -N\$NNODES -n\$NWORKERS -c7 --ntasks-per-gpu=1 --gpu-bind=closest \\
        python weak_scaling_const_comm.py \\
        --neurons-per-worker \$NEURONS_PER_WORKER \\
        --ticks \$TICKS \\
        --update-ticks \$UPDATE_TICKS \\
        --intra-cluster-degree \$INTRA_DEGREE \\
        --cross-cluster-edges \$CROSS_CLUSTER_EDGES \\
        --num-neighbor-clusters \$NUM_NEIGHBOR_CLUSTERS \\
        2>&1) || {
        echo "ERROR: Test failed for \$NNODES nodes, \$CROSS_CLUSTER_EDGES edges"
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" "\$TIMESTAMP" "\$CROSS_CLUSTER_EDGES" "\$CROSS_PERCENT" "\$NNODES" "\$NWORKERS" "\$TOTAL_NEURONS" "NA" "\$TICKS" "NA" "FAILED" >> \$RESULTS_FILE
        echo "Continuing to next edge density..."
        continue
    }

    # Display filtered output
    echo "\$OUTPUT" | grep -E "(WEAK SCALING|Network Size|Simulation time|SUCCESS|ERROR|Edge cut)"

    # Extract results
    SIM_TIME=\$(echo "\$OUTPUT" | grep "Simulation time:" | sed -E 's/.*Simulation time: ([0-9.]+)s.*/\1/' | head -1)
    TOTAL_EDGES=\$(echo "\$OUTPUT" | grep "Total edges:" | sed -E 's/.*Total edges: ([0-9,]+).*/\1/' | tr -d ',' | head -1)

    if [ -z "\$SIM_TIME" ]; then
        SIM_TIME="NA"
    fi
    if [ -z "\$TOTAL_EDGES" ]; then
        TOTAL_EDGES="NA"
    fi

    if echo "\$OUTPUT" | grep -q "SUCCESS"; then
        STATUS="SUCCESS"
    else
        STATUS="FAILED"
    fi

    # Save results
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" "\$TIMESTAMP" "\$CROSS_CLUSTER_EDGES" "\$CROSS_PERCENT" "\$NNODES" "\$NWORKERS" "\$TOTAL_NEURONS" "\$TOTAL_EDGES" "\$TICKS" "\$SIM_TIME" "\$STATUS" >> \$RESULTS_FILE
    echo "Results saved to \$RESULTS_FILE"

    echo ""
    echo "## Completed: CROSS_CLUSTER_EDGES = \$CROSS_CLUSTER_EDGES"
    echo ""
done

echo ""
echo "======================================================================"
echo "ALL EDGE DENSITIES COMPLETED FOR ${NODES} NODES!"
echo "======================================================================"
echo "Final results:"
cat \$RESULTS_FILE
EOF

    # Submit the job
    SUBMIT_OUTPUT=$(sbatch /tmp/job_${NODES}_multiedge.sh 2>&1)
    JOB_ID=$(echo "$SUBMIT_OUTPUT" | awk '{print $4}')

    if [ ! -z "$JOB_ID" ] && [[ "$SUBMIT_OUTPUT" == *"Submitted batch job"* ]]; then
        echo "  Job ID: $JOB_ID"
        JOB_COUNT=$((JOB_COUNT + 1))
    else
        echo "  ERROR: Failed to submit job"
        echo "  Output: $SUBMIT_OUTPUT"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
    echo ""

    # Clean up temp file
    rm -f /tmp/job_${NODES}_multiedge.sh
done

echo ""
echo "======================================================================"
echo "Submission Complete!"
echo "======================================================================"
echo "Successfully submitted: $JOB_COUNT / $NUM_JOBS jobs"
if [ $FAILED_COUNT -gt 0 ]; then
    echo "Failed submissions: $FAILED_COUNT"
fi
echo ""
echo "Check status with: squeue -u \$USER"
echo ""
echo "Results will be saved to:"
echo "  outputs/weak_<N>n_multiedge_<jobid>.csv"
echo ""
echo "Node configurations tested:"
for ((NODES=START_NODES; NODES<=END_NODES; NODES+=STEP)); do
    echo "  - ${NODES} nodes ($((NODES * 8)) GPUs, $((NODES * 8 * NEURONS_PER_WORKER)) neurons)"
done
echo ""
echo "Each job tests ${#CROSS_CLUSTER_EDGES_ARRAY[@]} edge densities: ${CROSS_CLUSTER_EDGES_ARRAY[@]}"
