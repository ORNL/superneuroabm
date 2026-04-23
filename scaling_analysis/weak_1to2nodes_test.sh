#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J weak_1to2_test
#SBATCH -o /lustre/orion/lrn088/proj-shared/objective3/xxz/superneuroabm/scaling_analysis/outputs/weak_1to2_test_%j.out
#SBATCH -t 00:30:00
#SBATCH -q debug
#SBATCH -N 2

unset SLURM_EXPORT_ENV

module load cpe/26.03
module load PrgEnv-gnu
module load miniforge3
module load rocm/7.2.0
module load craype-accel-amd-gfx90a

source activate /lustre/orion/proj-shared/lrn088/objective3/envs/superneuroabm_env_cupy14

export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

# CuPy and profiling output — keep side by side
WORK_DIR=/lustre/orion/lrn088/proj-shared/objective3/xxz/superneuroabm/scaling_analysis
export CUPY_CACHE_DIR=${WORK_DIR}/outputs/cupy-cache
export CUPY_CACHE_SAVE_CUDA_SOURCE=1
export ROCPROF_OUTPUT_DIR=${WORK_DIR}/outputs/rocprof_${SLURM_JOB_ID}

cd ${WORK_DIR}
mkdir -p outputs outputs/cupy-cache

NEURONS_PER_WORKER=5000
TICKS=50
UPDATE_TICKS=1
INTRA_DEGREE=10
NUM_NEIGHBOR_CLUSTERS=1
CROSS_CLUSTER_EDGES=5000

SHARED_CSV="outputs/weak_1to2_test_${SLURM_JOB_ID}.csv"

echo "======================================================================"
echo "Quick Test - 1 and 2 Nodes"
echo "======================================================================"

for NNODES in 1 2; do
    NWORKERS=$((NNODES * 8))
    TOTAL_NEURONS=$((NWORKERS * NEURONS_PER_WORKER))

    echo ""
    echo "======================================================================"
    echo "Testing: $NNODES nodes, $NWORKERS GPUs, $TOTAL_NEURONS neurons"
    echo "======================================================================"
    echo "Starting at: $(date)"

    set +e
    OUTPUT=$(srun -N$NNODES -n$NWORKERS -c7 --ntasks-per-gpu=1 --gpu-bind=closest \
        python weak_scaling_ring_topology.py \
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

    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Test failed for $NNODES nodes (exit code: $EXIT_CODE)"
        echo "$OUTPUT"
        continue
    fi

    echo "$OUTPUT" | grep -E "(WEAK SCALING|Network Size|Simulation time|SUCCESS|ERROR|TIMING|Total)"
    echo "Completed: $NNODES nodes"
done

echo ""
echo "======================================================================"
echo "TEST COMPLETED"
echo "======================================================================"
echo "Results: $SHARED_CSV"
cat $SHARED_CSV
