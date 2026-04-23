#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J weak_1to2_test
#SBATCH -o /lustre/orion/lrn088/proj-shared/objective3/xxz/superneuroabm/scaling_analysis/outputs/weak_1to2_test_%j.out
#SBATCH -t 00:30:00
#SBATCH -q debug
#SBATCH -N 2

unset SLURM_EXPORT_ENV

module load PrgEnv-gnu/8.6.0
module load cray-hdf5-parallel/1.12.2.11
module load miniforge3/23.11.0-0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a

source activate /lustre/orion/proj-shared/lrn088/objective3/envs/superneuroabm_env_xxz

cd /lustre/orion/lrn088/proj-shared/objective3/xxz/superneuroabm/scaling_analysis
mkdir -p outputs

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
