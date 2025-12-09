#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J 4_Kernel_2_2_Conv
#SBATCH -o %x-%j.out
#SBATCH -t 24:00:00
#SBATCH -p extended
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8

set -euo pipefail

# --- Output directories ---
RUN_DIR=/lustre/orion/lrn088/proj-shared/objective3/wfishell/runs/2_Layer_4_Kernel_2_2_ConvNet
OUTPUT_DIR=/lustre/orion/lrn088/proj-shared/objective3/wfishell/output/2_Layer_4_Kernel_2_2_ConvNet
MODEL_PATH=/lustre/orion/lrn088/proj-shared/objective3/wfishell/superneuroabm/superneuroabm/ssn/InitialModel/2_Layer_4_Kernel_2_2_ConvNet.json
RUN_NUM=0

# Create directories if they don't exist
mkdir -p "$RUN_DIR"
mkdir -p "$OUTPUT_DIR"

# Move SLURM output file to runs directory
mv "${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" "$RUN_DIR/" 2>/dev/null || true

# --- Environment setup ---
module load PrgEnv-gnu/8.6.0
module load miniforge3/23.11.0-0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a

source activate /lustre/orion/lrn088/proj-shared/objective3/envs/will_working_enviorment_cupy_13.6

# --- Performance optimization: MIOpen cache (speeds up kernel compilation) ---
export MIOPEN_USER_DB_PATH="/tmp/miopen-cache-$$"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

# NMNIST Data
NMNIST_DIR=/lustre/orion/lrn088/proj-shared/objective3/wfishell/superneuroabm/superneuroabm/ssn/data/NMNIST/Train

echo "=== Single GPU HIPT Attention Extraction ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Using 1 GPU for sequential processing"
echo "Output directory: $OUTPUT_DIR"
echo "Run logs: $RUN_DIR"

# --- Run Conv2dtNet tests ---
echo "=== Running Conv2dtNet Output Layer Weight Tests ==="

python /lustre/orion/lrn088/proj-shared/objective3/wfishell/superneuroabm/superneuroabm/ssn/test_scnn_nmnist_json.py \
    --data_path "$NMNIST_DIR" \
    --output_path "$OUTPUT_DIR" \
    --model_path "$MODEL_PATH" \
    --resume_from_run "$RUN_NUM"

echo "Job complete. Attention visualizations saved to $OUTPUT_DIR"
