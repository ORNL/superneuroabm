#!/bin/bash
#SBATCH -A csc536
#SBATCH -J sagesim_sir
#SBATCH -o logs/sagesim_sir_%j.o
#SBATCH -e logs/sagesim_sir_%j.e
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 4

# Only necessary if submitting like: sbatch --export=NONE ... (recommended)
# Do NOT include this line when submitting without --export=NONE
unset SLURM_EXPORT_ENV


# Load modules
module load PrgEnv-gnu/8.6.0
module load cray-hdf5-parallel/1.12.2.11
module load miniforge3/23.11.0-0
module load rocm/5.7.1
module load craype-accel-amd-gfx90a


# Activate your environment
source activate /lustre/orion/csc536/proj-shared/.conda/pkgs/sagesimenv/

# Point to source
export SRC_DIR=/lustre/orion/proj-shared/csc536/SAGESim/examples/sir/
MODULE_DIR=/lustre/orion/proj-shared/csc536/SAGESim/
export PYTHONPATH=${MODULE_DIR}:$PYTHONPATH

# Make run dir if not exists per job id
RUN_DIR=/lustre/orion/proj-shared/csc536/SAGESim/examples/sir/runs
if [ ! -d "$RUN_DIR" ]
then
        mkdir -p $RUN_DIR
fi
cd $RUN_DIR


# Run script
echo Running Python Script
time srun -N4 -n30 -c7 --gpus-per-task=1 --gpu-bind=closest python3 -u ${SRC_DIR}/run.py

echo Run Finished
date


