#!/bin/bash
#SBATCH -A csc536
#SBATCH -J superneuroabm_logic_gates
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

# Logan Gillum: You are likely seeing this error because ROCm 5 is not our default anymore. You will need to add this export to pickup on the older ROCm libraries:
# export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

# Activate your environment
source activate /lustre/orion/csc536/proj-shared/SAGESim/examples/sir/.conda/pkgs/sagesimenv

# Point to source
export SRC_DIR=/lustre/orion/proj-shared/csc536/superneuroabm/examples/frontier_logic_gates/
# SAGESIM_DIR=/lustre/orion/proj-shared/csc536/SAGESim/
# SUPERNEUROABM_DIR=/lustre/orion/proj-shared/csc536/superneuroabm/
# export PYTHONPATH=${SUPERNEUROABM_DIR}:$PYTHONPATH

# Make run dir if not exists per job id
RUN_DIR=/lustre/orion/proj-shared/csc536/superneuroabm/examples/frontier_logic_gates/
if [ ! -d "$RUN_DIR" ]
then
        mkdir -p $RUN_DIR
fi
cd $RUN_DIR


# Run script
echo Running Python Script
time srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest python -u ${SRC_DIR}/logic_gates.py

echo Run Finished
date


