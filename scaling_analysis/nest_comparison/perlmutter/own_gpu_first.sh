#!/bin/bash
# Keep every GPU of the step visible (CUDA IPC between ranks needs the peer's GPU),
# but put this rank's own GPU first so it is device 0.  usage: own_gpu_first.sh cmd...
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  n=$(nvidia-smi -L | wc -l); CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((n-1)))
fi
IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
n=${#G[@]}; i=$(( ${SLURM_LOCALID:-0} % n ))
order=("${G[@]:i}" "${G[@]:0:i}")
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${order[*]}")
exec "$@"
