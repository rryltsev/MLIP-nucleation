#!/bin/bash
set -eo pipefail
umask 077
cd "$GRACE_RUN_DIR"
mkdir -p cache
export XDG_CACHE_HOME="$PWD/cache"
export GRACE_CACHE="$PWD/cache/grace"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export TF_USE_LEGACY_KERAS=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export GRACE_CUDA_ROOT="$GRACE_ENV/lib/python3.11/site-packages/nvidia/cuda_nvcc"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$GRACE_CUDA_ROOT"
export PATH="$GRACE_CUDA_ROOT/bin:$PATH"
test -r "$GRACE_CUDA_ROOT/nvvm/libdevice/libdevice.10.bc"
hostname > node.txt
nvidia-smi > nvidia-smi.txt 2>&1
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > gpu.csv
"$GRACE_ENV/bin/python" -m pip freeze --all > pip-freeze.txt
"$GRACE_ENV/bin/python" grace-smoke.py
