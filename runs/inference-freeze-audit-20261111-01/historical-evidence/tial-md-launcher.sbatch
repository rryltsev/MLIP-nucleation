#!/bin/bash
set -eo pipefail
umask 077
ulimit -c 0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export DP_INTRA_OP_PARALLELISM_THREADS=4 DP_INTER_OP_PARALLELISM_THREADS=1
export DP_TF32_INFER=0 DP_AMP_INFER=0 DP_TRITON_INFER=0 DP_COMPILE_INFER=0
export TORCHINDUCTOR_COMPILE_THREADS=4 LOCAL_RANK=0
unset DEVICE
source /home/u3129/miniconda3/etc/profile.d/conda.sh
conda activate lammps_deepmd_dpa4
export CUDA_HOME=/opt/cuda-13.0
export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="$PWD/cache"
export TRITON_CACHE_DIR="$PWD/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/inductor"
export PYTORCH_KERNEL_CACHE_PATH="$PWD/cache/torch/kernels"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$PYTORCH_KERNEL_CACHE_PATH"
nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu --format=csv > gpu-at-start.txt
srun --exclusive --exact -N1 -n1 -c4 --gres=gpu:a100:1 --mem=28G --cpu-bind=cores python -u workflow.py preflight > preflight.log 2>&1
srun --exclusive --exact -N1 -n1 -c4 --gres=gpu:a100:1 --mem=28G --cpu-bind=cores python -u workflow.py worker 0 > worker-0.log 2>&1
