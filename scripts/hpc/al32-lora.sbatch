#!/bin/bash
set -eo pipefail
umask 077
ulimit -c 0
cd "$FINETUNE_RUN_DIR"
mkdir cache dpa4
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$PWD/cache" TORCHINDUCTOR_CACHE_DIR="$PWD/cache/inductor" TRITON_CACHE_DIR="$PWD/cache/triton"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MAX_JOBS=4
export DP_TF32_INFER=0 DP_AMP_INFER=0 DP_TRITON_INFER=0 DP_COMPILE_INFER=0
hostname > node.txt
nvidia-smi > nvidia-smi.txt
source /home/u3129/miniconda3/etc/profile.d/conda.sh
conda activate lammps_deepmd_dpa4
export CUDA_HOME=/opt/cuda-13.0 CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python al32-lora-run.py > lora-console.txt 2>&1
