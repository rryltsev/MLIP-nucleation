#!/bin/bash
set -eo pipefail
umask 077
ulimit -c 0
cd "$SLURM_SUBMIT_DIR"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1
export DP_TF32_INFER=0 DP_AMP_INFER=0 DP_TRITON_INFER=0 DP_COMPILE_INFER=0
# Keep compiler caches off the nearly full home6 quota. Preserve outputs in the run.
export TMPDIR="/tmp/u3129-inference-benchmark-$SLURM_JOB_ID"
mkdir -m 700 "$TMPDIR"
export XDG_CACHE_HOME="$TMPDIR/cache" TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor" TRITON_CACHE_DIR="$TMPDIR/triton"
export TORCHINDUCTOR_COMPILE_THREADS=4 MAX_JOBS=4
source /home/u3129/miniconda3/etc/profile.d/conda.sh
conda activate lammps_deepmd_dpa4
export CUDA_HOME=/opt/cuda-13.0
export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
df -h "$TMPDIR" . > storage.txt
srun --cpu-bind=cores python runner.py
