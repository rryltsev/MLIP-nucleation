#!/bin/bash
set -eo pipefail
umask 077
cd "$MLIP_RUN_DIR"
mkdir -p cache tmp
export XDG_CACHE_HOME="$PWD/cache"
export TMPDIR="$PWD/tmp"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=1
export DP_INTRA_OP_PARALLELISM_THREADS=4
export DP_INTER_OP_PARALLELISM_THREADS=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export DP_COMPILE=0
export DP_COMPILE_INFER=0
export DP_TRITON_INFER=0
hostname > node.txt
nvidia-smi > nvidia-smi.txt 2>&1
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > gpu.csv
source /home/u3129/miniconda3/etc/profile.d/conda.sh
conda activate "$MLIP_ENV"
python -m pip freeze > pip-freeze.txt
python mlip-smoke.py --backend "$MLIP_BACKEND"
