#!/bin/bash
set -eo pipefail
umask 077
cd "$DEPLOY_RUN_DIR"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
source /home6/u3129/mlip-research/envs/grace_ft_a100/activate-grace.sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/tensorflow:${LD_LIBRARY_PATH:-}"
python build-grace-lammps.py > build-console.txt 2>&1
