#!/bin/bash
#SBATCH --partition=apollo
#SBATCH --account=imet-2026-2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
set -eo pipefail
umask 077
cd "$PILOT_RUN_DIR"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python kplib-run.py setup
