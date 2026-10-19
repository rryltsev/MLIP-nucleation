#!/bin/bash
#SBATCH --partition=apollo
#SBATCH --account=imet-2026-2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
set -eo pipefail
umask 077
cd "$PILOT_RUN_DIR"
source /home/u3129/.local/opt/vasp/6.6.1-oneapi-2025/env.sh
source /home/u3129/.local/share/vasp/pseudopotentials/64-b3b00158ad69/env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python al-pilot.py "$PILOT_PHASE" ${PILOT_SOURCE:+--source "$PILOT_SOURCE"}
