#!/bin/bash
set -eo pipefail
umask 077
cd "$DEPLOY_RUN_DIR"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
source /home6/u3129/mlip-research/envs/grace_ft_a100/activate-grace.sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/tensorflow:${LD_LIBRARY_PATH:-}"
if [[ "${LMP_BUILD_VARIANT:-grace}" == deepmd-mpi ]]; then
  source /misc/umt_shared/opt/intel/oneapi/mpi/2021.17/env/vars.sh
  export LD_LIBRARY_PATH="$I_MPI_ROOT/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  export I_MPI_CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
  export I_MPI_CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
  export I_MPI_FABRICS=shm
fi
python build-grace-lammps.py > build-console.txt 2>&1
