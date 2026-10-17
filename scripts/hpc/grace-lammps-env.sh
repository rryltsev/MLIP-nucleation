# Source inside a suitable Slurm allocation, in a compute-writable directory.
source /home6/u3129/mlip-research/envs/grace_ft_a100/activate-grace.sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/tensorflow:${LD_LIBRARY_PATH:-}"
export PATH="/home6/u3129/mlip-research/software/lammps-grace-b1f4a6c8cc4e/build:$PATH"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.cache}"
export PYTHONDONTWRITEBYTECODE=1
