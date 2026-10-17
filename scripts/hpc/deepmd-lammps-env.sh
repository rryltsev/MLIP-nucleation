# Source inside a suitable Slurm allocation before native DPA4 LAMMPS inference.
source /home/u3129/miniconda3/etc/profile.d/conda.sh
conda activate lammps_deepmd_dpa4
source /misc/umt_shared/opt/intel/oneapi/mpi/2021.17/env/vars.sh
export DEEPMD_SITE="$CONDA_PREFIX/lib/python3.12/site-packages"
export DEEPMD_LMP_PLUGIN="$DEEPMD_SITE/deepmd/lib/libdeepmd_lmpplugin.so"
export DEEPMD_LMP_ROOT=/home6/u3129/mlip-research/software/lammps-deepmd-mpi-a33449868448
# The DeepMD wheel's native CUDA ops require cudart12; Torch uses its own cu13.
# Preserve both real libraries. Do not alias one CUDA major version to another.
export LD_LIBRARY_PATH="$I_MPI_ROOT/lib:$CONDA_PREFIX/lib:$DEEPMD_SITE/torch/lib:$DEEPMD_SITE/nvidia/cu13/lib:/opt/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export PATH="$DEEPMD_LMP_ROOT/build:$PATH"
export I_MPI_FABRICS=shm
export DP_BACKEND_PLUGIN_PATH="$DEEPMD_SITE/deepmd/lib"
export DP_INTRA_OP_PARALLELISM_THREADS=4 DP_INTER_OP_PARALLELISM_THREADS=1
export DP_TF32_INFER=0 DP_AMP_INFER=0 DP_TRITON_INFER=0
export PYTHONNOUSERSITE=1
