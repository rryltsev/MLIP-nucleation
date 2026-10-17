# Source inside a Slurm allocation before GRACE training or inference.
source /home/u3129/miniconda3/etc/profile.d/conda.sh
conda activate /home6/u3129/mlip-research/envs/grace_ft_a100
export TF_USE_LEGACY_KERAS=1
export PYTHONNOUSERSITE=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export GRACE_CUDA_ROOT="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvcc"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$GRACE_CUDA_ROOT"
export PATH="$GRACE_CUDA_ROOT/bin:$PATH"
# This is a reconstructible download cache, not an archive of accepted models.
export GRACE_CACHE=/home6/u3129/mlip-research/model-cache/grace
