# Source inside a Slurm allocation before GRACE training or inference.
source /home/u3129/miniconda3/etc/profile.d/conda.sh
conda activate /home6/u3129/mlip-research/envs/grace_ft_a100
export TF_USE_LEGACY_KERAS=1
export PYTHONNOUSERSITE=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
# This is a reconstructible download cache, not an archive of accepted models.
export GRACE_CACHE=/home6/u3129/mlip-research/model-cache/grace
