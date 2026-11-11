#!/bin/bash
set -eo pipefail
umask 077
ulimit -c 0
cd "$DEPLOY_RUN_DIR"
mkdir dpa4 grace cache
export XDG_CACHE_HOME="$PWD/cache"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/inductor" TRITON_CACHE_DIR="$PWD/cache/triton"
export TORCHINDUCTOR_COMPILE_THREADS=4 MAX_JOBS=4
export DP_TF32_INFER=0 DP_AMP_INFER=0 DP_TRITON_INFER=0 DP_COMPILE_INFER=0
hostname > node.txt
nvidia-smi > nvidia-smi.txt
source /home/u3129/miniconda3/etc/profile.d/conda.sh
conda activate lammps_deepmd_dpa4
export CUDA_HOME=/opt/cuda-13.0
export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
export PATH="$CUDA_HOME/bin:$PATH"
"$CXX" --version > cxx-version.txt
"$CUDA_HOME/bin/nvcc" --version > cuda-version.txt
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
set +e
dp_status=0
if [[ "$DEPLOY_BACKENDS" == *dpa4* ]]; then
  (cd dpa4 && python ../fm-deployment.py dpa4) > dpa4-console.txt 2>&1
  dp_status=$?
fi
set -e
source /home6/u3129/mlip-research/envs/grace_ft_a100/activate-grace.sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/tensorflow:${LD_LIBRARY_PATH:-}"
set +e
gr_status=0
if [[ "$DEPLOY_BACKENDS" == *grace* ]]; then
  (cd grace && python ../fm-deployment.py grace) > grace-console.txt 2>&1
  gr_status=$?
fi
set -e
python - "$dp_status" "$gr_status" <<'PY'
import json,sys,os
from pathlib import Path
r={'status':'passed' if sys.argv[1:]==['0','0'] else 'failed','cases':{}}
for name in ['dpa4','grace']:
 if name not in os.environ['DEPLOY_BACKENDS']: continue
 p=Path(name)/'result.json'
 r['cases'][name]=json.loads(p.read_text()) if p.exists() else {'status':'missing_result'}
Path('result.json').write_text(json.dumps(r,indent=2)+'\n')
PY
test "$dp_status" = 0 && test "$gr_status" = 0
