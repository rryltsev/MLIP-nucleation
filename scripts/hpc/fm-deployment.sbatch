#!/bin/bash
set -eo pipefail
umask 077
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
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
set +e
(cd dpa4 && python ../fm-deployment.py dpa4) > dpa4-console.txt 2>&1
dp_status=$?
set -e
source /home6/u3129/mlip-research/envs/grace_ft_a100/activate-grace.sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/tensorflow:${LD_LIBRARY_PATH:-}"
set +e
(cd grace && python ../fm-deployment.py grace) > grace-console.txt 2>&1
gr_status=$?
set -e
python - "$dp_status" "$gr_status" <<'PY'
import json,sys
from pathlib import Path
r={'status':'passed' if sys.argv[1:]==['0','0'] else 'failed','cases':{}}
for name in ['dpa4','grace']:
 p=Path(name)/'result.json'
 r['cases'][name]=json.loads(p.read_text()) if p.exists() else {'status':'missing_result'}
Path('result.json').write_text(json.dumps(r,indent=2)+'\n')
PY
test "$dp_status" = 0 && test "$gr_status" = 0
