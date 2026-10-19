#!/bin/bash
set -eo pipefail
umask 077
ulimit -c 0
cd "$FINETUNE_RUN_DIR"
mkdir cache dpa4 grace
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$PWD/cache" TORCHINDUCTOR_CACHE_DIR="$PWD/cache/inductor" TRITON_CACHE_DIR="$PWD/cache/triton"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=4 MAX_JOBS=4
export DP_TF32_INFER=0 DP_AMP_INFER=0 DP_TRITON_INFER=0 DP_COMPILE_INFER=0
hostname > node.txt
nvidia-smi > nvidia-smi.txt
source /home/u3129/miniconda3/etc/profile.d/conda.sh
BASELINE_LD_PATH="${LD_LIBRARY_PATH:-}"
conda activate lammps_deepmd_dpa4
export CUDA_HOME=/opt/cuda-13.0 CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$BASELINE_LD_PATH"
set +e
python al32-finetune-run.py dpa4 train-select > dpa4-console.txt 2>&1
dp_status=$?
set -e
source /home6/u3129/mlip-research/envs/grace_ft_a100/activate-grace.sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/tensorflow:$BASELINE_LD_PATH"
set +e
python al32-finetune-run.py grace train-select > grace-console.txt 2>&1
gr_status=$?
set -e
if [[ "$dp_status" == 0 && "$gr_status" == 0 ]]; then
  python al32-finetune-run.py all lock
  set +e
  python al32-finetune-run.py grace final > grace-final-console.txt 2>&1
  gr_status=$?
  set -e
  conda activate lammps_deepmd_dpa4
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$BASELINE_LD_PATH"
  set +e
  python al32-finetune-run.py dpa4 final > dpa4-final-console.txt 2>&1
  dp_status=$?
  set -e
fi
python - "$dp_status" "$gr_status" <<'PY'
import json,sys,os
from pathlib import Path
root=Path.cwd()
r={'status':'passed' if sys.argv[1:]==['0','0'] else 'failed','job_id':os.environ['SLURM_JOB_ID'],'cases':{}}
for name in ['dpa4','grace']:
 r['cases'][name]={}
 for stage in ['phase-result','final-metrics','selection']:
  p=root/name/(stage+'.json')
  if p.exists():r['cases'][name][stage]=json.loads(p.read_text())
Path('result.json').write_text(json.dumps(r,indent=2)+'\n')
PY
test "$dp_status" = 0 && test "$gr_status" = 0
