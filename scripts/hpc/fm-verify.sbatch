#!/bin/bash
set -eo pipefail
umask 077
cd "$FM_RUN_DIR"
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_USE_LEGACY_KERAS=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
set +e
/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python fm-verify.py dpa4 > dpa4-console.txt 2>&1
dp_status=$?
/home6/u3129/mlip-research/envs/grace_ft_a100/bin/python fm-verify.py grace > grace-console.txt 2>&1
gr_status=$?
set -e
python3 - "$dp_status" "$gr_status" <<'PY'
import json,sys
from pathlib import Path
r={'status':'passed' if sys.argv[1:]==['0','0'] else 'failed', 'cases':{}}
for name in ['dpa4','grace']:
 p=Path(name+'-verification.json')
 r['cases'][name]=json.loads(p.read_text()) if p.exists() else {'status':'missing_result'}
Path('result.json').write_text(json.dumps(r,indent=2)+'\n')
PY
test "$dp_status" = 0 && test "$gr_status" = 0
