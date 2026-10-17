#!/bin/bash
set -eo pipefail
umask 077
ulimit -c 0
cd "$DEPLOY_RUN_DIR"
mkdir dpa4
source ./deepmd-lammps-env.sh
export OMP_NUM_THREADS=4 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export DP_TF32_INFER=0 DP_AMP_INFER=0 DP_TRITON_INFER=0
ldd /home6/u3129/mlip-research/software/lammps-deepmd-mpi-a33449868448/build/lmp > ldd.txt
set +e
(cd dpa4 && python ../dpa4-lammps.py dpa4) > dpa4-console.txt 2>&1
status=$?
set -e
python - <<'PY'
from pathlib import Path
import json
r=json.loads(Path('dpa4/result.json').read_text())
Path('result.json').write_text(json.dumps({'status':r['status'],'cases':{'dpa4':r}},indent=2)+'\n')
PY
test "$status" = 0
