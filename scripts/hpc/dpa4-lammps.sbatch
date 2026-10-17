#!/bin/bash
set -eo pipefail
umask 077
ulimit -c 0
cd "$DEPLOY_RUN_DIR"
mkdir dpa4
source /home/u3129/miniconda3/etc/profile.d/conda.sh
conda activate lammps_deepmd_dpa4
source /misc/umt_shared/opt/intel/oneapi/mpi/2021.17/env/vars.sh
export LD_LIBRARY_PATH="$I_MPI_ROOT/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export I_MPI_FABRICS=shm
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
