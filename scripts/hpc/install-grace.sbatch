#!/bin/bash
set -eo pipefail
umask 077
cd "$GRACE_RUN_DIR"
export PYTHONDONTWRITEBYTECODE=1
export CONDA_PKGS_DIRS=/home6/u3129/mlip-research/package-cache/conda
export PIP_CACHE_DIR=/home6/u3129/mlip-research/package-cache/pip
export CONDA_ENVS_PATH=/home6/u3129/mlip-research/envs
export CONDA_NUMBER_CHANNEL_NOTICES=0
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"
test ! -e "$GRACE_ENV"
hostname > node.txt
/home/u3129/miniconda3/bin/conda create --yes --no-default-packages --override-channels \
    --channel conda-forge --prefix "$GRACE_ENV" python=3.11 pip > conda-create.log 2>&1
"$GRACE_ENV/bin/python" -m pip install --only-binary=:all: --report pip-install-report.json \
    'tensorpotential==0.6.1' 'tensorflow[and-cuda]==2.20.0' 'tf-keras~=2.20.0' 'dpdata==1.1.0' \
    > pip-install.log 2>&1
"$GRACE_ENV/bin/python" -m pip check > pip-check.txt 2>&1
"$GRACE_ENV/bin/python" -m pip freeze --all > pip-freeze.txt
/home/u3129/miniconda3/bin/conda list --prefix "$GRACE_ENV" --explicit > conda-explicit.txt
"$GRACE_ENV/bin/python" - <<'PY'
import importlib.metadata as md
import json, os, sys
from pathlib import Path
versions={p:md.version(p) for p in ['tensorpotential','tensorflow','tf-keras','dpdata','numpy','ase']}
assert versions['tensorpotential']=='0.6.1' and versions['tensorflow']=='2.20.0'
assert versions['tf-keras'].startswith('2.20.') and sys.version_info[:2]==(3,11)
result={'status':'installed','environment':os.environ['GRACE_ENV'],'python':sys.version,
        'versions':versions,'pip_check':'passed','job_id':os.environ['SLURM_JOB_ID']}
Path('result.json').write_text(json.dumps(result,indent=2)+'\n')
Path(os.environ['GRACE_ENV'],'INSTALLATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
PY
