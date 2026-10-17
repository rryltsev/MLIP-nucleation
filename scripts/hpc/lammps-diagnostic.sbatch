#!/bin/bash
set -eo pipefail
umask 077
ulimit -c 0
cd "$DEPLOY_RUN_DIR"
source /home/u3129/miniconda3/etc/profile.d/conda.sh
conda activate lammps_deepmd_dpa4
export LAMMPS_PLUGIN_PATH=""
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
binary="$CONDA_PREFIX/lib/python3.12/site-packages/lammps/lmp"
ldd "$binary" > ldd.txt
set +e
timeout 90 gdb -batch -ex run -ex 'thread apply all bt' --args "$binary" -h > backtrace.txt 2>&1
set -e
find "$CONDA_PREFIX/lib" -maxdepth 2 -name 'libmpi*' -printf '%p -> %l\n' > mpi-links.txt
source /misc/umt_shared/opt/intel/oneapi/mpi/2021.17/env/vars.sh
export LD_LIBRARY_PATH="$I_MPI_ROOT/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export I_MPI_FABRICS=shm
ldd "$binary" > ldd-intel-mpi.txt
set +e
timeout 60 "$binary" -h > help-intel-mpi.txt 2>&1
mpi_status=$?
set -e
export MPI_HELP_STATUS="$mpi_status"
python - <<'PY'
from pathlib import Path
import json,os
Path('result.json').write_text(json.dumps({'status':'passed','scope':'diagnostic execution only, not model validation','job_id':os.environ['SLURM_JOB_ID'],'backtrace':'backtrace.txt','intel_mpi_help_exitcode':int(os.environ['MPI_HELP_STATUS'])},indent=2)+'\n')
PY
