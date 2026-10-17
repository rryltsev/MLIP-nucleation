# Source this file from a shell or Slurm script before running VASP.
if ! type module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh
fi
module load intelmpi/intel64_2025
export VASP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$VASP_ROOT/src/vasp.6.6.1/bin:$PATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -s unlimited
