# Source this file from a shell or Slurm script before running VASP.
# Compute nodes use Lmod and do not expose the login node's Intel module name.
# Initialize the inspected, pinned toolchain directly from shared storage.
source /misc/umt_shared/opt/intel/oneapi/compiler/2025.3/env/vars.sh
source /misc/umt_shared/opt/intel/oneapi/mpi/2021.17/env/vars.sh
source /misc/umt_shared/opt/intel/oneapi/mkl/2025.3/env/vars.sh
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
export VASP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$VASP_ROOT/src/vasp.6.6.1/bin:$PATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -s unlimited
