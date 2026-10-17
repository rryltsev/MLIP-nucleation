#!/bin/bash
#SBATCH --partition=apollo
#SBATCH --account=imet-2026-2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -eo pipefail
umask 077
INSTALL_ROOT="${VASP_BUILD_ROOT:-/home6/u3129/.local/opt/vasp/6.6.1-oneapi-2025}"
RUN_DIR="${VASP_BUILD_RUN_DIR:-$SLURM_SUBMIT_DIR}"
SOURCE_DIR="$INSTALL_ROOT/src/vasp.6.6.1"
exec > "$RUN_DIR/runtime.log" 2>&1
trap 'rc=$?; printf "exit_code=%s\n" "$rc" >> "$RUN_DIR/status.txt"; exit "$rc"' EXIT
printf 'phase=building\njob_id=%s\nnode=%s\n' "$SLURM_JOB_ID" "$(hostname)" > "$RUN_DIR/status.txt"
source "$INSTALL_ROOT/env.sh"
ifx --version > "$RUN_DIR/compiler.txt"
mpirun --version > "$RUN_DIR/mpi.txt"
cd "$SOURCE_DIR"
for target in std gam ncl; do
    printf 'target=%s\n' "$target" >> "$RUN_DIR/status.txt"
    make DEPS=1 -j "$SLURM_NTASKS" "$target" > "$RUN_DIR/build-$target.log" 2>&1
    test -x "bin/vasp_$target"
done
sha256sum bin/vasp_std bin/vasp_gam bin/vasp_ncl > "$RUN_DIR/binaries.sha256"
for target in std gam ncl; do
    ldd "bin/vasp_$target" > "$RUN_DIR/ldd-$target.txt"
    if grep -q 'not found' "$RUN_DIR/ldd-$target.txt"; then
        printf 'Missing shared library for %s\n' "$target" >&2
        exit 1
    fi
done
printf 'phase=validating\n' >> "$RUN_DIR/status.txt"
export I_MPI_FABRICS=shm
export VASP_TESTSUITE_EXE_STD="mpirun -np 4 $SOURCE_DIR/bin/vasp_std"
export VASP_TESTSUITE_EXE_GAM="mpirun -np 4 $SOURCE_DIR/bin/vasp_gam"
export VASP_TESTSUITE_EXE_NCL="mpirun -np 4 $SOURCE_DIR/bin/vasp_ncl"
export VASP_TESTSUITE_TESTS="bulk_AlPAW H2Ostretch_efor Fe_r2SCAN_ncl_soc bulk_BN_vdW-DF3-opt1"
# --all avoids FAST filtering; the explicit list above still limits this to four cases.
make test_all > "$RUN_DIR/validation.log" 2>&1
cp testsuite/testsuite.log "$RUN_DIR/testsuite.log"
printf 'phase=complete\n' >> "$RUN_DIR/status.txt"
