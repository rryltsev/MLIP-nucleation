# Project settings appended to the licensed arch/makefile.include.oneapi template.
# Portable across the inspected CPU partitions; no node-specific -xHOST tuning.
VASP_TARGET_CPU = -xSSE4.2
FC = mpiifx
FCL = mpiifx -qmkl=sequential
# ASU's documented ifx workaround for VASP 6.6.x nonlocal vdW kernel generation.
# https://docs.rc.asu.edu/vasp/#known-issue-vdw-df-results-on-oneapi-vasp-66x-fixed-2026-10-15
vdw_nl.o: FFLAGS += -fp-model precise
