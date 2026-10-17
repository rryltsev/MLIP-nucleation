#!/bin/bash
# Isolate stdout creation on the node where batch launch failed.
set +e
srun --partition=apollo --account=imet-2026-2 --qos=normal \
    --job-name=vasp661-stdio-diagnostic --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --nodelist=apollo20 --mem=512M --time=00:02:00 --chdir=/tmp \
    --output=/home/u3129/.local/opt/vasp/6.6.1-oneapi-2025/runs/vasp661-oneapi-20261017-02/stdio-%j.out \
    bash -c 'printf "job_id=%s\n" "$SLURM_JOB_ID"; hostname' </dev/null 2>&1
printf 'stdio_diagnostic_exit_code=%s\n' "$?"
