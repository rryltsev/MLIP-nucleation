#!/bin/bash
set +e
srun --partition=apollo --account=imet-2026-2 --qos=normal \
    --job-name=vasp661-storage-diagnostic --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --mem=512M --time=00:02:00 --chdir=/tmp bash -c '
printf "job_id=%s\nnode=%s\n" "$SLURM_JOB_ID" "$(hostname)"
ls -ld /home6/u3129
umask 077
mkdir /home6/u3129/.mlip-vasp-storage-check-20261017
printf writecheck > /home6/u3129/.mlip-vasp-storage-check-20261017/result.txt
printf "write_exit=%s\n" "$?"
df -h /home6/u3129
' </dev/null 2>&1
printf 'diagnostic_exit_code=%s\n' "$?"
