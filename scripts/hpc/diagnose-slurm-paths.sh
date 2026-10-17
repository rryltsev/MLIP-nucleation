#!/bin/bash
# One CPU, 512 MiB, maximum two minutes; no compilation or VASP execution.
set +e
srun --partition=apollo --account=imet-2026-2 --qos=normal \
    --job-name=vasp661-launch-diagnostic --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --mem=512M --time=00:02:00 --chdir=/tmp bash -c '
printf "job_id=%s\nnode=%s\n" "$SLURM_JOB_ID" "$(hostname)"
for path in /home/u3129 /home4/u3129 /misc/home4/u3129 /home/u3129/.local/opt/vasp/6.6.1-oneapi-2025; do
    ls -ld "$path" 2>&1
done
test -r /home/u3129/.local/opt/vasp/6.6.1-oneapi-2025/runs/vasp661-oneapi-20261017-01/build.sbatch
printf "build_script_readable_exit=%s\n" "$?"
' </dev/null
printf 'diagnostic_exit_code=%s\n' "$?"
