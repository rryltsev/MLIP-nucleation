#!/bin/bash
set +e
srun --partition=apollo --account=imet-2026-2 --qos=normal \
    --job-name=vasp661-bootstrap-log --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --nodelist=apollo20 --mem=512M --time=00:02:00 --chdir=/tmp \
    bash -c '
cat /tmp/u3129-vasp661-20950117.out 2>&1
printf "HOME=%s\n" "$HOME"
ls -ld /home/u3129/.local/opt/vasp/6.6.1-oneapi-2025/runs/vasp661-oneapi-20261017-03
printf writecheck > /home/u3129/.local/opt/vasp/6.6.1-oneapi-2025/runs/vasp661-oneapi-20261017-03/path-write-check.txt
printf "write_exit=%s\n" "$?"
' </dev/null 2>&1
printf 'diagnostic_exit_code=%s\n' "$?"
