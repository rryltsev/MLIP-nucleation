"""Continue this bounded installation and A100 check; safe to resume recorded jobs.

Only the diagnosed install TIMEOUT permits one cached retry. No blind retry of
failed package resolution or training is allowed. Does not schedule recurrence.
"""
import json
from pathlib import Path
import time

from cluster_ssh import ROOT
from grace_cluster import ENV, submit, collect
from install_vasp import remote, write_remote
from lock_grace import main as lock


def wait_for_record(path, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = json.loads(path.read_text())
        if record['stage'] == 'collected':
            return record
        collect(path)
        record = json.loads(path.read_text())
        if record['stage'] == 'collected':
            return record
        time.sleep(40)
    raise TimeoutError('Bounded wait expired; recorded jobs remain recoverable: ' + str(path))


def main():
    path = ROOT/'runs/grace-install-20261017-02/manifest.json'
    record = wait_for_record(path, 1900)
    if record.get('runtime_status') != 'installed':
        if '|TIMEOUT|' not in record.get('scheduler_observation',''):
            raise RuntimeError('Installation failed for a non-timeout cause; diagnose before retrying')
        path = ROOT/'runs/grace-install-20261017-03/manifest.json'
        if not path.exists(): submit('install','03')
        record = wait_for_record(path, 3700)
        if record.get('runtime_status') != 'installed':
            raise RuntimeError('Cached installation did not pass; no further automatic retry')
    run_dir = record['remote_run_dir']
    remote('/home/u3129/miniconda3/bin/conda list --prefix ' + ENV +
           ' --explicit --sha256 > ' + run_dir + '/conda-explicit-sha256.txt')
    write_remote(ENV+'/activate-grace.sh',(ROOT/'scripts/hpc/grace-env.sh').read_bytes().replace(b'\r\n',b'\n'))
    collect(path)  # Also retrieve the newly created Conda checksum lock.
    lock(path.parent)
    print('INSTALLATION_OK; submitting/continuing the single A100 check',flush=True)
    path = ROOT/'runs/grace-smoke-20261017-01/manifest.json'
    if not path.exists(): submit('smoke','01')
    record = wait_for_record(path, 750)
    if record.get('runtime_status') != 'passed':
        raise RuntimeError('A100 check failed; diagnose the preserved result before any retry')
    print('GRACE_A100_CHECK_PASSED',flush=True)


if __name__ == '__main__': main()
