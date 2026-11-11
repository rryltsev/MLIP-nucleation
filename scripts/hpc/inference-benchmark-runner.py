"""Bounded exports and paired fresh-process timing; all attempts are retained."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path.cwd()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def run(command, logfile, env, timeout):
    begin = time.monotonic()
    with logfile.open('w') as log:
        try:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, timeout=timeout)
            code = process.returncode
        except subprocess.TimeoutExpired:
            code = -999
    return dict(command=command, returncode=code, seconds=time.monotonic() - begin)


def environment(case, plan):
    env = os.environ.copy()
    if case['backend'] == 'grace':
        prefix = plan['environments']['grace']
        cuda = prefix + '/lib/python3.11/site-packages/nvidia/cuda_nvcc'
        env.update(TF_USE_LEGACY_KERAS='1', TF_FORCE_GPU_ALLOW_GROWTH='true',
                   XLA_FLAGS='--xla_gpu_cuda_data_dir=' + cuda,
                   PATH=cuda + '/bin:' + env['PATH'])
    return env


def main():
    assert os.environ.get('SLURM_JOB_ID')
    plan = json.loads((ROOT / 'plan.json').read_text())
    for name, digest in plan['input_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    exports = ROOT / 'exports'
    exports.mkdir()
    (ROOT / 'results').mkdir()
    save(ROOT / 'hardware.json', {name: subprocess.check_output(command, text=True) for name, command in {
        'cpu': ['lscpu'], 'gpu': ['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv'],
        'allocation': ['scontrol', 'show', 'job', os.environ['SLURM_JOB_ID']]}.items()})
    available, attempts = [], []
    for case in plan['cases']:
        for name, digest in case['source_files'].items():
            with open(name, 'rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest, name
        if case.get('format') == 'pt2':
            target = exports / (case['id'] + '.pt2')
            env = environment(case, plan)
            env['DP_TRITON_INFER'] = str(case['triton_level'])
            record = run([plan['environments']['dpa4'] + '/bin/dp', '--pt', 'freeze', '-c', case['path'],
                          '-o', str(target)], exports / (case['id'] + '.log'), env, plan['freeze_timeout_seconds'])
            record['freeze_environment'] = {k: env.get(k) for k in ['DP_TRITON_INFER', 'DP_COMPILE_INFER', 'DP_TF32_INFER', 'DP_AMP_INFER']}
            record['source_files'] = case['source_files']
            if record['returncode'] == 0 and target.exists():
                record['sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
                available.append(case)
            else:
                record['returncode'] = record['returncode'] or -998
            save(exports / (case['id'] + '.json'), record)
        else:
            available.append(case)
    # Reverse the complete order for block1 to reduce ordering confounding.
    for block in [0, 1]:
        for case in available if block == 0 else list(reversed(available)):
            command = [plan['environments'][case['backend']] + '/bin/python',
                       str(ROOT / 'worker.py'), case['id'], str(block)]
            record = run(command, ROOT / 'results' / f"{case['id']}-block{block}.log",
                         environment(case, plan), plan['worker_timeout_seconds'])
            record.update(case=case['id'], block=block)
            attempts.append(record)
            save(ROOT / 'processes.json', attempts)
            print(json.dumps(record), flush=True)
    save(ROOT / 'completion.json', dict(finished=True, attempted_cases=len(available),
         all_cases_executed=len(available) == len(plan['cases']),
         processes_passed=all(r['returncode'] == 0 for r in attempts)))


if __name__ == '__main__':
    main()
