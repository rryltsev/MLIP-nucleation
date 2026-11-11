"""Prepare, stage, submit with an explicit budget, and collect one immutable benchmark."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

from cluster_ssh import ROOT, ssh_args
from install_vasp import remote, write_remote


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def prepare(run_id):
    folder = ROOT / 'runs' / run_id
    folder.mkdir(parents=True, exist_ok=False)
    package = folder / 'package'
    package.mkdir()
    (package / 'inputs').mkdir()
    tial = ROOT.parent / 'TiAl' / 'research' / 'tial_fm'
    precision = tial / 'runs' / 'grace-precision-20261107-02'
    old = load(precision / 'package' / 'plan.json')
    structures = ['Ti96', 'TiAl216', 'TiAl432', 'TiAl864']
    for name in structures:
        source = precision / 'package' / 'inputs' / 'timing' / (name + '.extxyz')
        (package / 'inputs' / source.name).write_bytes(source.read_bytes())
    mapping = {'worker.py': 'inference-benchmark-worker.py', 'runner.py': 'inference-benchmark-runner.py',
               'job.sbatch': 'inference-benchmark.sbatch'}
    for target, source in mapping.items():
        data = (ROOT / 'scripts' / 'hpc' / source).read_bytes().replace(b'\r\n', b'\n')
        if target.endswith('.py'):
            ast.parse(data)
        (package / target).write_bytes(data)
    mini = load(ROOT / 'models/foundation/dpa4-mini-omat24-20260805.json')
    neo = load(tial / 'runs/dpa4-n200-pilot-20261106-02/package/protocol.json')['model']
    cases = []
    for name, path, digest in [
        ('dpa4-mini', mini['remote_root'] + '/' + mini['filename'], mini['sha256']),
        ('dpa4-neo', neo['path'], neo['sha256'])]:
        for fmt, level in [('pt', 0), ('pt2', 0), ('pt2', 2)]:
            cases.append(dict(id=name + ('-pt' if fmt == 'pt' else f'-pt2-triton{level}'),
                              backend='dpa4', format=fmt, triton_level=level, path=path,
                              source_files={path: digest}, precision='checkpoint native float32'))
    one = load(ROOT / 'models/foundation/grace-1l-omat.json')
    files = {v['remote_path']: v['sha256'] for k, v in one['files'].items() if k.startswith('extracted/')}
    path = str(Path(next(k for k in files if k.endswith('model.yaml'))).parent).replace('\\', '/')
    cases.append(dict(id='grace-1l-fp64', backend='grace', precision='fp64', path=path, source_files=files))
    for precision_name in ['fp32', 'fp64']:
        card = old['checkpoints'][precision_name]
        hashes = card.get('file_hashes', {}).copy()
        if precision_name == 'fp32':
            hashes = load(precision / 'collected/controlled-cast.json')['derived_checkpoint_sha256']
            hashes['model.yaml'] = sha(precision / 'package/checkpoint-fp32/model.yaml')
        cases.append(dict(id='grace-2l-' + precision_name, backend='grace', precision=precision_name,
                          path=card['path'], source_files={card['path'] + '/' + k: v for k, v in hashes.items()},
                          provenance=card))
    plan = dict(run_id=run_id, scope='Inference throughput of fixed original foundation checkpoints; no training, DFT or held-out test access',
                structures=structures, geometry_source=str(precision / 'package/inputs/timing'),
                structures_scope='Existing GRACE timing geometries; TiAl cells from train, Ti96 diagnostic cell; same bytes for every model',
                warmup_calls=5, measured_calls=30, process_blocks=2, freeze_timeout_seconds=420,
                worker_timeout_seconds=120, cases=cases,
                environments=dict(dpa4='/home/u3129/miniconda3/envs/lammps_deepmd_dpa4',
                                  grace='/home6/u3129/mlip-research/envs/grace_ft_a100'),
                metric='Median full ASE E/F/stress latency, cache bypassed and GPU synchronized; cold load/export separate',
                export_tolerances=dict(energy_eV_per_atom=1e-5, force_max_eV_per_A=2e-4, stress_max_eV_per_A3=1e-6),
                tolerances_scope='Serialization consistency only, not DFT accuracy or physical MD validation',
                input_sha256={p.relative_to(package).as_posix(): sha(p) for p in package.rglob('*') if p.is_file()})
    save(package / 'plan.json', plan)
    manifest = dict(run_id=run_id, stage='prepared', job_id=None,
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    remote_run_dir='/home6/u3129/mlip-research/runs/' + run_id,
                    code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    resources=dict(partition='v100', account='imet-2026-2', qos='normal', nodes=1,
                                   tasks=1, cpus=4, gpus=1, memory_GiB=32, walltime='00:45:00'),
                    maximum_GPU_hours=.75, maximum_CPU_hours=3., user_authorized_budget=False,
                    input_sha256={p.relative_to(package).as_posix(): sha(p) for p in package.rglob('*') if p.is_file()})
    manifest['input_hash'] = hashlib.sha256(json.dumps(manifest['input_sha256'], sort_keys=True).encode()).hexdigest()
    save(folder / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2))


def stage(folder):
    m = load(folder / 'manifest.json')
    assert m['stage'] == 'prepared'
    root = m['remote_run_dir']
    # Only readonly model checks and writing a fresh, private input directory.
    remote('set -eu; umask 077; mkdir ' + shlex.quote(root))
    for name, digest in m['input_sha256'].items():
        path = folder / 'package' / name
        assert sha(path) == digest
        remote('mkdir -p ' + shlex.quote(root + '/' + str(Path(name).parent).replace('\\', '/')))
        write_remote(root + '/' + name, path.read_bytes())
    code = "import json,hashlib;from pathlib import Path;p=Path(" + repr(root) + ");m=json.loads((p/'plan.json').read_text());" \
           "checks={f:hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for c in m['cases'] for f,h in c['source_files'].items()};print(json.dumps(checks));assert all(checks.values())"
    verification = remote('python3 -c ' + shlex.quote(code))
    save(folder / 'remote-source-verification.json', json.loads(verification))
    m['stage'] = 'staged_awaiting_budget'
    save(folder / 'manifest.json', m)
    write_remote(root + '/manifest.json', (folder / 'manifest.json').read_bytes())
    print(m['stage'])


def submit(folder, authorized):
    m = load(folder / 'manifest.json')
    assert m['stage'] == 'staged_awaiting_budget' and authorized
    m.update(stage='submission_intent', user_authorized_budget=True,
             authorization=authorized, authorization_recorded_utc=datetime.now(timezone.utc).isoformat(),
             code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    save(folder / 'manifest.json', m)
    write_remote(m['remote_run_dir'] + '/manifest.json', (folder / 'manifest.json').read_bytes())
    root = shlex.quote(m['remote_run_dir'])
    command = ('set -euC; cd ' + root + '; test ! -e job.id; sbatch --parsable --account=imet-2026-2 '
               '--qos=normal --partition=v100 --gres=gpu:a100:1 --nodes=1 --ntasks=1 --cpus-per-task=4 '
               '--mem=32G --time=00:45:00 --job-name=' + shlex.quote(m['run_id']) +
               ' --output=slurm-%j.out job.sbatch > job.id; cat job.id')
    reply = remote(command).strip()
    job = reply.split(';')[0]
    assert job.isdigit(), 'Ambiguous submission; reconcile remote job.id, squeue and sacct before retry'
    m.update(stage='submitted', job_id=job)
    save(folder / 'manifest.json', m)
    write_remote(m['remote_run_dir'] + '/manifest.json', (folder / 'manifest.json').read_bytes())
    print(job)


def collect(folder):
    m = load(folder / 'manifest.json')
    assert m.get('job_id')
    status = remote('sacct -j ' + m['job_id'] + ' -P -n -o JobID,State,ExitCode,ElapsedRaw,CPUTimeRAW,NodeList')
    save(folder / 'scheduler.json', dict(observed_utc=datetime.now(timezone.utc).isoformat(), output=status))
    parent = next(row for row in status.splitlines() if row.startswith(m['job_id'] + '|'))
    assert parent.split('|')[1] in ['COMPLETED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'CANCELLED'], parent
    # Retrieve actual exported binaries and evidence, excluding caches (kept in node /tmp).
    remote_root = m['remote_run_dir']
    arc = folder / 'result.tar.gz'
    with arc.open('wb') as output:
        subprocess.run(ssh_args() + ['-o', 'BatchMode=yes', 'mlip-umt',
                       'tar -C ' + shlex.quote(remote_root) + ' -czf - .'], stdout=output, check=True)
    target = folder / 'collected'
    target.mkdir(exist_ok=False)
    with tarfile.open(arc) as archive:
        archive.extractall(target, filter='data')
    files = {p.relative_to(target).as_posix(): sha(p) for p in target.rglob('*') if p.is_file()}
    save(folder / 'collected-sha256.json', files)
    m.update(stage='collected', scheduler_parent=parent, result_archive_sha256=sha(arc))
    save(folder / 'manifest.json', m)
    print(m['stage'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'stage', 'submit', 'collect'])
    parser.add_argument('run_id')
    parser.add_argument('--authorization')
    a = parser.parse_args()
    assert Path(a.run_id).name == a.run_id
    folder = ROOT / 'runs' / a.run_id
    if a.command == 'prepare':
        prepare(a.run_id)
    elif a.command == 'stage':
        stage(folder)
    elif a.command == 'submit':
        submit(folder, a.authorization)
    else:
        collect(folder)
