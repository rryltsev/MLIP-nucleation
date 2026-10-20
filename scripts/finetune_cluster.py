"""Immutable bounded A100 DFT fine-tuning with no test labels in training payload."""
import argparse
import base64
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

from cluster_ssh import ROOT
from install_vasp import remote,write_remote
from vasp_potentials import save
from fm_cluster import collect
from al_pilot import backup


def sha(data):return hashlib.sha256(data).hexdigest()


def seconds(wall):
    h,m,s=map(int,wall.split(':'));return h*3600+m*60+s


def budget(cfg):
    prior=[json.loads(p.read_text()) for pattern in ('al32-*/manifest.json','fm-al32-baseline-*/manifest.json','fm-al32-finetune-*/manifest.json','fm-al32-lora-*/manifest.json') for p in (ROOT/'runs').glob(pattern) if json.loads(p.read_text()).get('job_id')]
    ids=[m['job_id'] for m in prior];assert len(set(ids))==len(ids)
    raw=remote('sacct -n -P -j '+','.join(ids)+' --format=JobIDRaw,State,CPUTimeRAW; squeue -h -u u3129 -o "%i|%j|%T"')
    accounting={f[0]:f for line in raw.splitlines() if len(f:=line.split('|'))>=3 and f[0] in ids and f[2].isdigit()}
    for line in raw.splitlines():
        f=line.split('|')
        if len(f)>=3 and f[0].isdigit() and f[1].startswith(('al32-','fm-al32-')):assert f[0] in ids,'Unrecorded related job'
    records=[];charge=0
    for m in prior:
        r=accounting[m['job_id']];state=r[1]
        if state in ('COMPLETED','FAILED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL') or state.startswith('CANCELLED'):cost=int(r[2]);basis='actual terminal'
        elif state in ('PENDING','RUNNING','COMPLETING','CONFIGURING'):
            resources=m['resources'];cost=seconds(resources['walltime'])*resources.get('ntasks',1)*resources.get('cpus_per_task',resources.get('cpus',1));basis='full reservation'
        else:raise ValueError('Unknown scheduler state: '+state)
        charge+=cost;records.append(dict(job_id=m['job_id'],state=state,CPU_seconds=cost,basis=basis))
    reserve=cfg['budget']['maximum_CPU_seconds'];assert charge+reserve<=128*3600
    return dict(prior_charged_CPU_seconds=charge,new_reserved_CPU_seconds=reserve,worst_case_CPU_hours=(charge+reserve)/3600,records=records,raw=raw)


def submit(run_id,lora=False):
    assert re.fullmatch(r'fm-al32-'+('lora' if lora else 'finetune')+r'-\d{8}-\d{2}',run_id)
    path=ROOT/'runs'/run_id/'manifest.json'
    if path.exists():raise FileExistsError('Recorded intent; reconcile before any retry')
    cfgpath=ROOT/('configs/training/al32-dpa4-lora-v1.json' if lora else 'configs/training/al32-foundation-finetune-v1.json');cfg=json.loads(cfgpath.read_text())
    snapshot=budget(cfg)
    pointer=json.loads((ROOT/cfg['dataset']).read_text());source=(ROOT/pointer['labels']).parent
    backup_record=json.loads((source/'artifact-backup.json').read_text())
    archive=ROOT/'artifacts/dft-run-backups'/Path(backup_record['remote_archive']).name
    assert sha(archive.read_bytes())==backup_record['sha256']
    export=json.loads((source/'export-manifest.json').read_text())
    payload={'study.json':cfgpath.read_bytes(),'dataset-manifest.json':(ROOT/pointer['geometry_manifest']).read_bytes(),
        'DFT-export-provenance.json':(source/'export-manifest.json').read_bytes()}
    assert sha(payload['dataset-manifest.json'])==cfg['geometry_manifest_sha256']
    with tarfile.open(archive) as tar:
        for split in ('train','validation'):
            name='export/'+split+'.extxyz';data=tar.extractfile(name).read();assert sha(data)==export['files'][name]
            payload['data/'+split+'.extxyz']=data
            for member in tar.getmembers():
                if member.isfile() and member.name.startswith('export/deepmd/'+split+'/'):
                    data=tar.extractfile(member).read();assert sha(data)==export['files'][member.name]
                    payload['data/'+member.name.removeprefix('export/')]=data
    sealed=dict(archive=backup_record['remote_archive'],archive_sha256=backup_record['sha256'],member='export/test.extxyz',member_sha256=export['files']['export/test.extxyz'],
        policy='Open only after both model selections are hashed and locked; test labels absent from training payload')
    if not lora:payload['sealed-test-source.json']=(json.dumps(sealed,indent=2)+'\n').encode()
    for backend,card in cfg['models'].items():payload[backend+'-checkpoint.json']=(ROOT/card).read_bytes()
    scripts=('al32-finetune-run.py','al32-lora-run.py','lora-child.py','al32-lora.sbatch') if lora else ('al32-finetune-run.py','fm-child.py','fm-finetune.py','al32-finetune.sbatch')
    for name in scripts:
        payload['run.sbatch' if name.endswith('.sbatch') else name]=(ROOT/'scripts/hpc'/name).read_bytes().replace(b'\r\n',b'\n')
    for name in ('baseline_metrics.py','finetune_selection.py'):payload[name]=(ROOT/'scripts'/name).read_bytes().replace(b'\r\n',b'\n')
    assert not any('/test/' in name or name.endswith('/test.extxyz') for name in payload)
    hashes={n:sha(b) for n,b in payload.items()}
    payload['run-inputs.json']=(json.dumps(hashes,indent=2)+'\n').encode()
    remote_dir='/home6/u3129/mlip-research/runs/'+run_id
    m=dict(run_id=run_id,stage='preparing',job_id=None,host='umt.imm.uran.ru',remote_run_dir=remote_dir,
        created_utc=datetime.now(timezone.utc).isoformat(),authorization=cfg['authorization'],resources=cfg['resources'],budget=cfg['budget'],budget_snapshot=snapshot,
        input_sha256={n:sha(b) for n,b in payload.items()},code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        test_policy='Train/validation only; no test labels or test archive access' if lora else 'No test labels in training/selection payload; final test opened only after both validation selections are locked')
    save(path,m)
    for n,b in payload.items():
        target=(ROOT/'artifacts/finetune-staging'/run_id/n) if n.endswith('.npy') else path.parent/n
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
    encoded=base64.b64encode(json.dumps({n:base64.b64encode(b).decode() for n,b in payload.items()}).encode()).decode()
    code='import base64,json,hashlib\nfrom pathlib import Path\nroot=Path('+repr(remote_dir)+');root.mkdir(mode=0o700)\n'
    code+='for n,v in json.loads(base64.b64decode('+repr(encoded)+')).items():\n p=root/n;p.parent.mkdir(parents=True,exist_ok=True);b=base64.b64decode(v)\n with p.open("xb") as f:f.write(b)\n assert hashlib.sha256(p.read_bytes()).hexdigest()==hashlib.sha256(b).hexdigest()\n'
    remote("umask 077; python3 - <<'PY'\n"+code+"\nPY\nbash -n "+shlex.quote(remote_dir+'/run.sbatch'),120)
    m['stage']='submission_intent';save(path,m);write_remote(remote_dir+'/submission-intent.json',path.read_bytes())
    q=shlex.quote(remote_dir);exports='ALL,FINETUNE_RUN_DIR='+remote_dir+',FINETUNE_CODE_COMMIT='+m['code_commit']
    reply=remote('set -euC; cd '+q+'; test ! -e job.id; sbatch --parsable --job-name='+run_id+
        ' --partition=v100 --account=imet-2026-2 --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --gres=gpu:a100:1 --time='+cfg['resources']['walltime']+' --chdir='+q+
        ' --output='+q+'/slurm-%j.out --export='+shlex.quote(exports)+' run.sbatch > job.id; cat job.id')
    job=reply.strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Uncertain outcome; reconcile job.id/scheduler before retry')
    m.update(stage='submitted',job_id=job);save(path,m);print(json.dumps(dict(run_id=run_id,job_id=job,worst_case_CPU_hours=snapshot['worst_case_CPU_hours'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    s.add_parser('submit').add_argument('run_id')
    s.add_parser('submit-lora').add_argument('run_id')
    for action in ('collect','backup'):s.add_parser(action).add_argument('manifest',type=Path)
    a=p.parse_args()
    if a.action=='submit':submit(a.run_id)
    elif a.action=='submit-lora':submit(a.run_id,lora=True)
    elif a.action=='collect':collect(a.manifest.resolve())
    else:backup(a.manifest.resolve())
