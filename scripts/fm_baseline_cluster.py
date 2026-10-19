"""Stage and submit an immutable, read-only A100 original-FM Al32 evaluation."""
import argparse
import base64
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

from cluster_ssh import ROOT
from install_vasp import remote,write_remote
from vasp_potentials import save
from fm_cluster import collect
from al_pilot import backup


def submit(run_id):
    assert re.fullmatch(r'fm-al32-baseline-\d{8}-\d{2}',run_id)
    path=ROOT/'runs'/run_id/'manifest.json'
    if path.exists():raise FileExistsError('Reconcile recorded intent; no blind retry')
    cfgpath=ROOT/'configs/evaluation/al32-foundation-baseline-v1.json';cfg=json.loads(cfgpath.read_text())
    dataset=json.loads((ROOT/cfg['dataset']).read_text())
    geometry=ROOT/dataset['geometry_manifest'];labels=ROOT/dataset['labels']
    assert hashlib.sha256(geometry.read_bytes()).hexdigest()==cfg['geometry_manifest_sha256']
    assert hashlib.sha256(labels.read_bytes()).hexdigest()==cfg['labels_sha256']
    assert dataset['split_counts']=={'train':21,'validation':6,'test':13}
    payload={'evaluation.json':cfgpath.read_bytes(),'dataset-manifest.json':geometry.read_bytes(),'dataset-pointer.json':(ROOT/cfg['dataset']).read_bytes(),
        'labels.json':labels.read_bytes(),'fm-al32-baseline.py':(ROOT/'scripts/hpc/fm-al32-baseline.py').read_bytes(),
        'run.sbatch':(ROOT/'scripts/hpc/fm-al32-baseline.sbatch').read_bytes()}
    for backend,card in cfg['models'].items():payload[backend+'-checkpoint.json']=(ROOT/card).read_bytes()
    for row in json.loads(geometry.read_text())['configurations']:
        payload[row['poscar']]=(geometry.parent/row['poscar']).read_bytes()
    # Preserve exact archived geometry/labels bytes, normalize only executable code.
    for n in ('fm-al32-baseline.py','run.sbatch'):payload[n]=payload[n].replace(b'\r\n',b'\n')
    hashes={n:hashlib.sha256(v).hexdigest() for n,v in payload.items()}
    payload['run-inputs.json']=(json.dumps(hashes,indent=2)+'\n').encode()
    run_dir='/home6/u3129/mlip-research/runs/'+run_id
    m=dict(run_id=run_id,remote_run_dir=run_dir,host='umt.imm.uran.ru',job_id=None,stage='preparing',
        created_utc=datetime.now(timezone.utc).isoformat(),authorization=cfg['authorization'],resources=cfg['resources'],budget=cfg['budget'],
        input_sha256={n:hashlib.sha256(v).hexdigest() for n,v in payload.items()},
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    save(path,m)
    for n,v in payload.items():
        p=path.parent/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(v)
    encoded=base64.b64encode(json.dumps({n:base64.b64encode(v).decode() for n,v in payload.items()}).encode()).decode()
    code="import base64,json,hashlib\nfrom pathlib import Path\nroot=Path("+repr(run_dir)+");root.mkdir(mode=0o700)\n"
    code+="for n,v in json.loads(base64.b64decode("+repr(encoded)+")).items():\n p=root/n;p.parent.mkdir(parents=True,exist_ok=True);b=base64.b64decode(v)\n with p.open('xb') as f:f.write(b)\n assert hashlib.sha256(p.read_bytes()).hexdigest()==hashlib.sha256(b).hexdigest()\n"
    remote("umask 077; python3 - <<'PY'\n"+code+"\nPY\nbash -n "+shlex.quote(run_dir+'/run.sbatch'),120)
    m['stage']='submission_intent';save(path,m);write_remote(run_dir+'/submission-intent.json',path.read_bytes())
    q=shlex.quote(run_dir);r=cfg['resources']
    exports='ALL,BASELINE_RUN_DIR='+run_dir+',BASELINE_CODE_COMMIT='+m['code_commit']
    reply=remote('set -euC; cd '+q+'; test ! -e job.id; sbatch --parsable --job-name='+run_id+
        ' --partition='+r['partition']+' --account='+r['account']+' --qos='+r['qos']+' --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time='+r['walltime']+
        ' --gres=gpu:a100:1 --chdir='+q+' --output='+q+'/slurm-%j.out --export='+shlex.quote(exports)+' run.sbatch > job.id; cat job.id')
    job=reply.strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Uncertain submission; reconcile job.id and scheduler before retry')
    m.update(stage='submitted',job_id=job);save(path,m);print(json.dumps(dict(run_id=run_id,job_id=job)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    s.add_parser('submit').add_argument('run_id')
    for name in ('collect','backup'):s.add_parser(name).add_argument('manifest',type=Path)
    a=p.parse_args()
    if a.action=='submit':submit(a.run_id)
    elif a.action=='collect':collect(a.manifest.resolve())
    else:backup(a.manifest.resolve())
