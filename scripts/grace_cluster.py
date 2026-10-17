"""Immutable Slurm installation/training checks for the isolated GRACE environment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from cluster_ssh import ROOT
from install_vasp import remote, write_remote
from vasp_potentials import save

ENV = "/home6/u3129/mlip-research/envs/grace_ft_a100"


def submit(phase, attempt):
    run_id = "grace-" + phase + "-20261017-" + attempt
    path = ROOT / "runs" / run_id / "manifest.json"
    if path.exists():
        raise FileExistsError("Run already recorded; reconcile job.id/scheduler before retrying")
    run_dir = "/home6/u3129/mlip-research/runs/" + run_id
    cpus, resume = 4, False
    if phase == "install":
        files = {"install-grace.sbatch": ROOT / "scripts/hpc/install-grace.sbatch",
                 "environment-proposal.yml": ROOT / "environments/grace-a100.proposed.yml"}
        batch, partition, walltime, gres = "install-grace.sbatch", "apollo", "00:30:00", None
        if attempt == '03':
            previous = json.loads((ROOT/'runs/grace-install-20261017-02/manifest.json').read_text())
            if previous.get('stage') != 'collected' or previous.get('runtime_status') == 'installed':
                raise RuntimeError('Reconcile the earlier installation before resuming')
            cpus, walltime, resume = 1, '01:00:00', True
    else:
        remote("test -r " + ENV + "/INSTALLATION.json")
        files = {"grace-smoke.sbatch": ROOT / "scripts/hpc/grace-smoke.sbatch",
                 "grace-smoke.py": ROOT / "scripts/hpc/grace-smoke.py",
                 "mlip-smoke.py": ROOT / "scripts/hpc/mlip-smoke.py"}
        batch, partition, walltime, gres = "grace-smoke.sbatch", "v100", "00:10:00", "gpu:a100:1"
        if attempt in ['02','03']:
            previous_attempt='%02d' % (int(attempt)-1)
            previous=json.loads((ROOT/('runs/grace-smoke-20261017-'+previous_attempt)/'manifest.json').read_text())
            if previous.get('stage') != 'collected' or previous.get('runtime_status') != 'failed':
                raise RuntimeError('Reconcile the earlier GPU check before retrying')
            walltime='00:05:00'
    payloads = {n:p.read_bytes().replace(b"\r\n",b"\n") for n,p in files.items()}
    hashes = {n:hashlib.sha256(b).hexdigest() for n,b in payloads.items()}
    manifest = {"run_id":run_id,"phase":phase,"host":"umt.imm.uran.ru","environment":ENV,
        "remote_run_dir":run_dir,"created_utc":datetime.now(timezone.utc).isoformat(),
        "authorization":"user requested isolated GRACE installation and a short A100 Slurm training check",
        "resources":{"partition":partition,"account":"imet-2026-2","qos":"normal","nodes":1,
                     "tasks":1,"cpus":cpus,"memory_gib":16,"walltime":walltime,"gres":gres},
        "resume_private_prefix":resume,
        "code_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "input_sha256":hashes,"input_hash":hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
        "stage":"preparing","job_id":None}
    save(path,manifest)
    q=shlex.quote(run_dir)
    parent=shlex.quote(run_dir.rsplit('/',1)[0])
    remote("set -eu; umask 077; mkdir -p " + parent + "; mkdir " + q)
    for n,b in payloads.items(): write_remote(run_dir+"/"+n,b)
    manifest['stage']='submission_intent'; save(path,manifest)
    write_remote(run_dir+"/manifest.json",path.read_bytes())
    flags=" --gres="+gres if gres else ""
    exports="ALL,GRACE_RUN_DIR="+run_dir+",GRACE_ENV="+ENV
    exports+=",GRACE_RESUME="+str(int(resume))
    answer=remote("set -euC; cd "+q+"; test ! -e job.id; sbatch --parsable --job-name="+run_id+
        " --partition="+partition+" --account=imet-2026-2 --qos=normal --nodes=1 --ntasks=1 --cpus-per-task="+str(cpus)+" --mem=16G --time="+
        walltime+flags+" --chdir="+q+" --output="+q+"/slurm-%j.out --export="+shlex.quote(exports)+" "+batch+" > job.id; cat job.id")
    job=answer.strip().split(';')[0]
    if not job.isdigit(): raise RuntimeError('Uncertain submission; reconcile before retrying')
    manifest.update(stage='submitted',job_id=job); save(path,manifest)
    write_remote(run_dir+"/manifest.json",path.read_bytes())
    print(json.dumps({'run_id':run_id,'job_id':job,'resources':manifest['resources']}))


def collect(path):
    manifest=json.loads(path.read_text())
    job=manifest['job_id']
    if not job or not job.isdigit(): raise ValueError('Reconcile remote job.id')
    state=remote("sacct -X -j "+job+" -nP --format=JobIDRaw,State,ExitCode,Elapsed,AllocCPUS,NodeList; squeue -h -j "+job+" -o '%i|%T|%R'")
    save(path.parent/'scheduler.json',{'observed_utc':datetime.now(timezone.utc).isoformat(),'output':state})
    if any('|'+s in state for s in ['RUNNING','PENDING','COMPLETING','CONFIGURING']):
        print(state); return
    if not any('|'+s in state for s in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED']):
        raise RuntimeError('Scheduler state unknown; reconcile before retry/collection')
    script="""python3 - <<'PY'
import json,hashlib
from pathlib import Path
root=Path(@ROOT@)
patterns=['*.json','*.txt','*.csv','*.log','*.out','*.yaml','data/provenance.json','data/*.extxyz','train/*.yaml','train/*.txt','train/*.log','train/seed/*/*.yaml','train/seed/*/*.txt']
records={}
for pattern in patterns:
 for p in root.glob(pattern):
  if p.name in ['manifest.json','job.id'] or not p.is_file(): continue
  b=p.read_bytes()
  if len(b)>2500000: continue
  records[p.relative_to(root).as_posix()]={'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode(errors='replace')}
print(json.dumps(records))
PY
""".replace('@ROOT@',repr(manifest['remote_run_dir']))
    records=json.loads(remote(script))
    for n,r in records.items():
        b=r['text'].encode()
        if hashlib.sha256(b).hexdigest()!=r['sha256']: raise ValueError('Hash mismatch: '+n)
        p=path.parent/n; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b)
    save(path.parent/'collected-sha256.json',{n:r['sha256'] for n,r in records.items()})
    result=json.loads(records['result.json']['text']) if 'result.json' in records else {'status':'missing_result'}
    manifest.update(stage='collected',runtime_status=result['status'],scheduler_observation=state)
    save(path,manifest)
    print(json.dumps({'run_id':manifest['run_id'],'scheduler':state,'runtime_status':result['status']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('submit'); p.add_argument('phase',choices=['install','smoke']); p.add_argument('--attempt',choices=['01','02','03'],default='01')
    p=sub.add_parser('collect'); p.add_argument('manifest',type=Path)
    args=parser.parse_args()
    submit(args.phase,args.attempt) if args.command=='submit' else collect(args.manifest)
