"""Persistent, bounded A100 foundation fine-tuning submission and collection."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from cluster_ssh import ROOT, ssh_args
from install_vasp import remote, write_remote
from vasp_potentials import save


def submit(attempt, source=None):
    run_id = ("fm-verify-20261017-" if source else "fm-finetune-20261017-") + attempt
    path = ROOT / "runs" / run_id / "manifest.json"
    if path.exists(): raise FileExistsError("Recorded run; reconcile before retry")
    run_dir = "/home6/u3129/mlip-research/runs/" + run_id
    files = {n:ROOT/"scripts/hpc"/n for n in ["fm-finetune.sbatch","fm-finetune.py","fm-child.py","mlip-smoke.py","grace-smoke.py"]}
    files.update({"dpa4-checkpoint.json":ROOT/"models/foundation/dpa4-mini-omat24-20260805.json",
                  "grace-checkpoint.json":ROOT/"models/foundation/grace-1l-omat.json"})
    if source:
        original = json.loads(source.read_text())
        if original['stage'] != 'collected': raise ValueError('Reconcile source run first')
        files = {n:ROOT/'scripts/hpc'/n for n in ['fm-verify.sbatch','fm-verify.py','fm-finetune.py']}
    payload = {n:p.read_bytes().replace(b"\r\n",b"\n") for n,p in files.items()}
    hashes = {n:hashlib.sha256(b).hexdigest() for n,b in payload.items()}
    manifest = {"run_id":run_id,"remote_run_dir":run_dir,"host":"umt.imm.uran.ru",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "authorization":"user requested download and short actual DPA4 and GRACE fine-tuning on A100; synthetic software diagnostic",
        "budget":{"gpu_count":1,"maximum_gpu_seconds":900,"child_training_timeout_seconds":360,"steps_each":20},
        "resources":{"partition":"v100","account":"imet-2026-2","qos":"normal","nodes":1,"tasks":1,"cpus":4,"memory_gib":32,"gres":"gpu:a100:1","walltime":"00:15:00"},
        "input_sha256":hashes,"input_hash":hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
        "code_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "stage":"preparing","job_id":None}
    if source:
        manifest.update(source_manifest=source.relative_to(ROOT).as_posix(),source_job_id=original['job_id'],
            source_run=original['remote_run_dir'],source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            authorization='read-only verification of existing user-requested fine-tuning checkpoints; no training',
            budget={'gpu_count':0,'maximum_cpu_seconds':300},
            resources={'partition':'apollo','account':'imet-2026-2','qos':'normal','nodes':1,'tasks':1,'cpus':1,'memory_gib':8,'walltime':'00:05:00'})
    save(path,manifest)
    remote("set -eu; umask 077; mkdir -p " + shlex.quote(run_dir.rsplit('/',1)[0]) + "; mkdir " + shlex.quote(run_dir))
    for n,b in payload.items(): write_remote(run_dir+"/"+n,b)
    manifest["stage"]="submission_intent"; save(path,manifest)
    write_remote(run_dir+"/manifest.json",path.read_bytes())
    q=shlex.quote(run_dir)
    resources=manifest['resources']; exports="ALL,FM_RUN_DIR="+run_dir
    if source: exports+=",FM_SOURCE_RUN="+original['remote_run_dir']
    flags=' --gres=gpu:a100:1' if not source else ''
    batch='fm-verify.sbatch' if source else 'fm-finetune.sbatch'
    answer=remote("set -euC; cd "+q+"; test ! -e job.id; sbatch --parsable --job-name="+run_id+
        " --partition="+resources['partition']+" --account=imet-2026-2 --qos=normal --nodes=1 --ntasks=1 --cpus-per-task="+str(resources['cpus'])+
        " --mem="+str(resources['memory_gib'])+"G"+flags+" --time="+resources['walltime']+" --chdir="+
        q+" --output="+q+"/slurm-%j.out --export="+shlex.quote(exports)+" "+batch+" > job.id; cat job.id")
    job=answer.strip().split(';')[0]
    if not job.isdigit(): raise RuntimeError("Uncertain submission; reconcile job.id and scheduler before retrying")
    manifest.update(stage="submitted",job_id=job); save(path,manifest)
    write_remote(run_dir+"/manifest.json",path.read_bytes())
    print(json.dumps({"run_id":run_id,"job_id":job}))


def collect(path):
    manifest=json.loads(path.read_text()); job=manifest["job_id"]
    if not job or not job.isdigit(): raise ValueError("Reconcile submission first")
    status=remote("sacct -X -j "+job+" -nP --format=JobIDRaw,State,ExitCode,Elapsed,ElapsedRaw,AllocCPUS,NodeList; squeue -h -j "+job+" -o '%i|%T|%R'")
    save(path.parent/"scheduler.json",{"observed_utc":datetime.now(timezone.utc).isoformat(),"output":status})
    if any('|'+s in status for s in ['RUNNING','PENDING','COMPLETING','CONFIGURING']): print(status); return
    if not any('|'+s in status for s in ['COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED']):
        raise RuntimeError("Unknown scheduler state")
    script="""python3 - <<'PY'
from pathlib import Path
import hashlib,json
root=Path(@ROOT@); records={}
for p in root.rglob('*'):
 rel=p.relative_to(root)
 if not p.is_file() or 'cache' in rel.parts or 'loaded-initial' in rel.parts or p.name in ['manifest.json','job.id']: continue
 if p.suffix not in ['.json','.yaml','.txt','.log','.out','.csv','.extxyz','.data','.lammps','.dump']: continue
 b=p.read_bytes()
 if len(b)>2500000: continue
 records[rel.as_posix()]={'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode()}
print(json.dumps(records))
PY
""".replace('@ROOT@',repr(manifest["remote_run_dir"]))
    records=json.loads(remote(script))
    for name,record in records.items():
        b=record["text"].encode(); assert hashlib.sha256(b).hexdigest()==record["sha256"]
        target=path.parent/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(b)
    save(path.parent/"collected-sha256.json",{n:r['sha256'] for n,r in records.items()})
    result=json.loads(records['result.json']['text']) if 'result.json' in records else {'status':'missing_result'}
    manifest.update(stage='collected',runtime_status=result['status'],scheduler_observation=status); save(path,manifest)
    print(json.dumps({'status':result['status'],'scheduler':status,'cases':{n:r.get('status',r.get('final-metrics',{}).get('status','missing_result')) for n,r in result.get('cases',{}).items()}}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('submit'); s.add_argument('--attempt',required=True)
    v=sub.add_parser('verify'); v.add_argument('manifest',type=Path); v.add_argument('--attempt',required=True)
    c=sub.add_parser('collect'); c.add_argument('manifest',type=Path)
    a=p.parse_args()
    if a.command=='submit': submit(a.attempt)
    elif a.command=='verify': submit(a.attempt,a.manifest.resolve())
    else: collect(a.manifest)
