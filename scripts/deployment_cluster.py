"""Submit immutable restart/export checks and their prerequisite LAMMPS build."""
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
from fm_cluster import collect


def submit(phase, attempt, backends='dpa4,grace'):
    run_id=('grace-lammps-build-' if phase=='build' else 'fm-deployment-')+'20261017-'+attempt
    path=ROOT/'runs'/run_id/'manifest.json'
    if path.exists(): raise FileExistsError('Recorded intent; reconcile scheduler before retry')
    run_dir='/home6/u3129/mlip-research/runs/'+run_id
    if phase=='build':
        names=['build-grace-lammps.sbatch','build-grace-lammps.py']; cpus=8; mem=32; partition='apollo'; wall='00:45:00'
    else:
        names=['fm-deployment.sbatch','fm-deployment.py','restart-child.py']; cpus=4; mem=32; partition='v100'; wall='00:20:00'
    payload={n:(ROOT/'scripts/hpc'/n).read_bytes().replace(b'\r\n',b'\n') for n in names}
    if phase!='build':
        payload['source-backup.json']=(ROOT/'runs/fm-finetune-20261017-01/artifact-backup.json').read_bytes().replace(b'\r\n',b'\n')
        build=json.loads((ROOT/'runs/grace-lammps-build-20261017-01/manifest.json').read_text())
        if build.get('stage')!='collected' or build.get('runtime_status')!='passed':raise ValueError('Collect successful GRACE LAMMPS build first')
        payload['grace-lammps-build.json']=(ROOT/'runs/grace-lammps-build-20261017-01/result.json').read_bytes().replace(b'\r\n',b'\n')
    hashes={n:hashlib.sha256(b).hexdigest() for n,b in payload.items()}
    manifest={'run_id':run_id,'host':'umt.imm.uran.ru','remote_run_dir':run_dir,'phase':phase,
        'created_utc':datetime.now(timezone.utc).isoformat(),'stage':'preparing','job_id':None,
        'authorization':'user requested true checkpoint restart and export equivalence for ASE/LAMMPS on A100; bounded private prerequisite build if needed',
        'resources':{'partition':partition,'account':'imet-2026-2','qos':'normal','cpus':cpus,'nodes':1,'tasks':1,'memory_gib':mem,'walltime':wall,'gpus':0 if phase=='build' else 1},
        'budget':{'maximum_gpu_seconds':0 if phase=='build' else 1200,'maximum_cpu_allocation_seconds':cpus*(2700 if phase=='build' else 1200)},
        'input_sha256':hashes,'input_hash':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_run':'runs/fm-finetune-20261017-01/assessment.json',
        'requested_backends':backends.split(','),
        'grace_lammps_source_revision':'b1f4a6c8cc4e0d0fc93d2dec3a1fcc83558062d4'}
    save(path,manifest); q=shlex.quote(run_dir)
    remote('set -eu; umask 077; mkdir -p '+shlex.quote(run_dir.rsplit('/',1)[0])+'; mkdir '+q)
    for n,b in payload.items():write_remote(run_dir+'/'+n,b)
    manifest['stage']='submission_intent';save(path,manifest);write_remote(run_dir+'/manifest.json',path.read_bytes())
    flags=' --gres=gpu:a100:1' if phase!='build' else ''
    exports='ALL,DEPLOY_RUN_DIR='+run_dir
    if phase!='build':exports+=',DEPLOY_BACKENDS='+backends.replace(',','_')
    reply=remote('set -euC; cd '+q+'; test ! -e job.id; sbatch --parsable --job-name='+run_id+
        ' --partition='+partition+' --account=imet-2026-2 --qos=normal --nodes=1 --ntasks=1 --cpus-per-task='+str(cpus)+
        ' --mem='+str(mem)+'G --time='+wall+flags+' --chdir='+q+' --output='+q+'/slurm-%j.out --export='+shlex.quote(exports)+' '+names[0]+' > job.id; cat job.id')
    job=reply.strip().split(';')[0]
    if not job.isdigit(): raise RuntimeError('Ambiguous submission; reconcile job.id and scheduler')
    manifest.update(stage='submitted',job_id=job);save(path,manifest);write_remote(run_dir+'/manifest.json',path.read_bytes())
    print(json.dumps({'run_id':run_id,'job_id':job}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('submit');s.add_argument('phase',choices=['build','gpu']);s.add_argument('--attempt',required=True);s.add_argument('--backends',choices=['dpa4','grace','dpa4,grace'],default='dpa4,grace')
    c=sub.add_parser('collect');c.add_argument('manifest',type=Path)
    a=p.parse_args();submit(a.phase,a.attempt,a.backends) if a.command=='submit' else collect(a.manifest)
