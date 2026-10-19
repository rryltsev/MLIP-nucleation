"""Stage finite kpLib setup/grid/DFT jobs under the shared Al32 CPU budget."""
import argparse
import base64
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import shlex

from cluster_ssh import ROOT
from install_vasp import remote
from vasp_potentials import save
from al_pilot import submit,collect,backup,reserved_cpu_seconds

CONFIG=ROOT/'configs/dft/al32-kplib-benchmark-v1.json'


def stage(run_id,phase,source=None):
    assert re.fullmatch(r'al32-kplib-(setup|grids|benchmark)-\d{8}-\d{2}',run_id)
    path=ROOT/'runs'/run_id/'manifest.json'
    if path.exists(): raise FileExistsError(path)
    cfg=json.loads(CONFIG.read_text()); resources=dict(cfg['phase_limits'][phase],partition='apollo',account='imet-2026-2',qos='normal',nodes=1)
    resources.pop('maximum_reserved_CPU_hours')
    payload={'system.json':CONFIG.read_bytes(),'kplib-run.py':(ROOT/'scripts/hpc/kplib-run.py').read_bytes(),
             'kplib_grid.py':(ROOT/'scripts/kplib_grid.py').read_bytes(),'al-pilot.py':(ROOT/'scripts/hpc/al-pilot.py').read_bytes()}
    geometry=json.loads((ROOT/cfg['geometry_manifest']).read_text()); refs={}
    for row in geometry['configurations']:
        if row['configuration_id'] not in cfg['configuration_ids']:continue
        assert row['split']=='train'
        payload[row['poscar']]=(ROOT/'datasets/al32-pilot-v1'/row['poscar']).read_bytes()
        refs[row['configuration_id']]=dict(row=row,label=json.loads((ROOT/cfg['reference_run']/'calculations'/(row['configuration_id']+'-reference')/'label.json').read_text()))
    payload['references.json']=(json.dumps(refs,indent=2)+'\n').encode()
    if phase=='setup': payload['kpLib-1.1.1.tar.gz']=(ROOT/'artifacts/kplib-source/kpLib-1.1.1.tar.gz').read_bytes()
    if phase=='benchmark':
        if source is None:raise ValueError('Collected grid manifest required')
        sm=json.loads(source.read_text()); assert sm['stage']=='collected'
        result=json.loads((source.parent/'result.json').read_text()); assert result['passed']
        payload['grids.json']=(source.parent/'grids.json').read_bytes()
    header='#!/bin/bash\n'+''.join(f'#SBATCH --{key}={value}\n' for key,value in [('partition','apollo'),('account','imet-2026-2'),('qos','normal'),('nodes',1),('ntasks',resources['ntasks']),('cpus-per-task',resources['cpus_per_task']),('mem',str(resources['memory_GiB'])+'G')])
    python='/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python' if phase!='grids' else cfg['kplib']['environment']+'/bin/python'
    body='set -eo pipefail\numask 077\ncd "$PILOT_RUN_DIR"\nexport PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n'
    if phase=='benchmark':body+='source /home/u3129/.local/opt/vasp/6.6.1-oneapi-2025/env.sh\nsource /home/u3129/.local/share/vasp/pseudopotentials/64-b3b00158ad69/env.sh\n'
    body+=f'exec {python} kplib-run.py {phase}\n'
    payload['run.sbatch']=(header+body).encode()
    payload={k:(v if k.endswith('.tar.gz') else v.replace(b'\r\n',b'\n')) for k,v in payload.items()}
    m=dict(run_id=run_id,phase=phase,stage='preparing',job_id=None,created_utc=datetime.now(timezone.utc).isoformat(),host='umt.imm.uran.ru',user='u3129',
        remote_run_dir='/home6/u3129/mlip-research/runs/'+run_id,resources=resources,budget_group='al32-pilot',calibration_source='',
        code_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),input_sha256={k:hashlib.sha256(v).hexdigest() for k,v in payload.items()})
    m['maximum_CPU_hours']=reserved_cpu_seconds(m)/3600
    save(path,m)
    encoded=base64.b64encode(json.dumps({k:base64.b64encode(v).decode() for k,v in payload.items()}).encode()).decode()
    program="""import base64,hashlib,json
from pathlib import Path
root=Path(ROOT); root.mkdir(mode=0o700)
for name,encoded in json.loads(base64.b64decode(PAYLOAD)).items():
 p=root/name; p.parent.mkdir(parents=True,exist_ok=True)
 data=base64.b64decode(encoded)
 with p.open('xb') as f:f.write(data)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==hashlib.sha256(data).hexdigest()
""".replace('Path(ROOT)','Path('+repr(m['remote_run_dir'])+')').replace('PAYLOAD',repr(encoded))
    remote("umask 077; python3 - <<'PY'\n"+program+'\nPY\nbash -n '+shlex.quote(m['remote_run_dir']+'/run.sbatch'),120)
    m['stage']='staged';save(path,m);print(path)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    st=sub.add_parser('stage');st.add_argument('run_id');st.add_argument('phase',choices=['setup','grids','benchmark']);st.add_argument('--source',type=Path)
    su=sub.add_parser('submit');su.add_argument('manifest',type=Path);su.add_argument('--authorization',type=Path,required=True)
    for name in ('collect','backup'):
        s=sub.add_parser(name);s.add_argument('manifest',type=Path)
    a=p.parse_args()
    if a.action=='stage':stage(a.run_id,a.phase,a.source)
    elif a.action=='submit':submit(a.manifest.resolve(),a.authorization.resolve())
    elif a.action=='collect':collect(a.manifest.resolve())
    else:backup(a.manifest.resolve())
