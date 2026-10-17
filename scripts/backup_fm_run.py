"""Archive completed diagnostic checkpoints in main home and verify local retrieval."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

from cluster_ssh import ROOT, ssh_args
from install_vasp import remote
from vasp_potentials import save


def main(path, initialization=False):
    manifest=json.loads(path.read_text())
    if manifest['stage']!='collected': raise ValueError('Collect terminal scheduler state first')
    source=manifest['remote_run_dir']
    suffix='-initialization' if initialization else ''
    destination='/home/u3129/.local/share/mlip-research/test-artifacts/'+manifest['run_id']+suffix+'.tar'
    script="""python3 - <<'PY'
import hashlib,json,tarfile
from pathlib import Path
root=Path(@SOURCE@); archive=Path(@DEST@)
patterns=['*.py','*.sbatch','*.json','*/input.json','*/input.yaml','*/result.json','*/data/**/*','dpa4/model.ckpt-*.pt','grace/seed/42/model.yaml','grace/seed/42/checkpoints/checkpoint.index','grace/seed/42/checkpoints/checkpoint.data-*']
if @INITIALIZATION@: patterns=['*/loaded-initial/*','*/restored/*']
else: patterns+=['*/exported.pt2','grace/seed/42/saved_model/**/*','*/comparison.extxyz','*/lammps-*/*']
files=sorted({p for pat in patterns for p in root.glob(pat) if p.is_file() and p.name!='core' and not p.name.startswith('core.')})
records={p.relative_to(root).as_posix():{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files}
archive.parent.mkdir(parents=True,exist_ok=True)
if not archive.exists():
 with archive.open('xb') as stream:
  with tarfile.open(fileobj=stream,mode='w') as tar:
   for p in files: tar.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
 archive.chmod(0o400)
with tarfile.open(archive) as tar:
 members={m.name:m for m in tar.getmembers()}
 assert members.keys()==records.keys()
 for name,record in records.items():
  assert hashlib.sha256(tar.extractfile(members[name]).read()).hexdigest()==record['sha256']
digest_state=hashlib.sha256()
with archive.open('rb') as f:
 for chunk in iter(lambda:f.read(1048576),b''): digest_state.update(chunk)
digest=digest_state.hexdigest()
print(json.dumps({'remote_archive':str(archive),'sha256':digest,'bytes':archive.stat().st_size,'members':records}))
PY
""".replace('@SOURCE@',repr(source)).replace('@DEST@',repr(destination)).replace('@INITIALIZATION@',str(initialization))
    record=json.loads(remote(script,timeout=180))
    local=ROOT/'artifacts/fm-run-backups'/Path(destination).name; local.parent.mkdir(parents=True,exist_ok=True)
    if not local.exists():
        partial=local.with_suffix('.partial')
        with partial.open('wb') as stream:
            subprocess.run(ssh_args()+['-o','BatchMode=yes','mlip-umt','cat '+shlex.quote(destination)],stdout=stream,check=True,timeout=600)
        partial.replace(local)
    with local.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
    if actual!=record['sha256']: raise ValueError('Local archive hash mismatch')
    with tarfile.open(local) as tar:
        for name,item in record['members'].items():
            assert hashlib.sha256(tar.extractfile(name).read()).hexdigest()==item['sha256']
    record.update(local_archive=str(local),verified_retrieval=True,
        policy='Actual private main-home archive and second workstation copy; no weights uploaded to GitHub; institutional backup/retention unconfirmed')
    save(path.parent/('initialization-backup.json' if initialization else 'artifact-backup.json'),record)
    print(json.dumps({k:v for k,v in record.items() if k!='members'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path);p.add_argument('--initialization',action='store_true')
    a=p.parse_args();main(a.manifest,a.initialization)
