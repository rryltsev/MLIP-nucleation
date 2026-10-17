"""Back up pinned LAMMPS source and compiled libraries; verify actual retrieval."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

from cluster_ssh import ROOT, ssh_args
from install_vasp import remote
from vasp_potentials import save

path=ROOT/'runs/grace-lammps-build-20261017-01'
build=json.loads((path/'result.json').read_text())
assert build['status']=='passed'
destination='/home/u3129/.local/share/mlip-research/test-artifacts/lammps-grace-'+build['upstream_git_commit'][:12]+'.tar'
script="""python3 - <<'PY'
from pathlib import Path
import hashlib,json,tarfile
root=Path(@ROOT@);archive=Path(@ARCHIVE@)
files=[root/'source.tar.gz',root/'build/lmp',root/'build/CMakeCache.txt']
files+=list((root/'build').rglob('*.so*'))
files=sorted(set(p for p in files if p.is_file() and not p.is_symlink()))
records={p.relative_to(root).as_posix():{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files}
archive.parent.mkdir(parents=True,exist_ok=True)
if not archive.exists():
 with archive.open('xb') as out:
  with tarfile.open(fileobj=out,mode='w') as tar:
   for p in files:tar.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
 archive.chmod(0o400)
with tarfile.open(archive) as tar:
 assert set(tar.getnames())==set(records)
 for n,r in records.items():assert hashlib.sha256(tar.extractfile(n).read()).hexdigest()==r['sha256']
h=hashlib.sha256()
with archive.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
print(json.dumps({'remote_archive':str(archive),'sha256':h.hexdigest(),'bytes':archive.stat().st_size,'members':records}))
PY
""".replace('@ROOT@',repr(build['prefix'])).replace('@ARCHIVE@',repr(destination))
record=json.loads(remote(script,timeout=180))
local=ROOT/'artifacts/fm-run-backups'/Path(destination).name;local.parent.mkdir(parents=True,exist_ok=True)
if not local.exists():
    partial=local.with_suffix('.partial')
    with partial.open('wb') as out:subprocess.run(ssh_args()+['-o','BatchMode=yes','mlip-umt','cat '+shlex.quote(destination)],stdout=out,check=True,timeout=600)
    partial.replace(local)
with local.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==record['sha256']
with tarfile.open(local) as tar:
    for n,r in record['members'].items():assert hashlib.sha256(tar.extractfile(n).read()).hexdigest()==r['sha256']
record.update(local_archive=str(local),verified_retrieval=True,scope='private source and binaries; TensorFlow runtime rebuilt separately from existing exact environment lock')
save(path/'artifact-backup.json',record)
print(json.dumps({k:v for k,v in record.items() if k!='members'}))
