"""Verify the full private cluster archive and retrieve a compact audit bundle.

This is an explicit alternative backend when a full workstation transfer is
impractical. Native weights remain in the full cluster archive; the workstation
bundle contains every input and small textual scientific output, not the weights.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

from cluster_ssh import ROOT,ssh_args
from install_vasp import remote


def main(run):
    manifest=json.loads((run/'manifest.json').read_text());assert manifest['stage']=='collected'
    archive='/home/u3129/.local/share/mlip-research/dft-artifacts/'+manifest['run_id']+'.tar.gz'
    program=r'''import hashlib,json,tarfile
from pathlib import Path
archive=Path(ARCHIVE); evidence=archive.with_name(archive.name.replace('.tar.gz','-evidence.tar.gz'))
inputs=set(INPUTS)
def digest(f):
 h=hashlib.sha256()
 for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
records={}; selected=[]
with tarfile.open(str(archive)) as source:
 members=source.getmembers()
 assert all(m.isfile() for m in members)
 assert len({m.name for m in members})==len(members)
 for m in members:
  with source.extractfile(m) as f:records[m.name]={'sha256':digest(f),'bytes':m.size}
  if m.name in inputs or (m.size<=2500000 and 'cache' not in m.name.split('/') and 'loaded-initial' not in m.name.split('/') and Path(m.name).suffix in ('.json','.extxyz','.yaml','.txt','.out','.log','.py','.sbatch','.csv')):selected.append(m)
 if not evidence.exists():
  with evidence.open('xb') as target:
   with tarfile.open(fileobj=target,mode='w:gz') as output:
    for m in selected:output.addfile(m,source.extractfile(m))
  evidence.chmod(0o400)
with tarfile.open(str(evidence)) as t:
 assert {m.name for m in t.getmembers()}=={m.name for m in selected}
 for m in t.getmembers():
  with t.extractfile(m) as f:assert digest(f)==records[m.name]['sha256']
with archive.open('rb') as f:fullhash=digest(f)
with evidence.open('rb') as f:evidencehash=digest(f)
print(json.dumps(dict(remote_archive=str(archive),sha256=fullhash,bytes=archive.stat().st_size,members=records,
 remote_archive_verified=True,remote_evidence_archive=str(evidence),evidence_sha256=evidencehash,evidence_bytes=evidence.stat().st_size)))
'''.replace('Path(ARCHIVE)','Path('+repr(archive)+')').replace('set(INPUTS)','set('+repr(list(manifest['input_sha256']))+')')
    record=json.loads(remote("python3 - <<'PY'\n"+program+"\nPY",300))
    record.update(observed_utc=datetime.now(timezone.utc).isoformat(),verified_retrieval=False,
        local_full_archive_status='Incomplete transfer; not a verified backup',
        verification_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        policy='Full native archive verified in cluster main home; actual audit bundle on workstation. Native weights are not fully copied to workstation; institutional retention unconfirmed.')
    path=run/'artifact-backup.json'
    path.write_text(json.dumps(record,indent=2)+'\n')
    local=ROOT/'artifacts/dft-run-backups'/Path(record['remote_evidence_archive']).name
    if not local.exists():
        partial=local.with_suffix('.partial')
        with partial.open('wb') as output:
            subprocess.run(ssh_args()+['-o','BatchMode=yes','mlip-umt','cat '+shlex.quote(record['remote_evidence_archive'])],stdout=output,check=True,timeout=180)
        assert hashlib.sha256(partial.read_bytes()).hexdigest()==record['evidence_sha256']
        partial.replace(local)
    assert hashlib.sha256(local.read_bytes()).hexdigest()==record['evidence_sha256']
    record.update(local_evidence_archive=str(local),evidence_verified_retrieval=True)
    path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='members'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    main(p.parse_args().run.resolve())
