"""Persist bounded Al32 Slurm submissions and collect only explicit public data.

Never blindly retry submit. A submission_intent without job_id requires reconciliation.
Licensed POTCAR and raw VASP outputs are backed up privately, never collected to Git.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

from cluster_ssh import ROOT, ssh_args
from install_vasp import remote, write_remote
from vasp_potentials import save

SYSTEM = ROOT / "configs/systems/al32-pilot-v1.json"
REMOTE = "/home6/u3129/mlip-research/runs/"


def stage(run_id, phase, source=None):
    if not re.fullmatch(r"al32-(calibration|labeling)-[0-9]{8}-[0-9]{2}", run_id):
        raise ValueError("Use a unique al32-calibration/labeling-YYYYMMDD-NN identifier")
    path = ROOT/"runs"/run_id/"manifest.json"
    if path.exists():
        raise FileExistsError("Run already exists")
    cfg = json.loads(SYSTEM.read_text())
    if phase == "label":
        if source is None:
            raise ValueError("Calibration manifest required")
        src = json.loads(source.read_text())
        cal = json.loads((source.parent/"calibration.json").read_text())
        if src["stage"] != "collected" or not cal["passed"]:
            raise ValueError("Collect accepted calibration first")
        source_remote = src["remote_run_dir"]
    else:
        source_remote = ""
    payloads = {"system.json": SYSTEM.read_bytes(),
                "al-pilot.py": (ROOT/"scripts/hpc/al-pilot.py").read_bytes(),
                "run.sbatch": (ROOT/"scripts/hpc/al-pilot.sbatch").read_bytes()}
    if phase == "calibrate":
        prepared = ROOT/"datasets/al32-pilot-v1"
        manifest=json.loads((prepared/"dataset-manifest.json").read_text())
        payloads["dataset-manifest.json"]=(prepared/"dataset-manifest.json").read_bytes()
        for row in manifest["configurations"]:
            data=(prepared/row["poscar"]).read_bytes()
            if hashlib.sha256(data).hexdigest()!=row["poscar_sha256"]:
                raise ValueError("Prepared structure hash mismatch")
            payloads[row["poscar"]]=data
    payloads = {k:v.replace(b"\r\n",b"\n") for k,v in payloads.items()}
    m = dict(run_id=run_id, phase=phase, stage="preparing", job_id=None,
        created_utc=datetime.now(timezone.utc).isoformat(), remote_run_dir=REMOTE+run_id,
        host="umt.imm.uran.ru", user="u3129", calibration_source=source_remote,
        code_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        input_sha256={k:hashlib.sha256(v).hexdigest() for k,v in payloads.items()},
        resources=dict(partition="apollo",account="imet-2026-2",qos="normal",nodes=1,
                       ntasks=16,cpus_per_task=1,memory_GiB=32,
                       walltime="02:00:00" if phase=="calibrate" else "06:00:00"),
        maximum_CPU_hours=32 if phase=="calibrate" else 96,
        authorization="not supplied: explicit authorization record required at submit")
    save(path,m)
    remote("set -eu; umask 077; test ! -e "+shlex.quote(m["remote_run_dir"])+"; mkdir "+shlex.quote(m["remote_run_dir"]))
    directories=sorted({str(Path(name).parent).replace('\\','/') for name in payloads if '/' in name})
    if directories:
        remote("set -eu; umask 077; mkdir -p "+" ".join(shlex.quote(m["remote_run_dir"]+"/"+d) for d in directories))
    for name,data in payloads.items():
        write_remote(m["remote_run_dir"]+"/"+name,data)
    m["stage"]="staged"
    save(path,m)
    print(path)


def submit(path, authorization):
    m=json.loads(path.read_text())
    if m["stage"]!="staged" or m["job_id"]:
        raise ValueError("Reconcile scheduler before any repeated submit")
    auth=json.loads(authorization.read_text())
    if not auth.get("user_authorized") or auth["maximum_CPU_hours"] < 128:
        raise ValueError("User-authorized 128 CPU-hour cap required")
    # This allocation covers exactly one calibration and one labeling job, no retries.
    other=[]
    for p in (ROOT/"runs").glob("al32-*/manifest.json"):
        x=json.loads(p.read_text())
        if p != path.resolve() and x.get("authorization_sha256")==hashlib.sha256(authorization.read_bytes()).hexdigest():
            if x["phase"]==m["phase"] and x["stage"] in ("submission_intent","submitted","collected"):
                raise ValueError("Budget phase already consumed; no automatic retry")
            other.append(x["maximum_CPU_hours"])
    if sum(other)+m["maximum_CPU_hours"] > auth["maximum_CPU_hours"]:
        raise ValueError("Reservation exceeds budget")
    m.update(stage="submission_intent", authorization=str(authorization.relative_to(ROOT)),
             authorization_sha256=hashlib.sha256(authorization.read_bytes()).hexdigest())
    save(path,m)
    write_remote(m["remote_run_dir"]+"/submission-intent.json",path.read_bytes())
    exports="ALL,PILOT_RUN_DIR="+m["remote_run_dir"]+",PILOT_PHASE="+m["phase"]+",PILOT_CODE_COMMIT="+m["code_commit"]
    if m["calibration_source"]: exports+=",PILOT_SOURCE="+m["calibration_source"]
    cmd="set -eu; cd "+shlex.quote(m["remote_run_dir"])+"; test ! -e job.id; sbatch --parsable --job-name="+shlex.quote(m["run_id"])+" --time="+m["resources"]["walltime"]+" --chdir=/tmp --output="+shlex.quote(m["remote_run_dir"]+"/slurm-%j.out")+" --export="+shlex.quote(exports)+" run.sbatch > job.id; cat job.id"
    job=remote(cmd).strip().split(";")[0]
    if not re.fullmatch(r"\d+",job): raise RuntimeError("Uncertain outcome; inspect job.id and scheduler")
    m.update(stage="submitted",job_id=job)
    save(path,m)
    print(job)


def collect(path):
    m=json.loads(path.read_text())
    if not m["job_id"]: raise ValueError("Reconcile submission first")
    state=remote("sacct -n -P -j "+m["job_id"]+" --format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,CPUTimeRAW").strip()
    lines=[l.split("|") for l in state.splitlines() if l.split("|")[0]==m["job_id"]]
    if not lines: raise RuntimeError("Accounting unknown; do not mark collected")
    if lines[0][1] not in ("COMPLETED","FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL"):
        print(state); return
    # Strict allowlist: no raw VASP, POTCAR or unreviewed binary artifacts in Git.
    program=r'''import base64,hashlib,json
from pathlib import Path
root=Path(ROOT)
names=['system.json','al-pilot.py','run.sbatch','submission-intent.json','dataset-manifest.json','calibration.json','calibration-source.json','labels.json','result.json','export-manifest.json']
names += [str(p.relative_to(root)) for p in root.glob('structures/*/POSCAR')]
names += [str(p.relative_to(root)) for p in root.glob('calculations/*/*') if p.name in ('POSCAR','INCAR','KPOINTS','input-manifest.json','potcar-provenance.json','label.json')]
names += [str(p.relative_to(root)) for p in root.glob('export/*.extxyz')]
files={}
for name in names:
 p=root/name
 if p.is_file():
  data=p.read_bytes(); data.decode('utf-8')
  files[name]={'sha256':hashlib.sha256(data).hexdigest(),'data':base64.b64encode(data).decode()}
print(json.dumps(files))'''.replace('Path(ROOT)', 'Path('+repr(m["remote_run_dir"])+')')
    files=json.loads(remote("python3 - <<'PY'\n"+program+"\nPY",180))
    import base64
    for name, item in files.items():
        target=path.parent/name
        target.parent.mkdir(parents=True,exist_ok=True)
        data=base64.b64decode(item["data"])
        assert hashlib.sha256(data).hexdigest()==item["sha256"]
        if target.exists() and target.read_bytes()!=data:
            raise ValueError("Immutable collected file differs: "+name)
        target.write_bytes(data)
    save(path.parent/"collection.json",dict(sacct=state,sha256={k:v["sha256"] for k,v in files.items()},
          verified=True,exclusions="POTCAR, raw VASP outputs and binary exports remain private"))
    m.update(stage="collected",scheduler_record=lines[0],collected_utc=datetime.now(timezone.utc).isoformat())
    save(path,m)
    print(json.dumps(dict(run_id=m["run_id"],scheduler=lines[0],files=len(files))))


def backup(path):
    m=json.loads(path.read_text())
    if m["stage"]!="collected": raise ValueError("Collect terminal run first")
    dest="/home/u3129/.local/share/mlip-research/dft-artifacts/"+m["run_id"]+".tar.gz"
    # Includes licensed POTCAR privately; never upload the archive to GitHub.
    program=r'''import hashlib,json,tarfile
from pathlib import Path
root=Path(SOURCE); archive=Path(DEST)
files=sorted(p for p in root.rglob('*') if p.is_file() and not p.name.startswith('core'))
records={str(p.relative_to(root)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files}
archive.parent.mkdir(parents=True,exist_ok=True)
if not archive.exists():
 with archive.open('xb') as f:
  with tarfile.open(fileobj=f,mode='w:gz') as t:
   for p in files: t.add(p,arcname=str(p.relative_to(root)),recursive=False)
 archive.chmod(0o400)
with tarfile.open(archive) as t:
 assert {m.name for m in t.getmembers()}==set(records)
 for name,record in records.items(): assert hashlib.sha256(t.extractfile(name).read()).hexdigest()==record['sha256']
print(json.dumps(dict(remote_archive=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,members=records)))'''.replace('Path(SOURCE)','Path('+repr(m["remote_run_dir"])+')').replace('Path(DEST)','Path('+repr(dest)+')')
    record=json.loads(remote("python3 - <<'PY'\n"+program+"\nPY",300))
    local=ROOT/"artifacts/dft-run-backups"/Path(dest).name
    local.parent.mkdir(parents=True,exist_ok=True)
    if not local.exists():
        partial=local.with_suffix(".partial")
        with partial.open("wb") as out:
            subprocess.run(ssh_args()+["-o","BatchMode=yes","mlip-umt","cat "+shlex.quote(dest)],stdout=out,check=True,timeout=600)
        partial.replace(local)
    with local.open("rb") as f: assert hashlib.file_digest(f,"sha256").hexdigest()==record["sha256"]
    with tarfile.open(local) as t:
        for n,r in record["members"].items(): assert hashlib.sha256(t.extractfile(n).read()).hexdigest()==r["sha256"]
    record.update(local_archive=str(local),verified_retrieval=True,
        policy="Private cluster main-home archive plus actual workstation copy; retention/institutional backup unconfirmed; contains licensed material, excluded from Git")
    save(path.parent/"artifact-backup.json",record)
    print(json.dumps({k:v for k,v in record.items() if k!="members"}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="action",required=True)
    st=sub.add_parser("stage"); st.add_argument("run_id"); st.add_argument("phase",choices=["calibrate","label"]); st.add_argument("--source",type=Path)
    su=sub.add_parser("submit"); su.add_argument("manifest",type=Path); su.add_argument("--authorization",required=True,type=Path)
    for name in ("collect","backup"):
        s=sub.add_parser(name); s.add_argument("manifest",type=Path)
    a=p.parse_args()
    if a.action=="stage": stage(a.run_id,a.phase,a.source)
    elif a.action=="submit": submit(a.manifest.resolve(),a.authorization.resolve())
    elif a.action=="collect": collect(a.manifest.resolve())
    else: backup(a.manifest.resolve())
