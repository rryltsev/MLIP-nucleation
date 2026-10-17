"""Submit and collect immutable, bounded MLIP runtime checks on UMT."""
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

PROFILES = {
    "a100": {"partition": "v100", "gres": "gpu:a100:1", "environment": "deepmd_320_a100", "backend": "pt"},
    "k40m": {"partition": "debug", "gres": "gpu:k40m:1", "environment": "deepmd_2210_k40", "backend": "tf"},
}


def submit(profile):
    cfg = PROFILES[profile]
    run_id = "mlip-smoke-20261017-" + profile + "-01"
    path = ROOT / "runs" / run_id / "manifest.json"
    if path.exists():
        raise FileExistsError("Run already recorded; reconcile scheduler/remote job.id before any retry")
    run_dir = "/home6/u3129/mlip-research/runs/" + run_id
    payloads = {n: (ROOT / "scripts/hpc" / n).read_bytes().replace(b"\r\n", b"\n")
                for n in ["mlip-smoke.py", "mlip-smoke.sbatch"]}
    manifest = {"run_id": run_id, "host": "umt.imm.uran.ru", "remote_run_dir": run_dir,
                "created_utc": datetime.now(timezone.utc).isoformat(), "purpose": "user-requested small MLIP GPU training checks",
                "authorization": "user requested small Slurm tests on A100 and K40m; initial cap two single-GPU jobs of ten minutes each",
                "profile": cfg, "resources": {"account": "imet-2026-2", "qos": "normal", "nodes": 1,
                "tasks": 1, "cpus": 4, "memory_gib": 16, "walltime": "00:10:00", "gpus": 1},
                "input_sha256": {n: hashlib.sha256(b).hexdigest() for n, b in payloads.items()},
                "input_hash": hashlib.sha256(json.dumps({n: hashlib.sha256(b).hexdigest() for n,b in payloads.items()},sort_keys=True).encode()).hexdigest(),
                "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "label_policy": "synthetic harmonic pairs only; no DFT labels or FM predictions",
                "stage": "preparing", "job_id": None}
    save(path, manifest)
    q = shlex.quote(run_dir)
    remote("set -eu; umask 077; test ! -e " + q + "; mkdir -p " + q)
    for name, content in payloads.items():
        write_remote(run_dir + "/" + name, content)
    manifest["stage"] = "submission_intent"; save(path, manifest)
    write_remote(run_dir + "/manifest.json", path.read_bytes())
    exports = "ALL,MLIP_RUN_DIR=" + run_dir + ",MLIP_ENV=" + cfg["environment"] + ",MLIP_BACKEND=" + cfg["backend"]
    response = remote("set -euC; cd " + q + "; test ! -e job.id; sbatch --parsable --job-name=" + run_id +
                      " --partition=" + cfg["partition"] + " --gres=" + cfg["gres"] +
                      " --account=imet-2026-2 --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:10:00" +
                      " --chdir=" + q + " --output=" + q + "/slurm-%j.out --export=" + shlex.quote(exports) +
                      " mlip-smoke.sbatch > job.id; cat job.id")
    job = response.strip().split(";")[0]
    if not job.isdigit():
        raise RuntimeError("Uncertain submission; inspect job.id and scheduler before retrying")
    manifest.update(stage="submitted", job_id=job); save(path, manifest)
    write_remote(run_dir + "/manifest.json", path.read_bytes())
    print(json.dumps({"run_id": run_id, "job_id": job, "profile": cfg}))


def collect(path):
    manifest = json.loads(path.read_text())
    job = manifest["job_id"]
    if not job or not job.isdigit():
        raise ValueError("Job unknown; reconcile remote job.id")
    state = remote("sacct -X -j " + job + " -nP --format=JobIDRaw,State,ExitCode,Elapsed,AllocCPUS,NodeList; "
                   "squeue -h -j " + job + " -o '%i|%T|%R'")
    save(path.parent / "scheduler.json", {"observed_utc": datetime.now(timezone.utc).isoformat(), "output": state})
    if any("|" + s in state for s in ["RUNNING", "PENDING", "CONFIGURING", "COMPLETING"]):
        print(state); return
    root = shlex.quote(manifest["remote_run_dir"])
    # Retrieve only small records. Checkpoints stay on the cluster with hashes.
    script = """python3 - <<'REMOTE'
import hashlib,json
from pathlib import Path
root=Path(@ROOT@)
names=['result.json','node.txt','gpu.csv','nvidia-smi.txt','pip-freeze.txt','slurm-@JOB@.out','data/provenance.json']
names += [str(p.relative_to(root)) for pattern in ['*/input.json','*/lcurve.out','*/train.log'] for p in root.glob(pattern)]
records={}
for name in names:
 p=root/name
 if p.is_file():
  data=p.read_bytes()
  if len(data)>1000000: raise ValueError('Record too large')
  records[name]={'sha256':hashlib.sha256(data).hexdigest(),'text':data.decode(errors='replace')}
print(json.dumps(records))
REMOTE
""".replace("@ROOT@", repr(manifest["remote_run_dir"])).replace("@JOB@", job)
    records = json.loads(remote(script))
    for name, record in records.items():
        data = record["text"].encode()
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("Transport checksum differs: " + name)
        target = path.parent / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
    save(path.parent / "collected-sha256.json", {n:r["sha256"] for n,r in records.items()})
    result = json.loads(records["result.json"]["text"]) if "result.json" in records else {"status":"missing_result"}
    manifest.update(stage="collected", runtime_status=result["status"], scheduler_observation=state)
    save(path, manifest)
    print(json.dumps({"run_id":manifest["run_id"],"scheduler":state,"runtime_status":result["status"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("submit"); p.add_argument("profile", choices=PROFILES)
    p = sub.add_parser("collect"); p.add_argument("manifest", type=Path)
    args = parser.parse_args()
    submit(args.profile) if args.command == "submit" else collect(args.manifest)
