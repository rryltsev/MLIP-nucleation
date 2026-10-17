"""Collect lightweight, read-only login-node observations; never submits jobs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from cluster_ssh import ROOT, ssh_args

PROBES = {
    "identity": "hostname -f; id; uname -sr; cat /etc/os-release; printf 'HOME=%s\\n' \"$HOME\"",
    "partitions": "scontrol --version; sinfo -h -o '%P|%a|%l|%D|%c|%m|%G|%t|%N'; scontrol show partition",
    "nodes": "scontrol show nodes -o",
    "account": "sacctmgr -nP show assoc where user=\"$(id -un)\" format=Cluster,Account,User,Partition,QOS,DefaultQOS,GrpTRES,MaxTRES,MaxJobs,MaxSubmit,MaxWall; squeue -u \"$(id -un)\" -o '%.18i %.12P %.12j %.8T %.10M %.6D %R'",
    "slurm_policy": "scontrol show config | sed -n -E '/^[[:space:]]*(AccountingStorageEnforce|SelectType|SelectTypeParameters|TaskPlugin|SchedulerType|DefMemPerCPU|DefMemPerNode|MaxMemPerCPU|MaxMemPerNode|GresTypes)[[:space:]]*=/p'",
    "modules": "type module; if type module >/dev/null 2>&1; then module -t list 2>&1; module -t avail 2>&1; fi",
    "executables": "for app in vasp vasp_std vasp_gam vasp_ncl python python3 conda mamba micromamba nvcc nvidia-smi gcc g++ mpirun mpiexec lmp lmp_mpi apptainer singularity; do command -v \"$app\" || :; done; python3 --version 2>&1; if command -v conda >/dev/null 2>&1; then conda env list; fi",
    "storage": "df -h \"$HOME\" /tmp; quota -s 2>&1; for path in /opt /usr/local /apps /shared /scratch /work; do if [ -d \"$path\" ]; then ls -ld \"$path\"; ls -1 \"$path\" | head -60; fi; done",
    "account_limits": "sacctmgr -nP show user where name=\"$(id -un)\" format=User,DefaultAccount; sacctmgr -nP show qos where name=normal,restricted format=Name,Flags,GrpTRES,GrpTRESMins,MaxTRESPerUser,MaxJobsPerUser,MaxSubmitJobsPerUser,MaxWall; sacctmgr -nP show assoc where account=imet-2026-2 user='' format=Cluster,Account,User,ParentName,QOS,GrpTRES,GrpTRESMins,GrpTRESRunMins,MaxTRES,MaxWall",
    "software_paths": "find /opt -maxdepth 1 \\( -iname '*vasp*' -o -iname '*lammps*' -o -iname '*miniconda*' -o -iname '*pytorch*' \\) -print; for path in \"$HOME/VASP\" \"$HOME/vasp\" \"$HOME/SOFT\" \"$HOME/Keller/SOFT\"; do if [ -d \"$path\" ]; then find \"$path\" -maxdepth 3 -type f \\( -name 'vasp_std' -o -name 'vasp_gam' -o -name 'vasp_ncl' -o -name 'vasp' \\) -print; fi; done",
    "environment_metadata": """python3 - <<'PY'
import glob, json, os
prefixes = ['/usr', os.path.expanduser('~/miniconda3')] + glob.glob(os.path.expanduser('~/miniconda3/envs/*'))
seen = set()
for prefix in prefixes:
    real = os.path.realpath(prefix)
    if real in seen: continue
    seen.add(real)
    packages = {}
    for path in glob.glob(prefix + '/conda-meta/*.json'):
        try:
            with open(path) as handle: p = json.load(handle)
            name = p.get('name', '')
            if any(word in name for word in ['python', 'torch', 'tensorflow', 'deepmd', 'dpdata', 'ase', 'mace', 'nequip', 'numpy', 'mpi', 'lammps', 'fairchem', 'cuda', 'cudnn', 'e3nn']):
                packages[name] = {'version': p.get('version'), 'build': p.get('build'), 'source': 'conda-meta'}
        except (OSError, ValueError): pass
    for path in glob.glob(prefix + '/lib/python*/site-packages/*.dist-info/METADATA'):
        try:
            name = version = None
            with open(path) as handle:
                for line in handle:
                    if line.startswith('Name: '): name = line[6:].strip().lower()
                    if line.startswith('Version: '): version = line[9:].strip()
                    if name and version: break
            if name and any(word in name for word in ['torch', 'tensorflow', 'deepmd', 'dpdata', 'ase', 'mace', 'nequip', 'numpy', 'mpi', 'lammps', 'fairchem', 'cuda', 'cudnn', 'e3nn']):
                packages[name] = {'version': version, 'source': 'dist-info'}
        except OSError: pass
    print(json.dumps({'prefix': prefix, 'realpath': real, 'python_exists': os.path.isfile(prefix + '/bin/python'), 'packages': packages}, sort_keys=True))
PY""",
    "vasp_layout": "ls -ld /opt/vasp; ls -1 /opt/vasp; find /opt/vasp -maxdepth 4 -type f \\( -name 'vasp*' -o -name '*.sh' -o -name 'README*' \\) -not -path '*/src/*' -not -path '*/pot*/*' -print; ls -1 \"$HOME\" | head -70; if [ -d \"$HOME/Keller/SOFT\" ]; then ls -1 \"$HOME/Keller/SOFT\"; fi",
    "module_details": "module show gpu/cuda-11.8 gpu/cuda-12.4 gpu/cuda-13.0 openmpi/gcc64_4.1.1 intelmpi/intel64_2025 2>&1; ls -ld /usr/local/cuda* /opt/cuda* /opt/conda 2>/dev/null",
    "user_software": "for path in \"$HOME/Keller\" \"$HOME/Sirenko\" \"$HOME/Submission\" \"$HOME/_scratch\" \"$HOME/_scratch2\"; do if [ -d \"$path\" ]; then ls -ld \"$path\"; ls -1 \"$path\" | head -40; fi; done; for path in /usr/local/bin /usr/bin \"$HOME/Keller/SOFT\"; do find \"$path\" -maxdepth 3 -type f -iname '*vasp*' -print 2>/dev/null; done; ls -l \"$HOME/miniconda3/envs/deepmd_320_a100/bin/python\" \"$HOME/miniconda3/envs/deepmd_320_a100/bin/dp\"; sshare -l -A imet-2026-2 -u \"$(id -un)\"",
    "environment_activation": "source \"$HOME/miniconda3/etc/profile.d/conda.sh\"; for env in deepmd_320_a100 deepmd_229_cpu deepmd_2210_k40; do conda activate \"$env\" || exit; printf 'ENV=%s\\n' \"$env\"; command -v python; python --version; command -v dp; command -v lmp; conda deactivate; done",
}


def collect(name, script, timeout):
    observed = datetime.now(timezone.utc).isoformat()
    command = ssh_args() + ["-o", "BatchMode=yes", "-o", "PreferredAuthentications=publickey",
                            "mlip-umt", "timeout %ds bash -s" % max(1, timeout - 3)]
    try:
        # Send LF bytes: Windows text-mode stdin otherwise produces CRLF for bash.
        result = subprocess.run(command, input=(script + "\n").encode("utf-8"),
                                capture_output=True, timeout=timeout)
        return {"name": name, "observed_utc": observed, "script": script,
                "returncode": result.returncode, "timed_out": result.returncode == 124,
                "stdout": result.stdout.decode("utf-8", errors="replace"),
                "stderr": result.stderr.decode("utf-8", errors="replace")}
    except subprocess.TimeoutExpired as exc:
        def decode(value):
            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
        return {"name": name, "observed_utc": observed, "script": script, "timed_out": True,
                "stdout": decode(exc.stdout), "stderr": decode(exc.stderr)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--probe", choices=PROBES, action="append")
    parser.add_argument("--timeout", type=int, default=35)
    args = parser.parse_args()
    document = {"host": "umt.imm.uran.ru", "user": "u3129", "scope": "read-only login-node inspection; no job submission", "probes": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in args.probe or PROBES:
        result = collect(name, PROBES[name], args.timeout)
        document["probes"].append(result)
        args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"probe": name, "returncode": result.get("returncode"),
                          "timed_out": result.get("timed_out", False),
                          "stdout_chars": len(result["stdout"]), "stderr": result["stderr"][:1200]},
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
