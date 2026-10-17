"""Copy the inspected licensed PAW distribution to a new private cluster library."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess

from cluster_ssh import ROOT, ssh_args
from install_vasp import remote, write_remote
from vasp_potentials import file_sha, inspect, save


def install(archive):
    registry_path = ROOT / "configs/vasp/paw-64.json"
    expected = json.loads(registry_path.read_text(encoding="utf-8"))
    local = inspect(archive)
    if local != expected:
        raise ValueError("Archive differs from inspected registry; reconcile before installation")
    root = "/home/u3129/.local/share/vasp/pseudopotentials/64-" + local["archive_sha256"][:12]
    run = ROOT / "runs/paw64-20261017-01"
    manifest_path = run / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Existing installation record; reconcile remote state before retrying")
    payloads = {
        "vasp_potentials.py": ROOT / "scripts/vasp_potentials.py",
        "env.sh": ROOT / "scripts/hpc/paw-env.sh",
    }
    manifest = {
        "run_id": run.name, "purpose": "user-requested private PAW library installation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": "umt.imm.uran.ru", "user": "u3129", "remote_root": root,
        "archive_sha256": local["archive_sha256"], "archive_bytes": local["archive_bytes"],
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "input_sha256": {n: file_sha(p) for n, p in payloads.items()},
        "installer_sha256": file_sha(Path(__file__)),
        "stage": "transfer_intent", "scheduler_job_id": None,
        "execution": "login-node file transfer, extraction and checksum verification only",
    }
    save(manifest_path, manifest)
    q = shlex.quote(root)
    remote("set -eu; umask 077; test ! -e " + q + "; mkdir -p " + q + "; chmod 700 " + q)
    write_remote(root + "/installation.json", manifest_path.read_bytes())
    print("Transferring licensed archive (57.6 MB)...", flush=True)
    with archive.open("rb") as stream:
        subprocess.run(ssh_args() + ["-o", "BatchMode=yes", "-o", "PreferredAuthentications=publickey",
                       "mlip-umt", "set -euC; umask 077; cat > " + q + "/Pseudopotentials_v6.4.zip.partial"],
                       stdin=stream, check=True, timeout=900)
    remote("set -eu; cd " + q + "; printf '%s  %s\\n' " + local["archive_sha256"] +
           " Pseudopotentials_v6.4.zip.partial | sha256sum -c -; mv -n Pseudopotentials_v6.4.zip.partial Pseudopotentials_v6.4.zip")
    manifest["stage"] = "archive_verified"
    save(manifest_path, manifest)
    for name, path in payloads.items():
        write_remote(root + "/" + name, path.read_bytes().replace(b"\r\n", b"\n"))
    print("Extracting and verifying all installed files...", flush=True)
    print(remote("set -eu; umask 077; python3 " + q + "/vasp_potentials.py unpack " +
                 q + "/Pseudopotentials_v6.4.zip --root " + q + " --expected-sha256 " + local["archive_sha256"],
                 timeout=180).strip())
    observed = json.loads(remote("cat " + q + "/registry.json"))
    if {k: observed[k] for k in expected} != expected:
        raise ValueError("Remote registry differs from local inventory")
    save(registry_path, observed)
    manifest.update(stage="installed_verified", completed_utc=datetime.now(timezone.utc).isoformat(),
                    registry="configs/vasp/paw-64.json", registry_sha256=file_sha(registry_path),
                    family_counts={f: x["potential_count"] for f, x in observed["families"].items()},
                    verification="ZIP SHA-256, nested archive SHA-256, all extracted file hashes and local/remote inventory match")
    save(manifest_path, manifest)
    write_remote(root + "/installation.json", manifest_path.read_bytes())
    print("Installed and verified: " + root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    install(parser.parse_args().archive)
