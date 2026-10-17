"""Stage licensed VASP privately on UMT, then submit a bounded Slurm build."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import shlex
import subprocess
import tarfile

from cluster_ssh import ROOT, ssh_args

REMOTE_ROOT = "/home/u3129/.local/opt/vasp/6.6.1-oneapi-2025"
WRITABLE_BUILD_ROOT = "/home6/u3129/.local/opt/vasp/6.6.1-oneapi-2025"
EXPECTED_SHA = "261f9c1fd730468106191afeb185821d67cb7a58c982f6e063824dd5432106b0"


def remote(script, timeout=60):
    return subprocess.run(ssh_args() + ["-o", "BatchMode=yes", "-o",
                          "PreferredAuthentications=publickey", "mlip-umt", "bash -s"],
                          input=(script + "\n").encode(), capture_output=True, timeout=timeout, check=True).stdout.decode()


def save(path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def write_remote(path, content):
    encoded = base64.b64encode(content).decode()
    remote("set -eu; umask 077; printf %s " + shlex.quote(encoded) + " | base64 -d > " + shlex.quote(path))


def stage(archive, run_id, manifest_path):
    install_root = WRITABLE_BUILD_ROOT
    digest = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
    if digest != EXPECTED_SHA:
        raise ValueError("Archive hash differs from the inspected distribution")
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            p = PurePosixPath(member.name)
            if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0] != "vasp.6.6.1":
                raise ValueError("Unsafe archive path")
            if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                raise ValueError("Unsupported archive member type")
            if member.issym() or member.islnk():
                target = posixpath.normpath(posixpath.join(posixpath.dirname(member.name), member.linkname)
                                            if member.issym() else member.linkname)
                if target != "vasp.6.6.1" and not target.startswith("vasp.6.6.1/"):
                    raise ValueError("Archive link escapes source tree")
    files = {
        "build.sbatch": ROOT / "scripts/hpc/build-vasp.sbatch",
        "overrides.mk": ROOT / "configs/vasp/oneapi-overrides.mk",
        "env.sh": ROOT / "scripts/hpc/vasp-env.sh",
    }
    payloads = {name: path.read_bytes().replace(b"\r\n", b"\n") for name, path in files.items()}
    run_dir = install_root + "/runs/" + run_id
    manifest = {
        "run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "user-requested installation and selected upstream validation tests",
        "host": "umt.imm.uran.ru", "user": "u3129", "remote_install_root": install_root,
        "remote_run_dir": run_dir, "archive_sha256": digest, "archive_size": archive.stat().st_size,
        "input_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "resources": {"partition": "apollo", "account": "imet-2026-2", "qos": "normal",
                      "nodes": 1, "cpus": 8, "memory_gib": 16, "walltime": "01:00:00"},
        "stage": "preparing", "job_id": None,
    }
    if manifest_path.exists():
        raise FileExistsError("Manifest already exists; inspect its state before continuing")
    save(manifest_path, manifest)
    remote("set -eu; umask 077; test ! -e " + shlex.quote(install_root) +
           "; mkdir -p " + shlex.quote(run_dir) + " " + shlex.quote(install_root + "/src") +
           "; chmod 700 " + shlex.quote(install_root))
    with archive.open("rb") as stream:
        subprocess.run(ssh_args() + ["-o", "BatchMode=yes", "mlip-umt",
                       "umask 077; cat > " + install_root + "/source.tgz"], stdin=stream, check=True, timeout=600)
    remote("set -eu; printf '%s  %s\\n' " + digest + " " + install_root +
           "/source.tgz | sha256sum -c -; tar --no-same-owner -xzf " + install_root +
           "/source.tgz -C " + install_root + "/src", timeout=120)
    for name, payload in payloads.items():
        target = install_root + "/env.sh" if name == "env.sh" else run_dir + "/" + name
        write_remote(target, payload)
    remote("set -eu; cd " + install_root + "/src/vasp.6.6.1; cp arch/makefile.include.oneapi makefile.include; "
           "printf '\\n' >> makefile.include; cat " + run_dir + "/overrides.mk >> makefile.include")
    manifest["stage"] = "staged"
    save(manifest_path, manifest)
    write_remote(run_dir + "/manifest.json", manifest_path.read_bytes())
    print(json.dumps(manifest, indent=2))


def submit(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["stage"] != "staged" or manifest["job_id"]:
        raise RuntimeError("Not eligible for submission; reconcile an earlier attempt first")
    manifest["stage"] = "submission_intent"
    save(path, manifest)
    write_remote(manifest["remote_run_dir"] + "/submission-intent.json", path.read_bytes())
    # No automatic resubmission after any uncertain outcome.
    result = remote("set -eu; cd " + shlex.quote(manifest["remote_run_dir"]) +
                    "; test ! -e job.id; sbatch --parsable --job-name=" +
                    shlex.quote(manifest["run_id"]) + " --chdir=/tmp" +
                    " --output=" + shlex.quote(manifest["remote_run_dir"] + "/slurm-%j.out") + " --export=" +
                    shlex.quote("ALL,VASP_BUILD_RUN_DIR=" + manifest["remote_run_dir"] +
                                ",VASP_BUILD_ROOT=" + manifest["remote_install_root"]) +
                    " build.sbatch > job.id; cat job.id")
    job = result.strip().split(";")[0]
    if not re.fullmatch(r"\d+", job):
        raise RuntimeError("Uncertain submission response; inspect remote job.id and scheduler before retrying")
    manifest.update(stage="submitted", job_id=job)
    save(path, manifest)
    print(json.dumps(manifest, indent=2))


def refresh(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["stage"] != "staged" or manifest["job_id"]:
        raise RuntimeError("Configuration updates are permitted only before submission")
    run_dir = manifest["remote_run_dir"]
    install_root = manifest["remote_install_root"]
    remote("test ! -e " + shlex.quote(run_dir + "/job.id"))
    files = {"build.sbatch": ROOT / "scripts/hpc/build-vasp.sbatch",
             "overrides.mk": ROOT / "configs/vasp/oneapi-overrides.mk",
             "env.sh": ROOT / "scripts/hpc/vasp-env.sh"}
    for name, source in files.items():
        payload = source.read_bytes().replace(b"\r\n", b"\n")
        write_remote(install_root + "/env.sh" if name == "env.sh" else run_dir + "/" + name, payload)
        manifest["input_sha256"][name] = hashlib.sha256(payload).hexdigest()
    remote("set -eu; cd " + install_root + "/src/vasp.6.6.1; cp arch/makefile.include.oneapi makefile.include; "
           "printf '\\n' >> makefile.include; cat " + run_dir + "/overrides.mk >> makefile.include")
    manifest["code_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    save(path, manifest)
    write_remote(run_dir + "/manifest.json", path.read_bytes())
    print("Staged configuration refreshed; no job submitted")


def prepare_retry(previous_path, run_id, new_path):
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    job_id = previous.get("job_id")
    if not job_id or not re.fullmatch(r"\d+", job_id):
        raise ValueError("Previous job ID is unknown; reconcile first")
    observation = remote("sacct -X -j " + job_id + " -nP --format=JobIDRaw,State,ExitCode")
    row = next((line.split("|") for line in observation.splitlines() if line.startswith(job_id + "|")), None)
    if not row or row[1] not in {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"}:
        raise RuntimeError("Previous job is not confirmed terminal/unsuccessful")
    if new_path.exists():
        raise FileExistsError("Retry manifest already exists")
    previous.update(stage="failed", scheduler_state=row[1], scheduler_exit_code=row[2])
    save(previous_path, previous)
    manifest = dict(previous)
    manifest.update(run_id=run_id, remote_run_dir=previous["remote_install_root"] + "/runs/" + run_id,
                    previous_job_id=job_id, job_id=None, stage="staged",
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    retry_reason="configuration repaired after recorded terminal failure; see input hashes and code commit")
    manifest.pop("scheduler_state", None)
    manifest.pop("scheduler_exit_code", None)
    remote("set -eu; umask 077; mkdir " + shlex.quote(manifest["remote_run_dir"]))
    save(new_path, manifest)
    refresh(new_path)


def migrate(previous_path, run_id, new_path):
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    job_id = previous["job_id"]
    if not re.fullmatch(r"\d+", job_id):
        raise ValueError("Invalid previous job")
    observation = remote("sacct -X -j " + job_id + " -nP --format=JobIDRaw,State,ExitCode")
    row = next((line.split("|") for line in observation.splitlines() if line.startswith(job_id + "|")), None)
    if not row or row[1] != "FAILED":
        raise RuntimeError("Confirm that the previous job failed before migration")
    if new_path.exists():
        raise FileExistsError("Migration manifest exists; reconcile first")
    previous.update(stage="failed", scheduler_state=row[1], scheduler_exit_code=row[2],
                    failure_reason="home4 is read-only from compute nodes; confirmed by Slurm diagnostic 20950118")
    save(previous_path, previous)
    manifest = dict(previous)
    manifest.update(run_id=run_id, remote_install_root=WRITABLE_BUILD_ROOT,
                    remote_run_dir=WRITABLE_BUILD_ROOT + "/runs/" + run_id,
                    previous_job_id=job_id, job_id=None, stage="preparing",
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    retry_reason="build and validate on compute-writable home6, then deploy binaries to read-only home4 from login node")
    manifest.pop("scheduler_state", None)
    manifest.pop("scheduler_exit_code", None)
    manifest.pop("failure_reason", None)
    save(new_path, manifest)
    remote("set -eu; umask 077; test ! -e " + WRITABLE_BUILD_ROOT +
           "; mkdir -p " + manifest["remote_run_dir"] + " " + WRITABLE_BUILD_ROOT +
           "/src; chmod 700 " + WRITABLE_BUILD_ROOT + "; cp " + REMOTE_ROOT +
           "/source.tgz " + WRITABLE_BUILD_ROOT + "/source.tgz; printf '%s  %s\\n' " + EXPECTED_SHA +
           " " + WRITABLE_BUILD_ROOT + "/source.tgz | sha256sum -c -; tar --no-same-owner -xzf " +
           WRITABLE_BUILD_ROOT + "/source.tgz -C " + WRITABLE_BUILD_ROOT + "/src", timeout=180)
    manifest["stage"] = "staged"
    save(new_path, manifest)
    refresh(new_path)


def finalize(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    job_id, run_dir, build_root = manifest["job_id"], manifest["remote_run_dir"], manifest["remote_install_root"]
    if not re.fullmatch(r"\d+", job_id):
        raise ValueError("Invalid job ID")
    accounting = remote("sacct -X -j " + job_id + " -nP --format=JobIDRaw,State,ExitCode,Elapsed,AllocCPUS,NodeList")
    row = next((line.split("|") for line in accounting.splitlines() if line.startswith(job_id + "|")), None)
    if not row or row[1:3] != ["COMPLETED", "0:0"]:
        raise RuntimeError("Build job is not successfully completed: " + accounting.strip())
    status_text = remote("cat " + shlex.quote(run_dir + "/status.txt"))
    if "phase=complete" not in status_text or "exit_code=0" not in status_text:
        raise RuntimeError("Build/validation completion marker missing")
    log = remote("cat " + shlex.quote(run_dir + "/testsuite.log"))
    if "SUCCESS: ALL SELECTED TESTS PASSED" not in log:
        raise RuntimeError("Upstream testsuite success summary missing")
    expected = {"bulk_AlPAW", "H2Ostretch_efor", "Fe_r2SCAN_ncl_soc", "bulk_BN_vdW-DF3-opt1"}
    cases = set(re.findall(r"CASE:\s*(\S+)", log))
    if cases != expected:
        raise RuntimeError("Unexpected test coverage; inspect testsuite log before deployment")
    binary_hashes = remote("cat " + shlex.quote(run_dir + "/binaries.sha256"))
    hashes = {PurePosixPath(line.split()[1]).name: line.split()[0] for line in binary_hashes.splitlines()}
    libraries = remote("cat " + run_dir + "/ldd-std.txt " + run_dir + "/ldd-gam.txt " + run_dir + "/ldd-ncl.txt")
    if "not found" in libraries:
        raise RuntimeError("Unresolved runtime dependency")
    deploy_script = """set -eu
umask 077
build=%s
dest=%s
mkdir -p "$dest/src/vasp.6.6.1/bin"
chmod 700 "$dest"
if [ ! -e "$dest/source.tgz" ]; then
    install -m 600 "$build/source.tgz" "$dest/source.tgz"
else
    cmp "$build/source.tgz" "$dest/source.tgz"
fi
for name in vasp_std vasp_gam vasp_ncl; do
    if [ -e "$dest/src/vasp.6.6.1/bin/$name" ]; then
        cmp "$build/src/vasp.6.6.1/bin/$name" "$dest/src/vasp.6.6.1/bin/$name"
    fi
done
for name in vasp_std vasp_gam vasp_ncl; do
    if [ ! -e "$dest/src/vasp.6.6.1/bin/$name" ]; then
        install -m 700 "$build/src/vasp.6.6.1/bin/$name" "$dest/src/vasp.6.6.1/bin/$name"
    fi
done
install -m 600 "$build/env.sh" "$dest/env.sh"
cd "$dest/src/vasp.6.6.1"
sha256sum bin/vasp_std bin/vasp_gam bin/vasp_ncl
""" % (shlex.quote(build_root), shlex.quote(REMOTE_ROOT))
    deployed = remote(deploy_script)
    deployed_hashes = {PurePosixPath(line.split()[1]).name: line.split()[0] for line in deployed.splitlines()}
    if deployed_hashes != hashes:
        raise RuntimeError("Deployed binary checksums differ")
    result = {
        "version": "6.6.1", "job_id": job_id, "scheduler_state": row[1], "exit_code": row[2],
        "elapsed": row[3], "allocated_cpus": row[4], "node": row[5],
        "verified_utc": datetime.now(timezone.utc).isoformat(), "install_root": REMOTE_ROOT,
        "build_root": build_root, "variants": sorted(hashes), "binary_sha256": hashes,
        "passed_upstream_cases": sorted(cases), "validation_scope": "four selected upstream cases, not full suite",
        "mode": "CPU MPI; SSE4.2 portable target; MKL sequential; no GPU or HDF5/optional external interfaces",
        "toolchain": "Intel ifx 2025.3.2, Intel MPI 2021.17, MKL 2025.3",
    }
    save(path.parent / "result.json", result)
    (path.parent / "validation-summary.txt").write_text(log, encoding="utf-8")
    (path.parent / "runtime-libraries.txt").write_text(libraries, encoding="utf-8")
    manifest.update(stage="complete", result="result.json", deployed_root=REMOTE_ROOT)
    save(path, manifest)
    write_remote(REMOTE_ROOT + "/installation.json", (path.parent / "result.json").read_bytes())
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("stage")
    p.add_argument("archive", type=Path)
    p.add_argument("--run-id", required=True)
    p.add_argument("--manifest", required=True, type=Path)
    p = sub.add_parser("submit")
    p.add_argument("manifest", type=Path)
    p = sub.add_parser("refresh")
    p.add_argument("manifest", type=Path)
    p = sub.add_parser("prepare-retry")
    p.add_argument("previous", type=Path)
    p.add_argument("--run-id", required=True)
    p.add_argument("--manifest", required=True, type=Path)
    p = sub.add_parser("migrate")
    p.add_argument("previous", type=Path)
    p.add_argument("--run-id", required=True)
    p.add_argument("--manifest", required=True, type=Path)
    p = sub.add_parser("finalize")
    p.add_argument("manifest", type=Path)
    args = parser.parse_args()
    if args.action == "stage":
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run_id):
            parser.error("Unsafe run ID")
        stage(args.archive, args.run_id, args.manifest)
    elif args.action == "submit":
        submit(args.manifest)
    elif args.action == "refresh":
        refresh(args.manifest)
    elif args.action == "finalize":
        finalize(args.manifest)
    else:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run_id):
            parser.error("Unsafe run ID")
        if args.action == "migrate":
            migrate(args.previous, args.run_id, args.manifest)
        else:
            prepare_retry(args.previous, args.run_id, args.manifest)


if __name__ == "__main__":
    main()
