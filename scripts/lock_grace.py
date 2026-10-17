"""Build a hash-locked pip wheel list from the collected GRACE installation report."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

from cluster_ssh import ROOT
from vasp_potentials import save


def main(run):
    report = json.loads((run / "pip-install-report.json").read_text(encoding="utf-8"))
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    if result["status"] != "installed":
        raise ValueError("Installation did not pass")
    artifacts = []
    for item in report["install"]:
        info = item["download_info"]
        url = info["url"]
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.username or parsed.password or not parsed.path.endswith(".whl"):
            raise ValueError("Expected public HTTPS wheel without embedded credentials")
        digest = info["archive_info"]["hashes"]["sha256"]
        if len(digest) != 64:
            raise ValueError("Invalid package SHA-256")
        artifacts.append({"name":item["metadata"]["name"],"version":item["metadata"]["version"],"url":url,"sha256":digest})
    artifacts.sort(key=lambda a:a["name"].lower())
    dest = ROOT / "environments/grace-a100-pip-linux-64.lock.txt"
    text = "# Exact wheels from the successful installation; requires the recorded Conda base.\n"
    text += "# Restore with: python -m pip install --no-deps --require-hashes -r THIS_FILE\n"
    text += "\n".join(a["name"] + " @ " + a["url"] + " --hash=sha256:" + a["sha256"] for a in artifacts) + "\n"
    dest.write_bytes(text.encode())
    conda = ROOT / "environments/grace-a100-conda-linux-64.lock.txt"
    conda.write_bytes((run / "conda-explicit-sha256.txt").read_bytes())
    save(ROOT / "environments/grace-a100.lock.json",{
        "scope":"Linux x86_64, observed Python 3.11; Conda base plus exact pip wheels",
        "installation_manifest":(run / "manifest.json").relative_to(ROOT).as_posix(),
        "versions":result["versions"],"python":result["python"],
        "conda_lock":conda.relative_to(ROOT).as_posix(),"pip_lock":dest.relative_to(ROOT).as_posix(),
        "file_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [conda,dest]},
        "wheel_artifacts":artifacts,
    })
    print("Locked %d pip wheels and the explicit Conda base" % len(artifacts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("installation_run", type=Path)
    main(parser.parse_args().installation_run.resolve())
