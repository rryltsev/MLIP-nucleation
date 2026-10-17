"""Build a hash-locked pip wheel list from the collected GRACE installation report."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

from cluster_ssh import ROOT
from vasp_potentials import save


def main(run):
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    if result["status"] != "installed":
        raise ValueError("Installation did not pass")
    installed = {}
    normalize = lambda s:s.lower().replace('_','-').replace('.','-')
    for line in (run / 'pip-freeze.txt').read_text().splitlines():
        if '==' in line:
            name, version = line.split('==',1); installed[normalize(name)] = version
    # A timed-out install may already have installed some wheels. Preserve the
    # earlier report, then select only artifacts matching the final versions.
    reports = sorted(run.parent.glob(run.name.rsplit('-',1)[0]+'-*/pip-install-report.json'))
    selected = {}
    for path in reports:
        for item in json.loads(path.read_text(encoding='utf-8'))['install']:
            name = normalize(item['metadata']['name'])
            if installed.get(name) == item['metadata']['version']:
                selected[name] = item
    artifacts = []
    for item in selected.values():
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
        "source_reports":{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in reports},
        "conda_lock":conda.relative_to(ROOT).as_posix(),"pip_lock":dest.relative_to(ROOT).as_posix(),
        "file_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [conda,dest]},
        "wheel_artifacts":artifacts,
    })
    print("Locked %d pip wheels and the explicit Conda base" % len(artifacts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("installation_run", type=Path)
    main(parser.parse_args().installation_run.resolve())
