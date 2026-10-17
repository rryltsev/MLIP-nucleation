"""Fetch pinned public training weights; retain actual local and cluster copies."""
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import urllib.request

from cluster_ssh import ROOT, ssh_args
from install_vasp import remote
from vasp_potentials import save

SPECS = {
    "dpa4-mini-omat24-20260805": {
        "repository": "deepmodelingcommunity/DPA4-OMat24",
        "revision": "173bfd5a26f90a756ee891777270ef117078d9be",
        "filename": "DPA4-Mini-OMat24-v20260805.pt",
        "sha256": "13edeea63448c0f8a0b38e08fd8ba196fb1e2fad5f12486b1cd23320829da268",
        "license": "CC-BY-NC-4.0", "pretraining": "OMat24",
        "configuration": "DPA4-Mini-OMat24-v20260805.json",
        "license_url": "https://huggingface.co/deepmodelingcommunity/DPA4-OMat24/blob/173bfd5a26f90a756ee891777270ef117078d9be/README.md",
    },
    "grace-1l-omat": {
        "repository": "AMS-ICAMS-RUB/grace-foundation-models",
        "revision": "46ead61dc8a7f90c11b1e59b8489e8e9bf97b888",
        "filename": "checkpoints/GRACE-1L-OMAT-checkpoint.tar.gz",
        "sha256": "ad76b7845df6bc8da5d9cea823f5b4ccca87f5efa58904644dd43fb923ed52ff",
        "license": "Academic Software License v1.0", "pretraining": "OMat24; no OAM fine-tuning",
        "license_url": "https://github.com/ICAMS/grace-tensorpotential/blob/39fa4a7489c7bf7c91ea187cc91b7713f97fe149/LICENSE.md",
    },
}


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fetch(url, path, expected=None):
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=90) as source, temporary.open("wb") as out:
            while chunk := source.read(1024 * 1024):
                out.write(chunk)
        if expected and digest(temporary) != expected:
            raise ValueError("Downloaded hash mismatch: " + url)
        temporary.replace(path)
    if expected and digest(path) != expected:
        raise ValueError("Cached hash mismatch: " + str(path))


def main():
    for name, spec in SPECS.items():
        local = ROOT / "artifacts/foundation-checkpoints" / name
        local.mkdir(parents=True, exist_ok=True)
        base = f"https://huggingface.co/{spec['repository']}/resolve/{spec['revision']}/"
        archive = local / Path(spec["filename"]).name
        fetch(base + spec["filename"], archive, spec["sha256"])
        card = local / "README.upstream.md"
        fetch(base + "README.md", card)
        payload = [archive, card]
        if "configuration" in spec:
            p = local / spec["configuration"]; fetch(base + spec["configuration"], p); payload.append(p)
        else:
            extracted = local / "extracted"
            extracted.mkdir(exist_ok=True)
            with tarfile.open(archive) as bundle:
                for member in bundle.getmembers():
                    target = (extracted / member.name).resolve()
                    if not target.is_relative_to(extracted.resolve()) or not (member.isfile() or member.isdir()):
                        raise ValueError("Unsafe checkpoint archive entry: " + member.name)
                    if member.isfile():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(bundle.extractfile(member).read())
            models = list(extracted.rglob("model.yaml"))
            if len(models) != 1: raise ValueError("Ambiguous GRACE model definition")
            payload.extend(p for p in models[0].parent.rglob("*") if p.is_file())
        target_root = "/home/u3129/.local/share/mlip-research/foundation-checkpoints/" + name + "-" + spec["sha256"][:12]
        records = {}
        for p in payload:
            rel = p.relative_to(local).as_posix()
            target = target_root + "/" + rel
            expected = digest(p)
            probe = remote("if test -f " + shlex.quote(target) + "; then sha256sum " + shlex.quote(target) + "; fi").strip()
            if probe:
                if probe.split()[0] != expected: raise ValueError("Existing remote artifact differs: " + target)
            else:
                remote("umask 077; mkdir -p " + shlex.quote(target.rsplit("/", 1)[0]))
                with p.open("rb") as stream:
                    subprocess.run(ssh_args() + ["-o", "BatchMode=yes", "mlip-umt",
                        "umask 077; set -C; cat > " + shlex.quote(target)], stdin=stream, check=True, timeout=180)
                actual = remote("sha256sum " + shlex.quote(target)).split()[0]
                if actual != expected: raise ValueError("Remote transfer hash mismatch: " + target)
            records[rel] = {"sha256": expected, "bytes": p.stat().st_size, "remote_path": target}
        if "configuration" in spec:
            elements = json.loads((local / spec["configuration"]).read_text())["model"]["type_map"]
        else:
            block = models[0].read_text().split("  element_map:\n",1)[1]
            elements = re.findall(r"^    ([A-Z][a-z]?): \d+$", block, re.M)
            elements = list(dict.fromkeys(elements))
        record = dict(spec, model_id=name, source_url=base + spec["filename"],
            chemical_elements=elements, chemical_coverage_note="Parameter/type-map coverage only; not proof of accuracy for every element or phase",
            cutoff_angstrom=6.0,
            remote_root=target_root, files=records, purpose="foundation initialization for software fine-tuning check; not selected scientific teacher",
            storage={"primary":"private cluster main home (readable on compute)",
                     "second_copy":str(local),"backup_verified":"all uploaded bytes matched local SHA-256",
                     "limitations":"No institutional retention guarantee; local second copy is machine-local, not a shared archive. Source metadata and runs are in Git."},
            reference_settings="OMat24 reference; exact PAW/DFT/electronic and energy-reference compatibility must be audited before real DFT fine-tuning",
            overlap_audit="no scientific held-out DFT set exists; synthetic diagnostic labels only")
        save(ROOT / "models/foundation" / (name + ".json"), record)
        print(json.dumps({"model":name,"remote_root":target_root,"files":len(records)}), flush=True)


if __name__ == "__main__": main()
