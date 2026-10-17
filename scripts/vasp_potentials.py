"""Inspect/install licensed PAW archives and assemble traceable POTCARs.

Python 3.6 compatible for the cluster. Only metadata may enter Git.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import zipfile

FAMILIES = {
    "PBE": "Pseudopotentials_v6.4/potpaw_PBE.64.tgz",
    "LDA": "Pseudopotentials_v6.4/potpaw_LDA.64.tgz",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def safe_name(name):
    p = PurePosixPath(name)
    if not p.parts or p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name:
        raise ValueError("Unsafe archive member: " + name)
    return str(p)


def inspect(archive, install_root=None):
    result = {
        "schema_version": 1,
        "library_id": "vasp-paw-64-" + file_sha(archive)[:12],
        "archive_name": archive.name,
        "archive_sha256": file_sha(archive),
        "archive_bytes": archive.stat().st_size,
        "families": {},
    }
    with zipfile.ZipFile(str(archive)) as z:
        if len(z.namelist()) != len(set(z.namelist())):
            raise ValueError("Duplicate ZIP entries")
        for entry in z.infolist():
            safe_name(entry.filename)
        if {n for n in z.namelist() if not n.endswith("/")} != set(FAMILIES.values()):
            raise ValueError("Unexpected distribution layout")
        for family, name in sorted(FAMILIES.items()):
            data = z.read(name)  # ZIP CRC verified by zipfile.
            records, potentials, names = [], {}, set()
            target = install_root / ("potpaw_" + family) if install_root else None
            if target:
                target.mkdir(mode=0o700)
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
                for member in t:
                    rel = safe_name(member.name)
                    if rel in names or not (member.isfile() or member.isdir()):
                        raise ValueError("Duplicate or unsupported TAR member: " + rel)
                    names.add(rel)
                    if member.isdir():
                        if target:
                            (target / rel).mkdir(mode=0o700, parents=True, exist_ok=True)
                        continue
                    payload = t.extractfile(member).read()
                    digest = sha(payload)
                    records.append([rel, digest])
                    if target:
                        dest = target / rel
                        dest.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                        with dest.open("xb") as stream:
                            stream.write(payload)
                        dest.chmod(0o600)
                    if PurePosixPath(rel).name == "POTCAR":
                        if len(PurePosixPath(rel).parts) != 2:
                            raise ValueError("Unexpected POTCAR layout")
                        text = payload.decode("ascii", errors="replace")
                        title = re.search(r"TITEL\s*=\s*([^\r\n]+)", text)
                        lexch = re.search(r"LEXCH\s*=\s*(\S+)", text)
                        if not title or not lexch:
                            raise ValueError("Missing POTCAR identification: " + rel)
                        potentials[PurePosixPath(rel).parts[0]] = {
                            "path": rel, "sha256": digest, "bytes": len(payload),
                            "titel": title.group(1).strip(), "lexch": lexch.group(1),
                        }
            result["families"][family] = {
                "directory": "potpaw_" + family,
                "archive_member": name, "archive_sha256": sha(data),
                "file_count": len(records),
                "files_sha256": sha(json.dumps(sorted(records), separators=(",", ":")).encode()),
                "potential_count": len(potentials), "potentials": potentials,
            }
    return result


def verify(registry_path):
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    root = Path(registry["remote_root"])
    for family in registry["families"].values():
        base = root / family["directory"]
        records = []
        for path in sorted(base.rglob("*")):
            if path.is_symlink():
                raise ValueError("Unexpected symlink")
            if path.is_file():
                records.append([path.relative_to(base).as_posix(), file_sha(path)])
        digest = sha(json.dumps(sorted(records), separators=(",", ":")).encode())
        if len(records) != family["file_count"] or digest != family["files_sha256"]:
            raise ValueError("Installed file inventory differs")
    print("All extracted files match the registry")


def assemble(registry_path, family, variants, output):
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    info = registry["families"][family]
    root = Path(registry["remote_root"]) / info["directory"]
    provenance_path = output.parent / "potcar-provenance.json"
    if output.exists() or provenance_path.exists():
        raise FileExistsError("Refusing to overwrite POTCAR or provenance")
    parts, sources = [], []
    for variant in variants:
        entry = info["potentials"][variant]  # Explicit variant; no fallback.
        payload = (root / entry["path"]).read_bytes()
        if sha(payload) != entry["sha256"]:
            raise ValueError("Potential hash mismatch: " + variant)
        if not payload.endswith(b"\n"):
            raise ValueError("Potential lacks terminal newline")
        parts.append(payload)
        sources.append(dict(entry, variant=variant))
    payload = b"".join(parts)
    provenance = {
        "library_id": registry["library_id"], "archive_sha256": registry["archive_sha256"],
        "registry_sha256": file_sha(registry_path), "family": family,
        "ordered_potentials": sources, "potcar_sha256": sha(payload),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "assembly_script_sha256": file_sha(Path(__file__)),
    }
    with output.open("xb") as stream:
        stream.write(payload)
    output.chmod(0o600)
    save(provenance_path, provenance)
    print(json.dumps({"potcar_sha256": sha(payload), "family": family, "variants": variants}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    scan = commands.add_parser("inspect")
    scan.add_argument("archive", type=Path)
    scan.add_argument("--output", type=Path, required=True)
    unpack = commands.add_parser("unpack")
    unpack.add_argument("archive", type=Path)
    unpack.add_argument("--root", type=Path, required=True)
    unpack.add_argument("--expected-sha256", required=True)
    check = commands.add_parser("verify")
    check.add_argument("registry", type=Path)
    build = commands.add_parser("assemble")
    build.add_argument("registry", type=Path)
    build.add_argument("--family", choices=sorted(FAMILIES), required=True)
    build.add_argument("--variants", nargs="+", required=True,
                       help="Exact potential names in POSCAR species order")
    build.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inspect":
        save(args.output, inspect(args.archive))
    elif args.command == "unpack":
        if file_sha(args.archive) != args.expected_sha256:
            raise ValueError("Distribution SHA-256 mismatch")
        # Validate the entire archive before writing extracted files.
        expected = inspect(args.archive)
        result = inspect(args.archive, args.root)
        if result != expected:
            raise ValueError("Archive changed during extraction")
        result.update(remote_root=str(args.root), installed_utc=datetime.now(timezone.utc).isoformat())
        save(args.root / "registry.json", result)
        verify(args.root / "registry.json")
    elif args.command == "verify":
        verify(args.registry)
    elif args.command == "assemble":
        assemble(args.registry, args.family, args.variants, args.output)
    else:
        parser.error("Choose a command")


if __name__ == "__main__":
    main()
