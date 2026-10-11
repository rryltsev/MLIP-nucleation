"""Offline consistency check; does not contact the cluster or submit jobs."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
cluster = json.loads((root / "configs/cluster.json").read_text(encoding="utf-8"))
state = json.loads((root / "project/state.json").read_text(encoding="utf-8"))
assert cluster["scheduler"] == "slurm"
assert cluster["dft_engine"] == "vasp"
assert 0 < cluster["port"] < 65536
assert (root / "project/brief.txt").is_file(), "Missing original research brief"
assert state["stage"]
ssh = (root / "configs/ssh_config").read_text(encoding="utf-8")
assert "StrictHostKeyChecking yes" in ssh
assert f'HostName {cluster["host"]}' in ssh
assert f'User {cluster["user"]}' in ssh
assert f'Port {cluster["port"]}' in ssh
missing = [key for key, value in cluster.items() if value is None]
print("Offline configuration consistency: OK")
print("Unresolved cluster fields: " + ", ".join(missing))
print("No SSH authentication, cluster software or scientific workflow was tested.")
