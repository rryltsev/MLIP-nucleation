"""Machine-local SSH key setup and password-free cluster commands (stdlib only)."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
KEY = Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".ssh" / "id_ed25519_mlip_umt"
STATUS = ROOT / "artifacts" / "ssh-setup.local.json"


def status(phase, **extra):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({"phase": phase, "utc": datetime.now(timezone.utc).isoformat(),
                                  "key_path": str(KEY), **extra}, indent=2), encoding="utf-8")


def ssh_args():
    return ["ssh", "-F", str(ROOT / "configs" / "ssh_config"),
            "-o", 'UserKnownHostsFile="' + (ROOT / "configs" / "known_hosts").as_posix() + '"',
            "-o", "IdentitiesOnly=yes", "-i", str(KEY)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup-key", action="store_true", help="Interactive one-time password login")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Remote command, defaults to hostname")
    args = parser.parse_args()
    if args.setup_key:
        status("preparing")
        KEY.parent.mkdir(parents=True, exist_ok=True)
        public_path = Path(str(KEY) + ".pub")
        if not KEY.exists():
            if public_path.exists():
                raise RuntimeError("Public key exists without private key; refusing to overwrite")
            subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(KEY),
                            "-C", "mlip-umt-" + os.environ.get("COMPUTERNAME", "local")], check=True)
        # Derive the public key to ensure that an existing .pub file is not mismatched.
        public_key = subprocess.run(["ssh-keygen", "-y", "-P", "", "-f", str(KEY)],
                                    check=True, capture_output=True, text=True).stdout.strip()
        if not public_key.startswith("ssh-ed25519 ") or "\n" in public_key:
            raise RuntimeError("Expected a single ED25519 public key")
        fingerprint = subprocess.run(["ssh-keygen", "-lf", str(public_path)],
                                     check=True, capture_output=True, text=True).stdout.strip()
        # Preserve all pre-existing authorized keys. No password is read by Python.
        remote = (
            "set -eu; umask 077; mkdir -p ~/.ssh; chmod 700 ~/.ssh; "
            "touch ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys; "
            "key=" + shlex.quote(public_key) + "; "
            "if ! grep -qxF -- \"$key\" ~/.ssh/authorized_keys; then "
            "printf '\\n%s\\n' \"$key\" >> ~/.ssh/authorized_keys; fi; "
            "printf 'Public key installed.\\n'"
        )
        print("Enter your CLUSTER password at the SSH prompt. Input is hidden.", flush=True)
        status("awaiting_password", fingerprint=fingerprint)
        subprocess.run(ssh_args() + ["-o", "PubkeyAuthentication=no", "-o",
                       "PreferredAuthentications=password,keyboard-interactive", "-o",
                       "NumberOfPasswordPrompts=1", "mlip-umt", remote], check=True)
        result = subprocess.run(ssh_args() + ["-o", "BatchMode=yes", "-o",
                                "PreferredAuthentications=publickey", "mlip-umt", "hostname"],
                                check=True, capture_output=True, text=True)
        status("verified", fingerprint=fingerprint, remote_hostname=result.stdout.strip())
        print("Password-free SSH verified: " + result.stdout.strip(), flush=True)
        return 0
    if not KEY.exists():
        parser.error("Local key is missing. Run --setup-key in an interactive terminal first.")
    command = args.command or ["hostname"]
    if command and command[0] == "--":
        command = command[1:]
    return subprocess.run(ssh_args() + ["-o", "BatchMode=yes", "mlip-umt", *command]).returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (subprocess.CalledProcessError, RuntimeError, OSError) as exc:
        status("failed", error=str(exc))
        print("SSH setup failed: " + str(exc), file=sys.stderr)
        sys.exit(1)
