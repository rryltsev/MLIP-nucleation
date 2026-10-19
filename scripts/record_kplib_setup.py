"""Verify archived kpLib binaries and record the installed, isolated environment."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    setup=ROOT/'runs/al32-kplib-setup-20261019-01'
    grids=ROOT/'runs/al32-kplib-grids-20261019-01'
    software=json.loads((setup/'software.json').read_text())
    runtime=json.loads((grids/'runtime.json').read_text())
    backup=json.loads((setup/'artifact-backup.json').read_text())
    archive=ROOT/'artifacts/dft-run-backups'/Path(backup['remote_archive']).name
    assert sha(archive.read_bytes())==backup['sha256']
    with tarfile.open(archive) as tar:
        wheel_data={p:tar.extractfile('wheelhouse/'+p).read() for p in software['wheel_files']}
    for name,data in wheel_data.items():assert sha(data)==software['wheel_files'][name]['sha256']
    built=next(name for name in wheel_data if name.lower().startswith('kplib-'))
    with zipfile.ZipFile(io.BytesIO(wheel_data[built])) as wheel:
        extension=next(n for n in wheel.namelist() if n.startswith('kpLib/lib.') and n.endswith('.so'))
        assert sha(wheel.read(extension))==runtime['extension_sha256']
    lock=dict(environment=software['environment'],activation=software['environment']+'/activate-kplib.sh',
        python=software['python'],base_interpreter=software['base_interpreter'],compiler=software['compiler'],
        installed_packages=software['installed_packages'],source_sha256=software['source_sha256'],
        wheel_files=software['wheel_files'],extension_sha256=runtime['extension_sha256'],
        setup_manifest=str(setup.relative_to(ROOT)/'manifest.json').replace('\\','/'),
        grid_manifest=str(grids.relative_to(ROOT)/'manifest.json').replace('\\','/'),
        wheel_backup=str(setup.relative_to(ROOT)/'artifact-backup.json').replace('\\','/'),
        wheel_archive_sha256=backup['sha256'],
        rebuild='On compatible Linux x86_64/Python3.12 create a fresh venv, verify every wheel hash from the private archive, then pip install --no-index --no-deps wheelhouse/*.whl; pip check. Preserve existing environments.',
        inventory_note='The run requirements.lock.txt is a wheel filename/checksum inventory, not a pip -r requirements file.')
    target=ROOT/'environments/kplib-linux-64.lock.json'
    target.write_text(json.dumps(lock,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(dict(verified_wheels=len(wheel_data),extension_sha256=runtime['extension_sha256'],lock=str(target))))


if __name__=='__main__':main()
