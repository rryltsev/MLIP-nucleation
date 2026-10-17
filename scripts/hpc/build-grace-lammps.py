"""Build a private pinned serial LAMMPS with the TensorFlow GRACE pair style."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import traceback
import urllib.request

variant=os.environ.get('LMP_BUILD_VARIANT','grace')
REVISION = 'b1f4a6c8cc4e0d0fc93d2dec3a1fcc83558062d4' if variant=='grace' else 'a33449868448baf3c73d8eacdb2d329b13361696'
root = Path.cwd()
prefix = Path('/home6/u3129/mlip-research/software/lammps-'+variant+'-'+REVISION[:12])
env = Path('/home6/u3129/mlip-research/envs/grace_ft_a100')
result = {'status':'building','upstream_git_commit':REVISION,'prefix':str(prefix),
          'job_id':os.environ['SLURM_JOB_ID'],'started_utc':datetime.now(timezone.utc).isoformat()}
try:
    prefix.mkdir(parents=True,exist_ok=False)
    archive = prefix/'source.tar.gz'
    repository='yury-lysogorskiy/lammps' if variant=='grace' else 'lammps/lammps'
    url = 'https://codeload.github.com/'+repository+'/tar.gz/'+REVISION
    with urllib.request.urlopen(url,timeout=90) as inp, archive.open('wb') as out:
        while b:=inp.read(1048576): out.write(b)
    result.update(source_url=url,source_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    with tarfile.open(archive) as bundle:
        bundle.extractall(prefix,filter='data')
    source = prefix/('lammps-'+REVISION)
    tf = env/'lib/python3.11/site-packages/tensorflow'
    command = ['cmake','-S',str(source/'cmake'),'-B',str(prefix/'build'),
        '-DCMAKE_BUILD_TYPE=Release','-DBUILD_MPI=OFF','-DBUILD_OMP=ON','-DBUILD_SHARED_LIBS=ON',
        '-DPKG_ML-PACE=ON','-DPKG_PLUGIN=ON','-DDOWNLOAD_POTENTIALS=OFF',
        '-DTF_LIB_FILE='+str(tf/'libtensorflow_cc.so.2'),'-DTF_INCLUDE_PATH='+str(tf/'include'),
        '-DTF_PATH='+str(tf),'-DPACE_PYTHON_EXEC='+str(env/'bin/python'),
        '-DCMAKE_INSTALL_PREFIX='+str(prefix/'install')]
    if variant=='deepmd':
        command=command[:command.index('-DPKG_ML-PACE=ON')]+['-DPKG_PLUGIN=ON','-DDOWNLOAD_POTENTIALS=OFF',
            '-DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-12/root/usr/bin/g++','-DCMAKE_C_COMPILER=/opt/rh/gcc-toolset-12/root/usr/bin/gcc']
    result['configure_command']=command
    for name, cmd, seconds in [('configure',command,600),
        ('build',['cmake','--build',str(prefix/'build'),'--parallel','8'],1800)]:
        with (root/(name+'.log')).open('wb') as out:
            subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=seconds)
    binary = prefix/'build/lmp'
    with (root/'lammps-help.txt').open('wb') as out:
        subprocess.run([str(binary),'-h'],stdout=out,stderr=subprocess.STDOUT,check=True,timeout=40)
    assert ('grace' if variant=='grace' else 'PLUGIN') in (root/'lammps-help.txt').read_text()
    result.update(status='passed',binary=str(binary),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    result['shared_library_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (prefix/'build').glob('liblammps.so*') if p.is_file() and not p.is_symlink()}
except Exception:
    result.update(status='failed',error=traceback.format_exc())
result['finished_utc']=datetime.now(timezone.utc).isoformat()
(root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
raise SystemExit(0 if result['status']=='passed' else 1)
