"""Consume the already validated DPA4 export; no training or export performed."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

tools=Path(__file__).parent;root=Path.cwd()
spec=importlib.util.spec_from_file_location('deployment',tools/'fm-deployment.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
record={'status':'running','backend':'dpa4','job_id':os.environ['SLURM_JOB_ID'],
        'scope':'LAMMPS consume-only test; no training or export','mpi_root':os.environ.get('I_MPI_ROOT')}
try:
    from ase.io import read
    source=json.loads((tools/'reuse-export.json').read_text());old=Path(source['remote_run_dir'])/'dpa4'
    for name,want in source['files'].items():assert helper.digest(old/name)==want,name
    previous=json.loads((old/'result.json').read_text())
    assert all(r['passed'] for r in previous['ase_equivalence'].values())
    model=old/'exported.pt2';structures=read(old/'comparison.extxyz',index=':')
    reference=json.loads((old/'checkpoint-predictions.json').read_text())
    package=Path(sys.prefix)/'lib/python3.12/site-packages'
    build=json.loads((tools/'deepmd-lammps-build.json').read_text())
    binary=Path(build['binary']);assert helper.digest(binary)==build['binary_sha256']
    plugin=package/'deepmd/lib/libdeepmd_lmpplugin.so'
    prediction=helper.lammps_predictions(binary,model,structures,plugin)
    helper.save(root/'lammps-export-predictions.json',prediction)
    record['lammps_equivalence']=helper.compare_predictions(reference,prediction,'lammps-equivalence')
    record.update(status='passed',consumed_source=source,lammps_binary_sha256=helper.digest(binary),plugin_sha256=helper.digest(plugin))
except Exception:record.update(status='failed',error=traceback.format_exc())
helper.save(root/'result.json',record);print(json.dumps(record,indent=2),flush=True)
sys.exit(0 if record['status']=='passed' else 1)
