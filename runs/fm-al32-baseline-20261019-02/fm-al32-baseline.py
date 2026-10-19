"""Read-only original foundation inference on frozen Al32 geometries."""
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from ase.io import read

ROOT=Path(__file__).resolve().parent
BACKEND=sys.argv[1]
WORK=ROOT/BACKEND


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')


def gpu_sample():
    result=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    return [r for r in result.splitlines() if r.split(',')[0].strip()==str(os.getpid())]


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm compute node required')
    result=dict(status='running',backend=BACKEND,job_id=os.environ['SLURM_JOB_ID'],host=os.uname().nodename,
        code_commit=os.environ['BASELINE_CODE_COMMIT'],started_utc=datetime.now(timezone.utc).isoformat(),
        scope='Original checkpoint inference only; no fitting or automatic energy-zero adjustment',python=sys.version,environment=sys.prefix)
    try:
        cfg=json.loads((ROOT/'evaluation.json').read_text());use_gpu=cfg['resources']['gpus']>0
        card=json.loads((ROOT/(BACKEND+'-checkpoint.json')).read_text())
        manifest=json.loads((ROOT/'run-inputs.json').read_text())
        for name,h in manifest.items():assert sha(ROOT/name)==h,name
        # Every model source file is pinned, including source model definition.
        for name,item in card['files'].items():assert sha(item['remote_path'])==item['sha256'],name
        result.update(model_id=card['model_id'],model_sha256=card['sha256'],source_file_sha256={n:x['sha256'] for n,x in card['files'].items()},input_sha256=manifest)
        rows=json.loads((ROOT/'dataset-manifest.json').read_text())['configurations']
        assert len(rows)==40 and len({r['configuration_id'] for r in rows})==40
        if BACKEND=='dpa4':
            import torch
            from deepmd.infer import DeepPot
            assert torch.cuda.is_available() if use_gpu else not torch.cuda.is_available()
            engine=DeepPot(str(Path(card['remote_root'])/card['filename']))
            types=engine.get_type_map()
            def evaluate(atoms):
                energy,force,virial=engine.eval(atoms.positions.reshape(1,-1),atoms.cell.array.reshape(1,9),[types.index(s) for s in atoms.get_chemical_symbols()])
                stress=-virial.reshape(3,3)/atoms.get_volume()
                return float(energy.ravel()[0]),force.reshape(-1,3),stress.ravel()[[0,4,8,5,2,1]]
            result['versions']={p:importlib.metadata.version(p) for p in ('deepmd-kit','torch','numpy','ase')}
            result['device']=torch.cuda.get_device_name(0) if use_gpu else 'CPU, CUDA_VISIBLE_DEVICES empty'
        else:
            import tensorflow as tf
            from tensorpotential import TensorPotential
            from tensorpotential.instructions.base import load_instructions
            from tensorpotential.calculator import TPCalculator
            assert tf.config.list_physical_devices('GPU') if use_gpu else not tf.config.list_physical_devices('GPU')
            source=Path(next(x['remote_path'] for n,x in card['files'].items() if n.endswith('model.yaml'))).parent
            model=TensorPotential(potential=load_instructions(str(source/'model.yaml')),param_dtype=tf.float64)
            model.load_checkpoint(checkpoint_name=str(source/'checkpoint'),expect_partial=True,assert_existing_objects_matched=True,model_only=True)
            model.model.decorate_compute_function(jit_compile=False)
            calculator=TPCalculator(model.model,enable_uq_if_available=False)
            def evaluate(atoms):
                atoms.calc=calculator
                return float(atoms.get_potential_energy()),atoms.get_forces(),atoms.get_stress()
            result['versions']={p:importlib.metadata.version(p) for p in ('tensorpotential','tensorflow','numpy','ase')}
            result['device']=[d.name for d in tf.config.list_physical_devices('GPU')] if use_gpu else 'CPU, CUDA_VISIBLE_DEVICES empty'
        predictions=[];telemetry=[]
        for row in rows:
            path=ROOT/row['poscar'];assert sha(path)==row['poscar_sha256']
            atoms=read(path,format='vasp');assert len(atoms)==32 and atoms.get_chemical_symbols()==['Al']*32
            started=time.perf_counter();energy,forces,stress=evaluate(atoms);seconds=time.perf_counter()-started
            assert forces.shape==(32,3) and stress.shape==(6,) and np.isfinite(np.r_[energy,forces.ravel(),stress]).all()
            predictions.append(dict(configuration_id=row['configuration_id'],parent_id=row['parent_id'],split=row['split'],
                energy_eV=energy,forces_eV_per_A=forces.tolist(),stress_eV_per_A3=stress.tolist(),volume_A3=atoms.get_volume(),inference_seconds=seconds))
            if use_gpu and len(predictions) in (1,20,40):telemetry.extend(gpu_sample())
            save(WORK/'predictions.json',dict(complete=False,model_id=card['model_id'],model_sha256=card['sha256'],frames=predictions))
        if use_gpu:assert telemetry,'Inference process not observed on GPU'
        # Verify files remain byte-identical after evaluation; no source overwrite.
        for item in card['files'].values():assert sha(item['remote_path'])==item['sha256']
        save(WORK/'predictions.json',dict(complete=True,model_id=card['model_id'],model_sha256=card['sha256'],
            contract=dict(energy='eV per cell, raw original model output',forces='eV/A Cartesian',stress='eV/A^3 tensile-positive Voigt xx yy zz yz xz xy'),frames=predictions))
        result.update(status='passed',frames=40,gpu_samples=telemetry,execution_device='GPU' if use_gpu else 'CPU',predictions_sha256=sha(WORK/'predictions.json'),source_unchanged=True)
    except Exception:
        result.update(status='failed',error=traceback.format_exc())
    result['finished_utc']=datetime.now(timezone.utc).isoformat();save(WORK/'result.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':sys.exit(main())
