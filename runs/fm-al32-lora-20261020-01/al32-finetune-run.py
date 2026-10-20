"""Bounded A100 fine-tuning, common validation selection, then sealed test evaluation."""
from datetime import datetime,timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback

import numpy as np
from ase.io import read
from baseline_metrics import evaluate
from finetune_selection import candidate_metrics,choose

ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'study.json').read_text())
ROWS=json.loads((ROOT/'dataset-manifest.json').read_text())['configurations']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,obj,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x' if exclusive else 'w') as f:f.write(json.dumps(obj,indent=2)+'\n')


def weights(backend,checkpoint):
    paths=[Path(checkpoint)] if backend=='dpa4' else [Path(str(checkpoint)+suffix) for suffix in ('.index','.data-00000-of-00001')]
    return {str(p):sha(p) for p in paths}


def verify():
    assert os.environ.get('SLURM_JOB_ID'),'Slurm required'
    for name,h in json.loads((ROOT/'run-inputs.json').read_text()).items():assert sha(ROOT/name)==h,name
    for backend in CFG['models']:
        card=json.loads((ROOT/(backend+'-checkpoint.json')).read_text())
        for item in card['files'].values():assert sha(item['remote_path'])==item['sha256']


def original(backend):
    card=json.loads((ROOT/(backend+'-checkpoint.json')).read_text())
    if backend=='dpa4':return str(Path(card['remote_root'])/card['filename']),None
    source=Path(next(x['remote_path'] for n,x in card['files'].items() if n.endswith('model.yaml'))).parent
    return str(source/'checkpoint'),str(source/'model.yaml')


def dataset(final=False):
    atoms=[]
    for split in ('train','validation'):
        frames=read(ROOT/'data'/(split+'.extxyz'),format='extxyz',index=':')
        assert len(frames)==CFG['split_counts'][split]
        atoms.extend(frames)
    if final:
        lock=json.loads((ROOT/'selection-lock.json').read_text())
        assert lock['status']=='both_selections_locked'
        for backend,h in lock['selection_sha256'].items():
            assert sha(ROOT/backend/'selection.json')==h
            for p,want in json.loads((ROOT/backend/'selection.json').read_text())['weight_sha256'].items():assert sha(p)==want
        sealed=json.loads((ROOT/'sealed-test-source.json').read_text())
        assert sha(sealed['archive'])==sealed['archive_sha256']
        with tarfile.open(sealed['archive']) as tar:content=tar.extractfile(sealed['member']).read()
        assert hashlib.sha256(content).hexdigest()==sealed['member_sha256']
        opened=ROOT/'data/final-test.extxyz'
        if opened.exists():assert opened.read_bytes()==content
        else:opened.write_bytes(content)
        test=read(io.StringIO(content.decode()),format='extxyz',index=':');assert len(test)==13
        atoms.extend(test)
    rows=[r for r in ROWS if final or r['split']!='test']
    byid={a.info['configuration_id']:a for a in atoms}
    assert len(byid)==len(rows)==len(atoms)
    labels={}
    for r in rows:
        cid=r['configuration_id'];a=byid[cid]
        assert a.info['parent_id']==r['parent_id'] and len(a)==32 and a.get_chemical_symbols()==['Al']*32
        labels[cid]=dict(energy_free_eV=float(a.get_potential_energy()),forces_eV_per_A=a.get_forces().tolist(),stress_eV_per_A3=a.get_stress().tolist())
    return rows,byid,labels


def predict(backend,checkpoint,model_yaml,rows,atoms):
    if backend=='dpa4':
        import torch
        from deepmd.infer import DeepPot
        assert torch.cuda.is_available(),'A100 required'
        model=DeepPot(checkpoint);types=model.get_type_map()
        def one(a):
            e,f,v=model.eval(a.positions.reshape(1,-1),a.cell.array.reshape(1,9),[types.index(s) for s in a.get_chemical_symbols()])
            stress=-v.reshape(3,3)/a.get_volume()
            return float(e.ravel()[0]),f.reshape(-1,3),stress.ravel()[[0,4,8,5,2,1]]
    else:
        import tensorflow as tf
        from tensorpotential import TensorPotential
        from tensorpotential.instructions.base import load_instructions
        from tensorpotential.calculator import TPCalculator
        assert tf.config.list_physical_devices('GPU'),'A100 required'
        model=TensorPotential(potential=load_instructions(model_yaml),param_dtype=tf.float64)
        model.load_checkpoint(checkpoint_name=checkpoint,expect_partial=True,assert_existing_objects_matched=True,model_only=True)
        model.model.decorate_compute_function(jit_compile=False)
        calculator=TPCalculator(model.model,enable_uq_if_available=False)
        def one(a):
            a.calc=calculator
            return float(a.get_potential_energy()),a.get_forces(),a.get_stress()
    result={}
    for row in rows:
        a=atoms[row['configuration_id']].copy();e,f,s=one(a)
        assert np.asarray(f).shape==(32,3) and np.asarray(s).shape==(6,) and np.isfinite(np.r_[e,np.asarray(f).ravel(),s]).all()
        result[row['configuration_id']]=dict(energy_eV=e,forces_eV_per_A=np.asarray(f).tolist(),stress_eV_per_A3=np.asarray(s).tolist())
    return result


def train(backend):
    work=ROOT/backend;work.mkdir(exist_ok=True);t=CFG['training']
    source,model_yaml=original(backend)
    if backend=='dpa4':
        card=json.loads((ROOT/'dpa4-checkpoint.json').read_text())
        cfg=json.loads((Path(card['remote_root'])/card['configuration']).read_text())
        cfg['model']['use_compile']=False
        cfg['optimizer']={'type':'Adam'}
        cfg['learning_rate']=dict(type='exp',start_lr=t['learning_rate_start'],stop_lr=t['learning_rate_end'],decay_steps=100)
        cfg['loss']=dict(type='ener',intensive_ener_virial=True)
        for suffix,key in [('e','energy_prefactor'),('f','force_prefactor'),('v','virial_prefactor')]:
            cfg['loss']['start_pref_'+suffix]=cfg['loss']['limit_pref_'+suffix]=t['dpa4_loss'][key]
        cfg['training']=dict(training_data=dict(systems=[str(ROOT/'data/deepmd/train')],batch_size=t['batch_size']),
            validation_data=dict(systems=[str(ROOT/'data/deepmd/validation')],batch_size=1,numb_btch=6),
            numb_steps=t['dpa4_steps'],seed=CFG['seed'],disp_freq=100,save_freq=t['dpa4_checkpoint_interval'],max_ckpt_keep=20,disp_file='lcurve.out',save_ckpt='model.ckpt')
        save(work/'input.json',cfg)
        command=[sys.executable,str(ROOT/'fm-child.py'),backend,'--pt','train','input.json','--finetune',source]
    else:
        import yaml
        cfg=dict(seed=CFG['seed'],cutoff=6.,data=dict(filename=str(ROOT/'data/train.extxyz'),test_filename=str(ROOT/'data/validation.extxyz'),reference_energy=0,stress_units='eV/A3',max_workers=1),
            potential=dict(filename=model_yaml,checkpoint_name=source,shift=False,reduce_elements=False),
            fit=dict(maxiter=t['grace_epochs'],optimizer='Adam',opt_params=dict(learning_rate=t['learning_rate_start'],use_ema=False),
                reset_epoch_and_step=True,reset_optimizer=True,eval_init_stats=True,scheduler='cosine_decay',scheduler_params=dict(minimum_learning_rate=t['learning_rate_end']),
                loss={name:dict(type='square',weight=t['grace_loss'][name+'_weight']) for name in ('energy','forces','stress')},
                batch_size=t['batch_size'],test_batch_size=2,jit_compile=False,checkpoint_freq=t['grace_checkpoint_interval'],save_all_regular_checkpoints=True,
                train_max_n_buckets=1,test_max_n_buckets=1))
        (work/'input.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        command=[sys.executable,str(ROOT/'fm-child.py'),backend,'input.yaml']
    env=dict(os.environ,FM_LOADED_SNAPSHOT=str(work/'loaded-initial/model'))
    (work/'loaded-initial').mkdir()
    started=time.monotonic();telemetry=[]
    with (work/'train.log').open('wb') as stream:
        proc=subprocess.Popen(command,cwd=work,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        while proc.poll() is None:
            if time.monotonic()-started>1500:
                import signal
                os.killpg(proc.pid,signal.SIGKILL);proc.wait();raise TimeoutError('Training child exceeded1500s; no automatic retry')
            q=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
            telemetry.extend(line for line in q.splitlines() if line.split(',')[0].strip()==str(proc.pid))
            time.sleep(1)
    save(work/'training-process.json',dict(command=command,PID=proc.pid,exit_code=proc.returncode,seconds=time.monotonic()-started,GPU_samples=telemetry))
    assert proc.returncode==0,(work/'train.log').read_text(errors='replace')[-4000:]
    assert telemetry,'Training child not observed on GPU'
    if backend=='dpa4':
        candidates=[(step,str(work/f'model.ckpt-{step}.pt'),None) for step in range(t['dpa4_checkpoint_interval'],t['dpa4_steps']+1,t['dpa4_checkpoint_interval'])]
        before=source;after=candidates[-1][1]
    else:
        candidates=[(epoch,str(work/f'seed/{CFG["seed"]}/checkpoints/checkpoint.epoch_{epoch}'),str(work/f'seed/{CFG["seed"]}/model.yaml')) for epoch in range(t['grace_checkpoint_interval'],t['grace_epochs']+1,t['grace_checkpoint_interval'])]
        before=source;after=candidates[-1][1]
    spec=importlib.util.spec_from_file_location('initialization_audit',ROOT/'fm-finetune.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    oldcwd=Path.cwd();os.chdir(work)
    try:check=helper.compare(Path(before),work/'loaded-initial/model',Path(after),backend)
    finally:os.chdir(oldcwd)
    save(work/'weight-update-check.json',check)
    rows,atoms,labels=dataset(final=False);history=[]
    for progress,checkpoint,yaml_path in [(0,source,model_yaml)]+candidates:
        hashes=weights(backend,checkpoint)
        predictions=predict(backend,checkpoint,yaml_path,rows,atoms)
        scored=candidate_metrics(rows,predictions,labels,CFG['selection'])
        item=dict(progress=progress,checkpoint=checkpoint,model_yaml=yaml_path,weight_sha256=hashes,**scored)
        save(work/f'candidate-evaluations/{progress:04d}.json',dict(**item,predictions=predictions),exclusive=True)
        history.append(item)
    winner=choose(history)
    selection=dict(status='validation_selected',backend=backend,**winner,
        baseline_delta_Al_eV_per_atom=history[0]['metrics']['delta_Al_eV_per_atom'],
        test_labels_opened=False,selection_rule=CFG['selection'],all_candidates=history)
    save(work/'selection.json',selection,exclusive=True)
    save(work/'phase-result.json',dict(status='passed',phase='train-select',selected_progress=winner['progress'],baseline_retained=winner['progress']==0))


def lock():
    checks={}
    for backend in CFG['models']:
        p=ROOT/backend/'selection.json';s=json.loads(p.read_text())
        assert s['status']=='validation_selected' and not s['test_labels_opened']
        for name,h in s['weight_sha256'].items():assert sha(name)==h
        checks[backend]=sha(p)
    save(ROOT/'selection-lock.json',dict(status='both_selections_locked',selection_sha256=checks,
        study_sha256=sha(ROOT/'study.json'),created_utc=datetime.now(timezone.utc).isoformat()),exclusive=True)


def final(backend):
    rows,atoms,labels=dataset(final=True)
    chosen=json.loads((ROOT/backend/'selection.json').read_text());source,model_yaml=original(backend)
    outputs={};metrics={}
    for name,checkpoint,yaml_path,delta in [('original',source,model_yaml,chosen['baseline_delta_Al_eV_per_atom']),
        ('selected',chosen['checkpoint'],chosen['model_yaml'],chosen['metrics']['delta_Al_eV_per_atom'])]:
        pred=predict(backend,checkpoint,yaml_path,rows,atoms)
        outputs[name]=pred;metrics[name]=evaluate(rows,pred,labels,fixed_train_offset=delta)
    save(ROOT/backend/'final-predictions.json',dict(complete=True,units=dict(energy='eV/cell',forces='eV/A',stress='eV/A^3 tensile Voigt xx yy zz yz xz xy'),predictions=outputs))
    save(ROOT/backend/'final-metrics.json',dict(status='passed',metrics=metrics,selection_lock_sha256=sha(ROOT/'selection-lock.json'),
        predictions_sha256=sha(ROOT/backend/'final-predictions.json'),test_used_only_after_lock=True))


if __name__=='__main__':
    backend,phase=sys.argv[1:3]
    try:
        verify()
        if phase=='train-select':train(backend)
        elif phase=='lock':lock()
        elif phase=='final':final(backend)
        else:raise ValueError(phase)
    except Exception:
        save(ROOT/(backend if backend!='all' else '')/('failure-'+phase+'.json'),dict(status='failed',error=traceback.format_exc()))
        raise
