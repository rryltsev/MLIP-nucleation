"""One fixed DPA4 LoRA experiment; train/validation only, no test access."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
from finetune_selection import candidate_metrics,choose

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('base_finetune',ROOT/'al32-finetune-run.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
CFG=base.CFG
WORK=ROOT/'dpa4'


def differences(left,right,rows,allow_constant=False):
    delta=np.asarray([(right[r['configuration_id']]['energy_eV']-left[r['configuration_id']]['energy_eV'])/r['atoms'] for r in rows])
    offset=float(np.mean([v for v,r in zip(delta,rows) if r['split']=='train'])) if allow_constant else 0.
    return dict(energy_constant_eV_per_atom=offset,energy_nonconstant_max_eV_per_atom=float(np.max(np.abs(delta-offset))),
        force_max_eV_per_A=max(float(np.max(np.abs(np.asarray(right[c]['forces_eV_per_A'])-left[c]['forces_eV_per_A']))) for c in left),
        stress_max_eV_per_A3=max(float(np.max(np.abs(np.asarray(right[c]['stress_eV_per_A3'])-left[c]['stress_eV_per_A3']))) for c in left))


def check_difference(evidence):
    for key,tolerance in CFG['lora']['zero_step_tolerances'].items():assert evidence[key]<=tolerance,(key,evidence[key],tolerance)


def live_predict(model,rows,atoms):
    import torch
    was_training=model.training;model.eval();device=next(model.parameters()).device
    result={};types=model.get_type_map()
    try:
        for row in rows:
            a=atoms[row['configuration_id']]
            with torch.enable_grad():
                out=model(coord=torch.tensor(a.positions[None],dtype=torch.float64,device=device),
                    atype=torch.tensor([[types.index(s) for s in a.get_chemical_symbols()]],dtype=torch.int64,device=device),
                    box=torch.tensor(a.cell.array.reshape(1,9),dtype=torch.float64,device=device))
            stress=-out['virial'].detach().cpu().numpy().reshape(3,3)/a.get_volume()
            result[row['configuration_id']]=dict(energy_eV=float(out['energy'].detach().cpu().numpy().ravel()[0]),
                forces_eV_per_A=out['force'].detach().cpu().numpy().reshape(-1,3).tolist(),stress_eV_per_A3=stress.ravel()[[0,4,8,5,2,1]].tolist())
    finally:model.train(was_training)
    return result


def main():
    result=dict(status='failed',job_id=os.environ.get('SLURM_JOB_ID'),scope='27 train/validation frames only; no test evaluation')
    try:
        base.verify();rows,atoms,labels=base.dataset(final=False)
        assert {r['split'] for r in rows}=={'train','validation'}
        card=json.loads((ROOT/'dpa4-checkpoint.json').read_text());source=Path(card['remote_root'])/card['filename']
        cfg=json.loads((Path(card['remote_root'])/card['configuration']).read_text());t=CFG['training']
        cfg['model']['use_compile']=False;cfg['model']['lora']={k:CFG['lora'][k] for k in ['rank','alpha']}
        cfg['optimizer']={'type':'Adam'}
        cfg['learning_rate']=dict(type='exp',start_lr=t['learning_rate_start'],stop_lr=t['learning_rate_end'],decay_steps=100)
        cfg['loss']=dict(type='ener',intensive_ener_virial=True)
        for suffix,key in [('e','energy_prefactor'),('f','force_prefactor'),('v','virial_prefactor')]:
            cfg['loss']['start_pref_'+suffix]=cfg['loss']['limit_pref_'+suffix]=t['dpa4_loss'][key]
        cfg['training']=dict(training_data=dict(systems=[str(ROOT/'data/deepmd/train')],batch_size=t['batch_size']),
            validation_data=dict(systems=[str(ROOT/'data/deepmd/validation')],batch_size=1,numb_btch=6),
            numb_steps=t['dpa4_steps'],seed=CFG['seed'],disp_freq=100,save_freq=t['dpa4_checkpoint_interval'],max_ckpt_keep=20,disp_file='lcurve.out',save_ckpt='model.ckpt')
        base.save(WORK/'input.json',cfg)
        cmd=[sys.executable,str(ROOT/'lora-child.py'),'--pt','train','input.json','--finetune',str(source)]
        started=time.monotonic();telemetry=[]
        with (WORK/'train.log').open('wb') as log:
            child=subprocess.Popen(cmd,cwd=WORK,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                if time.monotonic()-started>600:
                    os.killpg(child.pid,signal.SIGKILL);child.wait();raise TimeoutError('LoRA child exceeded600s; no automatic retry')
                gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
                telemetry.extend(line for line in gpu.splitlines() if line.split(',')[0].strip()==str(child.pid));time.sleep(1)
        base.save(WORK/'training-process.json',dict(command=cmd,PID=child.pid,exit_code=child.returncode,seconds=time.monotonic()-started,GPU_samples=telemetry))
        assert child.returncode==0,'See dpa4/train.log';assert telemetry,'GPU process not observed'
        zero=json.loads((WORK/'zero-step.json').read_text());assert zero['status']=='passed'
        originals=json.loads((WORK/'initial-predictions.json').read_text())['original']
        history=[dict(progress=0,checkpoint=str(source),weight_sha256={str(source):base.sha(source)},**candidate_metrics(rows,originals,labels,CFG['selection']))]
        for step in range(t['dpa4_checkpoint_interval'],t['dpa4_steps']+1,t['dpa4_checkpoint_interval']):
            evidence=json.loads((WORK/f'export-checks/{step:04d}.json').read_text())
            assert evidence['status']=='passed';checkpoint=WORK/f'merged/{step:04d}.pt'
            assert base.sha(checkpoint)==evidence['merged_sha256']
            scored=candidate_metrics(rows,evidence['merged_predictions'],labels,CFG['selection'])
            item=dict(progress=step,checkpoint=str(checkpoint),weight_sha256={str(checkpoint):base.sha(checkpoint)},**scored)
            base.save(WORK/f'candidate-evaluations/{step:04d}.json',dict(**item,predictions=evidence['merged_predictions']),exclusive=True)
            history.append(item)
        winner=choose(history);val=winner['metrics']['metrics_by_split']['validation'];baseline=history[0]['metrics']['metrics_by_split']['validation']
        guard={key:val[key]['rmse']<=baseline[key]['rmse'] for key in ('forces_eV_per_A','stress_eV_per_A3')}
        selection=dict(status='validation_selected',**winner,all_candidates=history,test_labels_opened=False,derivative_nonregression=guard,derivative_guard_passed=all(guard.values()))
        base.save(WORK/'selection.json',selection,exclusive=True)
        base.save(ROOT/'selection-lock.json',dict(status='validation_selection_locked_no_test',selection_sha256=base.sha(WORK/'selection.json'),weight_sha256=winner['weight_sha256']),exclusive=True)
        result.update(status='passed',selected_progress=winner['progress'],derivative_guard_passed=all(guard.values()),baseline_retained=winner['progress']==0)
    except Exception:result['error']=traceback.format_exc()
    base.save(ROOT/'result.json',result)
    print(json.dumps(result));return 0 if result['status']=='passed' else 1


if __name__=='__main__':sys.exit(main())
