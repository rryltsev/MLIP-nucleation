"""Audit a completed LoRA run from its verified private evidence bundle."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from datetime import datetime,timezone

import numpy as np
from ase.io import read
from assess_finetune import close
from finetune_selection import candidate_metrics,choose

ROOT=Path(__file__).resolve().parents[1]


def sha(b):return hashlib.sha256(b).hexdigest()


def differences(left,right,rows,allow_constant=False):
    delta=np.asarray([(right[r['configuration_id']]['energy_eV']-left[r['configuration_id']]['energy_eV'])/r['atoms'] for r in rows])
    offset=float(np.mean([v for v,r in zip(delta,rows) if r['split']=='train'])) if allow_constant else 0.
    return dict(energy_constant_eV_per_atom=offset,energy_nonconstant_max_eV_per_atom=float(np.max(np.abs(delta-offset))),
        force_max_eV_per_A=max(float(np.max(np.abs(np.asarray(right[c]['forces_eV_per_A'])-left[c]['forces_eV_per_A']))) for c in left),
        stress_max_eV_per_A3=max(float(np.max(np.abs(np.asarray(right[c]['stress_eV_per_A3'])-left[c]['stress_eV_per_A3']))) for c in left))


def main(run):
    manifest=json.loads((run/'manifest.json').read_text());assert manifest['stage']=='collected' and manifest['runtime_status']=='passed'
    backup=json.loads((run/'artifact-backup.json').read_text());assert backup['remote_archive_verified'] and backup['evidence_verified_retrieval']
    bundle=ROOT/'artifacts/dft-run-backups'/Path(backup['remote_evidence_archive']).name
    assert sha(bundle.read_bytes())==backup['evidence_sha256']
    hashes={n:r['sha256'] for n,r in backup['members'].items()};data={}
    with tarfile.open(bundle) as tar:
        members=tar.getmembers();assert len({m.name for m in members})==len(members)
        for m in members:
            assert m.isfile();content=tar.extractfile(m).read();assert sha(content)==hashes[m.name]
            data[m.name]=content
    for n,h in manifest['input_sha256'].items():assert sha(data[n])==h
    assert not any('sealed-test' in n or '/test/' in n or 'final-test' in n for n in manifest['input_sha256'])
    for n,h in json.loads((run/'collected-sha256.json').read_text()).items():assert sha((run/n).read_bytes())==hashes[n]==h
    cfg=json.loads(data['study.json']);allrows=json.loads(data['dataset-manifest.json'])['configurations']
    assert sha(data['dataset-manifest.json'])==cfg['geometry_manifest_sha256']
    rows=[r for r in allrows if r['split'] in ('train','validation')];assert len(rows)==27
    labels={}
    for split in ['train','validation']:
        frames=read(io.StringIO(data['data/'+split+'.extxyz'].decode()),format='extxyz',index=':');assert len(frames)==cfg['split_counts'][split]
        for a in frames:labels[a.info['configuration_id']]=dict(energy_free_eV=float(a.get_potential_energy()),forces_eV_per_A=a.get_forces().tolist(),stress_eV_per_A3=a.get_stress().tolist())
    zero=json.loads(data['dpa4/zero-step.json']);initial=json.loads(data['dpa4/initial-predictions.json'])
    for key,left,right,constant in [('original_vs_initialized','original','merged',True),('live_vs_merged','live','merged',False)]:
        diff=differences(initial[left],initial[right],rows,constant);assert close(diff,zero[key])
        assert all(diff[k]<=v for k,v in cfg['lora']['zero_step_tolerances'].items())
    assert zero['status']=='passed' and zero['all_adapter_B_zero']
    assert zero['initial_merged_sha256']==hashes['dpa4/initial-merged.pt']
    inventory=json.loads(data['dpa4/parameter-inventory.json']);updates=json.loads(data['dpa4/parameter-update-check.json'])
    assert updates['status']=='passed' and not updates['frozen_changed'] and updates['updated_adapters']
    assert inventory['parameters'].keys()==updates['final_sha256'].keys()
    for name,p in inventory['parameters'].items():
        if not p['trainable']:assert p['initial_sha256']==updates['final_sha256'][name]
    for name in updates['updated_adapters']+updates['updated_original_parameters']:
        p=inventory['parameters'][name]
        assert p['trainable'] and p['initial_sha256']!=updates['final_sha256'][name]
    assert all(inventory['parameters'][n]['adapter'] for n in updates['updated_adapters'])
    s=json.loads(data['dpa4/selection.json']);lock=json.loads(data['selection-lock.json'])
    assert not s['test_labels_opened'] and lock['selection_sha256']==hashes['dpa4/selection.json']
    card=json.loads(data['dpa4-checkpoint.json']);original_path=card['files'][card['filename']]['remote_path']
    marker='/mlip-research/runs/'+manifest['run_id']+'/'
    history=[];export_maxima={}
    assert [x['progress'] for x in s['all_candidates']]==[0]+list(range(cfg['training']['dpa4_checkpoint_interval'],cfg['training']['dpa4_steps']+1,cfg['training']['dpa4_checkpoint_interval']))
    for item in s['all_candidates']:
        step=item['progress']
        if step==0:predictions=initial['original'];assert item['weight_sha256']=={original_path:card['sha256']}
        else:
            e=json.loads(data[f'dpa4/export-checks/{step:04d}.json'])
            diff=differences(e['live_predictions'],e['merged_predictions'],rows)
            for k,v in diff.items():export_maxima[k]=max(export_maxima.get(k,0.),abs(v))
            assert e['status']=='passed' and close(diff,e['differences'])
            assert all(diff[k]<=v for k,v in cfg['lora']['zero_step_tolerances'].items())
            assert e['merged_sha256']==hashes[f'dpa4/merged/{step:04d}.pt']
            native=e['native_checkpoint']
            native_relative=native.split(marker,1)[1] if marker in native else 'dpa4/'+native
            assert '..' not in native_relative.split('/')
            assert e['native_sha256']==hashes[native_relative]
            candidate=json.loads(data[f'dpa4/candidate-evaluations/{step:04d}.json'])
            predictions=candidate['predictions'];assert predictions==e['merged_predictions']
        recomputed=candidate_metrics(rows,predictions,labels,cfg['selection'])
        assert close(recomputed['metrics'],item['metrics']) and close(recomputed['score'],item['score'])
        for p,h in item['weight_sha256'].items():
            if step:assert hashes[p.split(marker,1)[1]]==h
        history.append(item)
    expected=choose(history);assert s['progress']==expected['progress'] and s['weight_sha256']==expected['weight_sha256']
    assert close(s['metrics'],expected['metrics']) and close(s['score'],expected['score'])
    assert lock['weight_sha256']==s['weight_sha256']
    original=history[0]['metrics']['metrics_by_split']['validation'];selected=s['metrics']['metrics_by_split']['validation']
    guards={k:selected[k]['rmse']<=original[k]['rmse'] for k in ('forces_eV_per_A','stress_eV_per_A3')}
    assert guards==s['derivative_nonregression'] and all(guards.values())==s['derivative_guard_passed']
    process=json.loads(data['dpa4/training-process.json']);assert process['exit_code']==0 and process['GPU_samples']
    prior=ROOT/cfg['lora']['matched_full_finetune'];full=json.loads(prior.read_text())
    assert set(full['metrics']['metrics_by_split'])=={'train','validation'}
    report=dict(status='passed',assessed_utc=datetime.now(timezone.utc).isoformat(),analysis_source_sha256=sha(Path(__file__).read_bytes()),
        analysis_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),training_code_commit=manifest['code_commit'],
        source_archive_sha256=backup['sha256'],evidence_sha256=backup['evidence_sha256'],backup_record_sha256=sha((run/'artifact-backup.json').read_bytes()),
        prior_full_selection_sha256=sha(prior.read_bytes()),study_sha256=hashes['study.json'],selection_lock_sha256=hashes['selection-lock.json'],
        selected_progress=s['progress'],weight_sha256=s['weight_sha256'],derivative_guard_passed=s['derivative_guard_passed'],
        parameter_counts={k:inventory[k] for k in ['total','trainable','frozen']},updated_adapter_tensors=len(updates['updated_adapters']),
        export_maxima=export_maxima,export_checkpoints=len(history)-1,
        parameter_scope=dict(adapter_parameters=sum(v['numel'] for v in inventory['parameters'].values() if v['adapter']),trainable_original_parameters=sum(v['numel'] for v in inventory['parameters'].values() if v['trainable'] and not v['adapter']),updated_original_tensors=len(updates['updated_original_parameters'])),
        zero_step=zero,training_seconds=process['seconds'],GPU_sample_count=len(process['GPU_samples']),
        validation_comparison=dict(original=original,prior_full_selected=full['metrics']['metrics_by_split']['validation'],lora_selected=selected),
        selected_metrics=s['metrics'],history=[dict(progress=x['progress'],score=x['score']) for x in history],
        archive_backend='Full native archive verified on cluster; actual evidence bundle on workstation; no claim of full local binary copy',
        limitations=cfg['limitations'],test_policy='No old test predictions, label access, re-splitting or test-driven acquisition')
    (run/'assessment.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','selected_progress','parameter_counts','derivative_guard_passed']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);main(p.parse_args().run.resolve())
