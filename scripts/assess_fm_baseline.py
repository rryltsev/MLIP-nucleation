"""Verify immutable original-model inference and compute train-aligned DFT errors."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

import numpy as np
from assess_al32_dataset import load_archive
from baseline_metrics import evaluate

ROOT=Path(__file__).resolve().parents[1]


def sha(data):return hashlib.sha256(data).hexdigest()


def verified_archive(path):
    m=json.loads((path/'manifest.json').read_text())
    assert m['stage']=='collected'
    b=json.loads((path/'artifact-backup.json').read_text())
    archive=ROOT/'artifacts/dft-run-backups'/Path(b['remote_archive']).name
    assert b['verified_retrieval'] and archive.stat().st_size==b['bytes'] and sha(archive.read_bytes())==b['sha256']
    with tarfile.open(archive) as tar:
        members=tar.getmembers();assert all(x.isfile() for x in members)
        assert len({x.name for x in members})==len(members) and {x.name for x in members}==set(b['members'])
        data={x.name:tar.extractfile(x).read() for x in members}
    for n,content in data.items():assert sha(content)==b['members'][n]['sha256']
    for n,h in m['input_sha256'].items():assert sha(data[n])==h
    collected=json.loads((path/'collected-sha256.json').read_text())
    for n,h in collected.items():assert sha((path/n).read_bytes())==h==sha(data[n])
    return m,data,b


def main(path):
    m,data,b=verified_archive(path)
    assert m['runtime_status']=='passed'
    result=json.loads(data['result.json']);assert result['status']=='passed'
    cfg=json.loads(data['evaluation.json']);frozen=json.loads(data['dataset-manifest.json'])
    rows=frozen['configurations'];labels_doc=json.loads(data['labels.json'])
    assert sha(data['labels.json'])==cfg['labels_sha256']
    assert sha(data['dataset-manifest.json'])==cfg['geometry_manifest_sha256']
    assert {s:sum(r['split']==s for r in rows) for s in ('train','validation','test')}=={'train':21,'validation':6,'test':13}
    pointer=json.loads(data['dataset-pointer.json'])
    dft,dfbackup=load_archive((ROOT/pointer['labels']).parent)
    assert dft['labels.json']==data['labels.json']
    labels={x['configuration_id']:x for x in labels_doc['labels']}
    assert len(rows)==len(labels)==40 and labels_doc['complete']
    metrics={};predhashes={};cases={}
    for backend in ('dpa4','grace'):
        case=json.loads(data[backend+'/result.json']);card=json.loads(data[backend+'-checkpoint.json'])
        pred=json.loads(data[backend+'/predictions.json'])
        assert case==result['cases'][backend] and case['status']=='passed'
        assert case['source_unchanged'] and case['job_id']==m['job_id']
        if cfg['resources']['gpus']:
            assert case['gpu_samples'] and case['execution_device']=='GPU'
        else:assert case['execution_device']=='CPU' and not case['gpu_samples']
        assert case['code_commit']==m['code_commit'] and case['frames']==40
        assert pred['complete'] and pred['model_sha256']==card['sha256']==case['model_sha256']
        assert pred['model_id']==card['model_id']==case['model_id']
        assert case['source_file_sha256']=={n:x['sha256'] for n,x in card['files'].items()}
        assert case['input_sha256']==json.loads(data['run-inputs.json'])
        assert sha(data[backend+'/predictions.json'])==case['predictions_sha256']
        frames=pred['frames'];byid={x['configuration_id']:x for x in frames}
        assert len(frames)==len(byid)==40 and [x['configuration_id'] for x in frames]==[r['configuration_id'] for r in rows]
        for row in rows:
            cid=row['configuration_id'];p=byid[cid];label=labels[cid]
            assert p['parent_id']==row['parent_id']==label['parent_id']
            assert p['split']==row['split']==label['split']
            assert sha(data[row['poscar']])==row['poscar_sha256']==label['source_hashes']['POSCAR']
            assert abs(p['volume_A3']-label['volume_A3'])<1e-6
            assert np.asarray(p['forces_eV_per_A']).shape==(32,3) and np.asarray(p['stress_eV_per_A3']).shape==(6,)
        metrics[backend]=evaluate(rows,byid,labels)
        assert len(metrics[backend]['offset_fit_ids'])==21
        assert abs(metrics[backend]['metrics_by_split']['train']['energy_aligned_eV_per_atom']['signed_mean'])<1e-10
        predhashes[backend]=sha(data[backend+'/predictions.json']);cases[backend]=case
    scheduler=json.loads((path/'scheduler.json').read_text())
    parent=next(r.split('|') for r in scheduler['output'].splitlines() if r.split('|')[0]==m['job_id'])
    assert parent[1:3]==['COMPLETED','0:0']
    cpu_seconds=int(parent[4])*int(parent[5]);gpu_seconds=int(parent[4])*cfg['resources']['gpus']
    assert cpu_seconds<=cfg['budget']['maximum_allocated_CPU_seconds'] and gpu_seconds<=cfg['budget']['maximum_GPU_seconds']
    checks={};attempts=[];total_cpu=0;total_gpu=0
    for run in sorted((ROOT/'runs').glob('fm-al32-baseline-20261019-*/manifest.json')):
        rm,rd,rb=verified_archive(run.parent)
        observation=json.loads((run.parent/'scheduler.json').read_text())
        record=next(l.split('|') for l in observation['output'].splitlines() if l.split('|')[0]==rm['job_id'])
        charge=int(record[4])*int(record[5]);total_cpu+=charge;total_gpu+=int(record[4])*rm['resources']['gpus']
        attempts.append(dict(run_id=rm['run_id'],job_id=rm['job_id'],state=record[1],CPU_seconds=charge,GPU_seconds=int(record[4])*rm['resources']['gpus'],archive_sha256=rb['sha256']))
        if 'derivative_check' not in json.loads(rd['evaluation.json']):continue
        checks[rm['run_id']]={}
        for backend in ('dpa4','grace'):
            check=json.loads(rd[backend+'/derivative-check.json'])
            assert check['configuration_id']=='fcc-04' and check['split']=='train' and check['source_unchanged']
            assert abs(check['force_absolute_error_eV_per_A']-abs(check['analytic_force_x_eV_per_A']-check['finite_difference_force_x_eV_per_A']))<1e-12
            assert abs(check['max_stress_error_eV_per_A3']-np.max(np.abs(np.array(check['analytic_stress_eV_per_A3'])-check['finite_difference_stress_eV_per_A3'])))<1e-12
            settings=check['settings']
            assert check['passed']==bool(check['force_absolute_error_eV_per_A']<=settings['force_tolerance_eV_per_A'] and check['max_stress_error_eV_per_A3']<=settings['stress_tolerance_eV_per_A3'])
            checks[rm['run_id']][backend]=check
    assert all(check['passed'] for check in checks['fm-al32-baseline-20261019-04'].values())
    assert total_cpu<=6000 and total_gpu==0
    combined=json.loads((path/'scheduler-all-attempts.json').read_text())
    assert combined['total_CPU_seconds']==412226+total_cpu and not combined['live_related_jobs']
    for attempt in attempts:
        actual=next(j for j in combined['jobs'] if j['job_id']==attempt['job_id'])
        assert actual['CPU_seconds']==attempt['CPU_seconds']
    assessment=dict(assessed_utc=datetime.now(timezone.utc).isoformat(),status='passed',
        analysis_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        analysis_source_sha256=sha(Path(__file__).read_bytes()),metrics_source_sha256=sha((ROOT/'scripts/baseline_metrics.py').read_bytes()),
        inference_code_commit=m['code_commit'],job_id=m['job_id'],metrics=metrics,predictions_sha256=predhashes,
        config_sha256=sha(data['evaluation.json']),labels_sha256=sha(data['labels.json']),geometry_manifest_sha256=sha(data['dataset-manifest.json']),
        source_cases=cases,baseline_backup={k:b[k] for k in ('sha256','bytes','remote_archive','local_archive','verified_retrieval')},
        DFT_backup=dfbackup,units=dict(energy='eV/atom',forces='eV/A Cartesian components',stress='eV/A^3 six tensile-positive Voigt components xx yy zz yz xz xy',eV_per_A3_to_GPa=160.2176634),
        derivative_checks=checks,budget=dict(allocated_GPU_seconds=total_gpu,allocated_CPU_seconds=total_cpu,attempts=attempts,
                    baseline_only_CPU_seconds=cpu_seconds,prior_DFT_kplib_CPU_hours=412226/3600,
                    conservative_combined_CPU_hours=(412226+total_cpu)/3600),
        test_policy=cfg['test_policy'],limitations=cfg['limits'])
    (path/'assessment.json').write_text(json.dumps(assessment,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({backend:{'delta_Al_eV_per_atom':x['delta_Al_eV_per_atom'],'splits':x['metrics_by_split']} for backend,x in metrics.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    main(p.parse_args().run.resolve())
