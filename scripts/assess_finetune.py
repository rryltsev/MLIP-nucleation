"""Independently audit archived validation selection and the locked final test."""
import argparse
from datetime import datetime,timezone
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import tarfile

from ase.io import read
from baseline_metrics import evaluate
from finetune_selection import candidate_metrics,choose

ROOT=Path(__file__).resolve().parents[1]


def sha(data):return hashlib.sha256(data).hexdigest()


def close(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(close(v,b[k]) for k,v in a.items())
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return isinstance(b,(int,float)) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)
    return a==b


def main(run):
    manifest=json.loads((run/'manifest.json').read_text());assert manifest['stage']=='collected' and manifest['runtime_status']=='passed'
    b=json.loads((run/'artifact-backup.json').read_text());assert b['verified_retrieval']
    archive=Path(b['local_archive'])
    if not archive.is_file():archive=ROOT/'artifacts/dft-run-backups'/Path(b['remote_archive']).name
    with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==b['sha256']
    data={};hashes={}
    with tarfile.open(archive) as tar:
        members=tar.getmembers();assert all(m.isfile() for m in members)
        assert len({m.name for m in members})==len(members) and {m.name for m in members}==set(b['members'])
        for member in members:
            with tar.extractfile(member) as f:hashes[member.name]=hashlib.file_digest(f,'sha256').hexdigest()
            assert hashes[member.name]==b['members'][member.name]['sha256']
            if member.size<2500000 and 'cache' not in member.name.split('/') and Path(member.name).suffix in ('.json','.extxyz'):
                data[member.name]=tar.extractfile(member).read()
    for name,h in manifest['input_sha256'].items():assert hashes[name]==h
    assert not any('/test/' in n or n.endswith('/test.extxyz') for n in manifest['input_sha256'])
    for n,h in json.loads((run/'collected-sha256.json').read_text()).items():assert sha((run/n).read_bytes())==h==hashes[n]
    cfg=json.loads(data['study.json']);rows=json.loads(data['dataset-manifest.json'])['configurations']
    assert sha(data['dataset-manifest.json'])==cfg['geometry_manifest_sha256']
    lock=json.loads(data['selection-lock.json']);assert lock['status']=='both_selections_locked'
    assert lock['study_sha256']==hashes['study.json']
    labels={}
    for split in ('train','validation'):
        atoms=read(io.StringIO(data['data/'+split+'.extxyz'].decode()),format='extxyz',index=':')
        assert len(atoms)==cfg['split_counts'][split]
        for a in atoms:labels[a.info['configuration_id']]=dict(energy_free_eV=float(a.get_potential_energy()),forces_eV_per_A=a.get_forces().tolist(),stress_eV_per_A3=a.get_stress().tolist())
    fitting_rows=[r for r in rows if r['split']!='test'];selections={}
    marker='/mlip-research/runs/'+manifest['run_id']+'/'
    for backend in cfg['models']:
        card=json.loads(data[backend+'-checkpoint.json'])
        original_hashes={v['remote_path']:v['sha256'] for v in card['files'].values()}
        s=json.loads(data[backend+'/selection.json']);assert sha(data[backend+'/selection.json'])==lock['selection_sha256'][backend]
        candidates=[]
        for stored in s['all_candidates']:
            progress=stored['progress'];name=backend+f'/candidate-evaluations/{progress:04d}.json'
            evidence=json.loads(data[name]);recomputed=candidate_metrics(fitting_rows,evidence['predictions'],labels,cfg['selection'])
            assert close(recomputed['score'],stored['score']) and close(recomputed['metrics'],stored['metrics'])
            for p,h in stored['weight_sha256'].items():
                if progress==0:assert original_hashes[p]==h
                else:
                    assert marker in p
                    relative=p.split(marker,1)[1];assert '..' not in relative.split('/') and hashes[relative]==h
            candidates.append(stored)
        expected=choose(candidates)
        assert s['progress']==expected['progress'] and s['weight_sha256']==expected['weight_sha256'] and s['score']==expected['score']
        assert not s['test_labels_opened'];selections[backend]=s
    sealed=json.loads(data['sealed-test-source.json']);test=data['data/final-test.extxyz']
    assert sealed['member_sha256']==json.loads(data['DFT-export-provenance.json'])['files']['export/test.extxyz']
    assert sha(test)==sealed['member_sha256']
    test_atoms=read(io.StringIO(test.decode()),format='extxyz',index=':');assert len(test_atoms)==13
    for a in test_atoms:labels[a.info['configuration_id']]=dict(energy_free_eV=float(a.get_potential_energy()),forces_eV_per_A=a.get_forces().tolist(),stress_eV_per_A3=a.get_stress().tolist())
    reports={}
    for backend in cfg['models']:
        pred=json.loads(data[backend+'/final-predictions.json']);stored=json.loads(data[backend+'/final-metrics.json'])
        assert stored['status']=='passed' and stored['selection_lock_sha256']==hashes['selection-lock.json'] and stored['test_used_only_after_lock']
        assert stored['predictions_sha256']==hashes[backend+'/final-predictions.json']
        result={}
        for kind in ('original','selected'):
            delta=selections[backend]['baseline_delta_Al_eV_per_atom'] if kind=='original' else selections[backend]['metrics']['delta_Al_eV_per_atom']
            result[kind]=evaluate(rows,pred['predictions'][kind],labels,fixed_train_offset=delta)
        assert close(result,stored['metrics']);reports[backend]=result
    report=dict(assessed_utc=datetime.now(timezone.utc).isoformat(),status='passed',
        analysis_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        analysis_source_sha256=sha(Path(__file__).read_bytes()),inference_code_commit=manifest['code_commit'],
        study_sha256=hashes['study.json'],selection_lock_sha256=hashes['selection-lock.json'],raw_archive_sha256=b['sha256'],
        selection={k:{f:v[f] for f in ('progress','score','checkpoint','weight_sha256')} for k,v in selections.items()},
        metrics=reports,test_policy='Test used only after both immutable validation selections; no subsequent selection/tuning',limitations=cfg['limitations'])
    (run/'assessment.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='passed',selection=report['selection']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    main(parser.parse_args().run.resolve())
