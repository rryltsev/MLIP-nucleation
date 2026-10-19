"""Audit a terminal kpLib benchmark from verified private raw VASP archives.

Recompute fixed accuracy gates and paired timings without importing the worker.
Requires NumPy and ASE; never extracts POTCAR or changes the accepted dataset.
"""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from ase.io import read
from assess_al32_dataset import load_archive,differences
from kplib_grid import reduce_time_reversal,kpoints

ROOT=Path(__file__).resolve().parents[1]
POTCAR='1b07010218b572e3404ce1eea8fed39e24b52859ebef1517bffa2ef69b8d1315'


def sha(data):return hashlib.sha256(data).hexdigest()


def save(path,obj):path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8')


def raw_case(data,cid,name,row,manifest,grid):
    prefix='calculations/'+cid+'-'+name+'/'
    label=json.loads(data[prefix+'label.json'])
    assert label['configuration_id']==cid and label['case']==name
    assert label['parent_id']==row['parent_id'] and label['split']==row['split']=='train'
    assert label['job_id']==manifest['job_id'] and label['code_commit']==manifest['code_commit']
    assert label['electronic_converged'] and label['normal_termination']
    for n,h in label['source_hashes'].items():assert sha(data[prefix+n])==h
    assert label['source_hashes']['POSCAR']==row['poscar_sha256']
    assert label['source_hashes']['POTCAR']==POTCAR
    im=json.loads(data[prefix+'input-manifest.json'])
    for n,h in im['input_sha256'].items():assert sha(data[prefix+n])==h
    provenance=json.loads(data[prefix+'potcar-provenance.json'])
    assert provenance['family']=='PBE' and [p['variant'] for p in provenance['ordered_potentials']]==['Al']
    assert provenance['potcar_sha256']==POTCAR
    incar={k.strip():v.strip() for k,v in (l.split('=',1) for l in data[prefix+'INCAR'].decode().splitlines())}
    assert float(incar['ENCUT'])==450 and float(incar['SIGMA'])==0.05
    assert incar['GGA']=='PE' and incar['LREAL']=='.FALSE.' and incar['PREC']=='Accurate'
    text=data[prefix+'KPOINTS'].decode()
    if grid is None:
        assert text=='Fresh Gamma7 baseline\n0\nGamma\n7 7 7\n0 0 0\n'
    else:
        full=name=='selected-full'
        expected=kpoints(grid['full_coords'] if full else grid['tr_coords'],
                         grid['full_weights'] if full else grid['tr_weights'],name)
        assert text==expected
    xml=ET.fromstring(data[prefix+'vasprun.xml'])
    calc=xml.findall('calculation');assert len(calc)==1;calc=calc[0]
    pars={e.get('name'):(e.text or '').strip() for e in xml.findall('parameters//i')}
    requested={e.get('name'):(e.text or '').strip() for e in xml.findall('incar//i')}
    assert float(pars['ENMAX'])==450==float(requested['ENCUT'])
    assert float(pars['EDIFF'])==1e-7 and float(pars['SIGMA'])==0.05
    assert [int(pars[n]) for n in ('NSW','ISPIN','ISYM','ISMEAR')]==[0,1,0,0]
    steps=calc.findall('scstep');assert 0<len(steps)==label['electronic_steps']<int(pars['NELM'])
    out=data[prefix+'OUTCAR'].decode(errors='replace')
    assert 'aborting loop because EDIFF is reached' in out
    assert 'General timing and accounting informations for this job' in out
    energy=float(calc.find("energy/i[@name='e_fr_energy']").text)
    assert abs(energy-label['energy_free_eV'])<1e-8
    last=steps[-1].find('energy')
    sigma0=energy+float(last.find("i[@name='e_0_energy']").text)-float(last.find("i[@name='e_fr_energy']").text)
    assert abs(sigma0-label['energy_sigma0_eV'])<1e-8
    matrix=lambda path:np.array([[float(x) for x in e.text.split()] for e in calc.findall(path)])
    force=matrix("varray[@name='forces']/v")
    pressure=matrix("varray[@name='stress']/v")
    stress=(-pressure*1e8/1.602176634e11).ravel()[[0,4,8,5,2,1]]
    assert force.shape==(32,3) and np.max(np.abs(force-label['forces_eV_per_A']))<1e-10
    assert np.allclose(stress,label['stress_eV_per_A3'],rtol=1e-7,atol=1e-10)
    basis=matrix("structure/crystal/varray[@name='basis']/v")
    volume=float(np.linalg.det(basis));assert abs(volume-label['volume_A3'])<1e-6
    atoms=read(io.StringIO(data[prefix+'POSCAR'].decode()),format='vasp')
    assert atoms.get_chemical_symbols()==['Al']*32
    assert np.allclose(atoms.cell.array,basis,atol=1e-8,rtol=0)
    delta=matrix("structure/varray[@name='positions']/v")-atoms.get_scaled_positions();delta-=np.rint(delta)
    assert np.max(np.abs(delta))<1e-8
    coords=np.array([[float(x) for x in e.text.split()] for e in xml.findall("kpoints/varray[@name='kpointlist']/v")])
    weights=np.array([float(e.text) for e in xml.findall("kpoints/varray[@name='weights']/v")])
    assert len(coords)==len(weights)==label['actual_kpoints'] and abs(weights.sum()-1)<1e-6
    if grid is not None:
        gcoords=grid['full_coords'] if full else grid['tr_coords']
        gweights=np.array(grid['full_weights'] if full else grid['tr_weights'])
        assert coords.shape==np.asarray(gcoords).shape
        delta=coords-gcoords;delta-=np.rint(delta);assert np.max(np.abs(delta))<1e-7
        assert np.max(np.abs(weights-gweights/gweights.sum()))<1e-7
    assert np.isfinite(np.r_[energy,sigma0,force.ravel(),stress,volume,label['wall_seconds']]).all()
    assert label['wall_seconds']>0 and volume>0
    return label,dict(configuration_id=cid,case=name,actual_kpoints=len(coords),electronic_steps=len(steps),
        wall_seconds=label['wall_seconds'],energy_eV_per_atom=energy/32,
        net_force_norm_eV_per_A=float(np.linalg.norm(force.sum(axis=0))),
        stress_antisymmetry_kbar=float(np.max(np.abs(pressure-pressure.T))))


def main(path):
    manifest=json.loads((path/'manifest.json').read_text())
    assert manifest['scheduler_record'][1:3]==['COMPLETED','0:0']
    data,backup=load_archive(path)
    assert json.loads(data['result.json'])['passed']
    for n,h in manifest['input_sha256'].items():assert sha(data[n])==h
    cfg=json.loads(data['system.json']);grids=json.loads(data['grids.json'])
    refs=json.loads(data['references.json']);stored=json.loads(data['benchmark.json'])
    frozen=json.loads((ROOT/cfg['geometry_manifest']).read_text())
    assert sha((ROOT/cfg['geometry_manifest']).read_bytes())=='65a4c6dce1a072504030dca2b0340db72a497e562e40b2717104294c3030fbdd'
    rows={r['configuration_id']:r for r in frozen['configurations']}
    caldata,calbackup=load_archive(ROOT/cfg['reference_run'])
    runtime=json.loads(data['runtime.json'])
    assert runtime['ntasks']==16 and runtime['grids_sha256']==sha(data['grids.json'])
    assert runtime['reference_labels_sha256']==sha(data['references.json'])
    distances=cfg['kplib']['min_distance_A'];anchor=cfg['kplib']['anchor_min_distance_A']
    selected=stored['chosen_min_distance_A'];cases={};diagnostics=[]
    for cid,ref in refs.items():
        assert ref['row']==rows[cid] and rows[cid]['split']=='train'
        assert ref['label']==json.loads(caldata['calculations/'+cid+'-reference/label.json'])
        for d in distances:
            g=grids[cid][str(d)];tr,w=reduce_time_reversal(g['full_coords'])
            assert tr==g['tr_coords'] and w==g['tr_weights']
            assert g['num_total_kpts']==sum(w)==len(g['full_coords'])
            assert g['full_weights']==[1]*len(g['full_coords'])
            assert len(tr)==g['num_time_reversal_kpts'] and g['effective_min_distance_A']>=d-1e-6
        names=['gamma7-r0','gamma7-r1']+['d'+str(d) for d in distances]
        if selected is not None:names+=['selected-repeat','selected-full']
        assert set(names)==set(stored['cases'][cid]);cases[cid]={}
        for name in names:
            grid=None if name.startswith('gamma7-') else grids[cid][str(int(name[1:]) if name.startswith('d') else selected)]
            label,diag=raw_case(data,cid,name,rows[cid],manifest,grid)
            assert label==stored['cases'][cid][name] and label['host']==runtime['host']
            cases[cid][name]=label;diagnostics.append(diag)
    assert set(cases)==set(refs)==set(cfg['configuration_ids'])
    actual_dirs={n.split('/')[1] for n in data if n.startswith('calculations/')}
    assert actual_dirs=={cid+'-'+name for cid,cc in cases.items() for name in cc}
    def passed(a,b,tol):return all(v<=tol[k] for k,v in differences(a,b).items())
    checks=[]
    for d in distances:
        for cid in refs:
            a=cases[cid]['d'+str(d)];b=cases[cid]['d'+str(anchor)]
            checks.append(dict(min_distance_A=d,configuration_id=cid,
                versus_gamma8=differences(a,refs[cid]['label']),versus_anchor=differences(a,b),
                passed=passed(a,refs[cid]['label'],cfg['tolerances']) and passed(a,b,cfg['tolerances'])))
    candidates=[d for d in distances if d<anchor and all(c['passed'] for c in checks if c['min_distance_A']>=d)]
    assert selected==(min(candidates) if candidates else None) and checks==stored['checks']
    timing={};representation={}
    for cid in refs if selected is not None else []:
        cc=cases[cid];reduced=cc['d'+str(selected)];full=cc['selected-full']
        representation[cid]=dict(differences=differences(full,reduced),passed=passed(full,reduced,cfg['representation_tolerances']),
            full_actual_kpoints=full['actual_kpoints'],reduced_actual_kpoints=reduced['actual_kpoints'])
        baseline=[cc['gamma7-r'+str(i)]['wall_seconds'] for i in range(2)]
        chosen=[reduced['wall_seconds'],cc['selected-repeat']['wall_seconds']]
        bm=statistics.median(baseline);km=statistics.median(chosen);gen=grids[cid][str(selected)]['generation_seconds']
        timing[cid]=dict(gamma7_wall_samples_seconds=baseline,kplib_wall_samples_seconds=chosen,
            gamma7_median_seconds=bm,kplib_median_seconds=km,grid_generation_seconds=gen,
            end_to_end_speedup=bm/(km+gen),at_least10percent_faster=km+gen<=.9*bm,
            gamma7_actual_kpoints=cc['gamma7-r0']['actual_kpoints'],kplib_actual_kpoints=reduced['actual_kpoints'],
            baseline_error=differences(cc['gamma7-r0'],refs[cid]['label']),kplib_error=differences(reduced,refs[cid]['label']))
    assert timing==stored['timing'] and representation==stored['representation_checks']
    baseline_ok=all(passed(cases[cid]['gamma7-r0'],refs[cid]['label'],cfg['tolerances']) for cid in refs)
    assert baseline_ok==stored['baseline_accuracy_passed']
    recommend=selected is not None and baseline_ok and all(r['passed'] for r in representation.values()) and all(t['at_least10percent_faster'] for t in timing.values())
    assert recommend==stored['recommend_for_this_pilot']
    budget=[]
    for p in sorted((ROOT/'runs').glob('al32-*/manifest.json')):
        m=json.loads(p.read_text());r=m['scheduler_record']
        assert r[1] in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL')
        budget.append(dict(run_id=m['run_id'],job_id=m['job_id'],state=r[1],allocated_CPU_seconds=int(r[5])))
    total=sum(r['allocated_CPU_seconds'] for r in budget);assert total<=128*3600
    assessment=dict(assessed_utc=datetime.now(timezone.utc).isoformat(),
        analysis_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        analysis_source_sha256=sha(Path(__file__).read_bytes()),raw_label_quality_audit_passed=True,
        raw_cases_verified=len(diagnostics),selected_min_distance_A=selected,recommend_for_this_pilot=recommend,
        decision=stored['decision'],checks=checks,timing=timing,representation_checks=representation,
        source_sha256={n:sha(data[n]) for n in ('system.json','grids.json','references.json','benchmark.json','runtime.json')},
        raw_backup=backup,reference_backup=calbackup,budget=dict(jobs=budget,total_CPU_seconds=total,total_CPU_hours=total/3600,remaining_CPU_hours=128-total/3600),
        diagnostics=diagnostics,limitations=stored['limitations'])
    save(path/'quality-assessment.json',assessment)
    with (path/'convergence.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['configuration_id','min_distance_A','actual_kpoints','wall_seconds']+[f'{ref}_{key}' for ref in ('gamma8','anchor') for key in cfg['tolerances']]+['passed']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for check in checks:
            cid=check['configuration_id'];d=check['min_distance_A'];label=cases[cid]['d'+str(d)]
            row={k:check[k] for k in ('configuration_id','min_distance_A','passed')}
            row.update(actual_kpoints=label['actual_kpoints'],wall_seconds=label['wall_seconds'])
            for ref in ('gamma8','anchor'):
                row.update({ref+'_'+k:v for k,v in check['versus_'+ref].items()})
            writer.writerow(row)
    print(json.dumps({k:assessment[k] for k in ('raw_cases_verified','selected_min_distance_A','recommend_for_this_pilot','decision','budget')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    main(p.parse_args().run.resolve())
