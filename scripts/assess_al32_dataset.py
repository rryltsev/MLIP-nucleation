"""Independently audit dense Al32 calibration, raw labels and both exports.

Reads verified private tar archives without extracting licensed content. Requires
NumPy and ASE. Does not run VASP, select training hyperparameters or fit models.
"""
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import xml.etree.ElementTree as ET

import numpy as np
from ase.io import read

ROOT=Path(__file__).resolve().parents[1]
CAL=ROOT/'runs/al32-calibration-20261018-02'
LAB=ROOT/'runs/al32-labeling-20261018-02'


def sha(data): return hashlib.sha256(data).hexdigest()


def save(path,obj):
    with path.open('w',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(obj,indent=2,sort_keys=True)+'\n')


def load_archive(root):
    record=json.loads((root/'artifact-backup.json').read_text())
    archive=Path(record['local_archive'])
    if not archive.is_file(): archive=ROOT/'artifacts/dft-run-backups'/Path(record['remote_archive']).name
    assert record['verified_retrieval'] and archive.stat().st_size==record['bytes'] and sha(archive.read_bytes())==record['sha256']
    with tarfile.open(archive) as t:
        members=t.getmembers()
        assert all(m.isfile() for m in members)
        assert len({m.name for m in members})==len(members)
        assert {m.name for m in members}==set(record['members'])
        data={m.name:t.extractfile(m).read() for m in members}
    for name,content in data.items():
        assert sha(content)==record['members'][name]['sha256']
    collection=json.loads((root/'collection.json').read_text())
    for name,h in collection['sha256'].items():
        assert sha((root/name).read_bytes())==h==sha(data[name])
    return data,{k:record[k] for k in ('sha256','bytes','remote_archive','local_archive','verified_retrieval')}


def differences(a,b):
    return dict(energy_eV_per_atom=abs(a['energy_free_eV']-b['energy_free_eV'])/32,
        max_force_component_eV_per_A=float(np.max(np.abs(np.array(a['forces_eV_per_A'])-b['forces_eV_per_A']))),
        max_stress_component_eV_per_A3=float(np.max(np.abs(np.array(a['stress_eV_per_A3'])-b['stress_eV_per_A3']))))


def raw_label(data,root,row,case,encut,k,sigma,cfg,manifest):
    prefix='calculations/'+case+'/'
    label=json.loads(data[prefix+'label.json'])
    assert label['electronic_converged'] and label['normal_termination']
    assert label['job_id']==manifest['job_id'] and label['code_commit']==manifest['code_commit']
    assert label['configuration_id']==row['configuration_id'] and label['parent_id']==row['parent_id'] and label['split']==row['split']
    for name,h in label['source_hashes'].items(): assert sha(data[prefix+name])==h
    assert label['source_hashes']['POTCAR']==cfg['reference_protocol']['potcar_sha256']
    assert label['source_hashes']['POSCAR']==row['poscar_sha256']
    inputs=json.loads(data[prefix+'input-manifest.json'])
    for name,h in inputs['input_sha256'].items(): assert sha(data[prefix+name])==h
    provenance=json.loads(data[prefix+'potcar-provenance.json'])
    assert provenance['family']=='PBE' and [x['variant'] for x in provenance['ordered_potentials']]==['Al']
    assert provenance['potcar_sha256']==label['source_hashes']['POTCAR']
    incar={a.strip():b.strip() for a,b in (line.split('=',1) for line in data[prefix+'INCAR'].decode().splitlines())}
    assert incar['GGA']=='PE' and incar['LREAL']=='.FALSE.' and incar['PREC']=='Accurate'
    assert float(incar['ENCUT'])==encut and float(incar['SIGMA'])==sigma
    assert [int(x) for x in data[prefix+'KPOINTS'].decode().splitlines()[3].split()]==[k]*3
    xml=ET.fromstring(data[prefix+'vasprun.xml'])
    calcs=xml.findall('calculation'); assert len(calcs)==1
    calc=calcs[0]
    pars={e.get('name'):(e.text or '').strip() for e in xml.findall('parameters//i')}
    requested={e.get('name'):(e.text or '').strip() for e in xml.findall('incar//i')}
    assert float(pars['ENMAX'])==encut==float(requested['ENCUT'])
    assert float(pars['SIGMA'])==sigma and float(pars['EDIFF'])==1e-7
    assert [int(pars[n]) for n in ('NSW','ISPIN','ISYM','ISMEAR')]==[0,1,0,0]
    steps=calc.findall('scstep')
    assert len(steps)==label['electronic_steps'] and 0<len(steps)<int(pars['NELM'])
    out=data[prefix+'OUTCAR'].decode(errors='replace')
    assert 'aborting loop because EDIFF is reached' in out and 'General timing and accounting informations for this job' in out
    energy=float(calc.find("energy/i[@name='e_fr_energy']").text)
    assert abs(energy-label['energy_free_eV'])<1e-8
    last=steps[-1].find('energy')
    sigma0=energy+float(last.find("i[@name='e_0_energy']").text)-float(last.find("i[@name='e_fr_energy']").text)
    assert abs(sigma0-label['energy_sigma0_eV'])<1e-8
    assert [e[0].text.strip() for e in xml.findall("atominfo/array[@name='atoms']/set/rc")]==['Al']*32
    matrix=lambda path:np.array([[float(x) for x in e.text.split()] for e in calc.findall(path)])
    basis=matrix("structure/crystal/varray[@name='basis']/v")
    volume=float(np.linalg.det(basis))
    assert abs(volume-label['volume_A3'])<max(1e-6,volume*1e-9)
    force=matrix("varray[@name='forces']/v")
    assert force.shape==(32,3) and np.max(np.abs(force-label['forces_eV_per_A']))<1e-10
    pressure=matrix("varray[@name='stress']/v")
    # SI kbar to eV/A^3, tensile positive; independent CODATA2019 constant.
    stress=(-pressure*1e8/1.602176634e11).ravel()[[0,4,8,5,2,1]]
    assert np.allclose(stress,label['stress_eV_per_A3'],rtol=1e-7,atol=1e-10)
    assert np.isfinite(np.r_[energy,sigma0,volume,force.ravel(),stress]).all() and volume>0
    atoms=read(io.StringIO(data[prefix+'POSCAR'].decode()),format='vasp')
    assert np.allclose(atoms.cell.array,basis,atol=1e-8,rtol=0)
    frac=matrix("structure/varray[@name='positions']/v")
    delta=frac-atoms.get_scaled_positions(); delta-=np.rint(delta)
    assert np.max(np.abs(delta))<1e-8
    diagnostic=dict(configuration_id=row['configuration_id'],parent_id=row['parent_id'],split=row['split'],case=case,
        electronic_steps=len(steps),energy_eV_per_atom=energy/32,volume_A3=volume,
        max_force_norm_eV_per_A=float(np.linalg.norm(force,axis=1).max()),
        net_force_norm_eV_per_A=float(np.linalg.norm(force.sum(axis=0))),
        max_abs_stress_eV_per_A3=float(np.max(np.abs(stress))),
        stress_antisymmetry_kbar=float(np.max(np.abs(pressure-pressure.T))))
    return label,atoms,diagnostic


def main():
    caldata,calbackup=load_archive(CAL); labdata,labbackup=load_archive(LAB)
    cm=json.loads((CAL/'manifest.json').read_text()); lm=json.loads((LAB/'manifest.json').read_text())
    for root,m,data in ((CAL,cm,caldata),(LAB,lm,labdata)):
        assert m['scheduler_record'][1:3]==['COMPLETED','0:0']
        assert json.loads(data['result.json'])['passed']
        for name,h in m['input_sha256'].items(): assert sha(data[name])==h
    cfg=json.loads(caldata['system.json']); assert caldata['system.json']==labdata['system.json']
    frozen=json.loads(caldata['dataset-manifest.json'])
    assert sha(caldata['dataset-manifest.json'])==cfg['frozen_dataset_manifest_sha256']
    assert caldata['dataset-manifest.json']==labdata['dataset-manifest.json']==(ROOT/'datasets/al32-pilot-v1/dataset-manifest.json').read_bytes()
    rows={r['configuration_id']:r for r in frozen['configurations']}
    assert len(rows)==40 and len({r['poscar_sha256'] for r in rows.values()})==40
    parents={}
    for row in rows.values(): assert parents.setdefault(row['parent_id'],row['split'])==row['split']
    counts={s:sum(r['split']==s for r in rows.values()) for s in ('train','validation','test')}
    assert counts=={'train':21,'validation':6,'test':13}
    labels_by_parent={}; caldiagnostics=[]
    for cid in cfg['calibration_configuration_ids']:
        assert rows[cid]['split']=='train'
        labels_by_parent[cid]={}
        for case,cut,k,sigma in cfg['calibration_cases']:
            label,_,diag=raw_label(caldata,CAL,rows[cid],cid+'-'+case,cut,k,sigma,cfg,cm)
            labels_by_parent[cid][case]=label; caldiagnostics.append(diag)
    stored=json.loads(caldata['calibration.json'])
    assert stored['code_sha256']==sha(caldata['al-pilot.py'])==sha(labdata['al-pilot.py'])
    assert stored['system_sha256']==sha(caldata['system.json'])
    def checks(pairs):
        result=[]
        for cid,labels in labels_by_parent.items():
            for left,right in pairs:
                diff=differences(labels[left],labels[right])
                result.append(dict(configuration_id=cid,left=left,right=right,differences=diff,
                    passed=all(v<=cfg['acceptance'][key] for key,v in diff.items())))
        return result
    selection=cfg['calibration_selection']; refs=checks(selection['reference_checks'])
    assert refs==stored['reference_checks'] and all(c['passed'] for c in refs)
    candidates=[]
    for candidate in selection['candidates']:
        cc=checks(candidate['case_pairs'])
        candidates.append(dict(id=candidate['id'],protocol=candidate['protocol'],checks=cc,passed=all(c['passed'] for c in cc)))
    assert candidates==stored['candidates']
    chosen=next(c for c in candidates if c['passed'])
    assert chosen['id']==stored['selected_candidate'] and chosen['protocol']==stored['protocol'] and stored['passed']
    protocol=chosen['protocol']
    provenance=json.loads(labdata['calibration-source.json'])
    assert provenance['sha256']==sha(caldata['calibration.json'])
    bulk=json.loads(labdata['labels.json']); assert bulk['complete'] and bulk['source']=='DFT/VASP'
    assert bulk['contract']==cfg['label_contract'] and len(bulk['labels'])==40
    labeled=[]; atoms_by_id={}; diagnostics=[]
    for row in rows.values():
        label,atoms,diag=raw_label(labdata,LAB,row,row['configuration_id'],protocol['encut_eV'],protocol['gamma_mesh'],protocol['sigma_eV'],cfg,lm)
        labeled.append(label); atoms_by_id[row['configuration_id']]=atoms; diagnostics.append(diag)
    assert labeled==bulk['labels']
    export=json.loads(labdata['export-manifest.json'])
    assert export['labels_sha256']==sha(labdata['labels.json']) and export['dataset_manifest_sha256']==sha(labdata['dataset-manifest.json'])
    for name,h in export['files'].items(): assert sha(labdata[name])==h
    maxima={k:0.0 for k in ('energy_eV','force_eV_per_A','stress_eV_per_A3','virial_eV','position_A','cell_A')}
    by_id={l['configuration_id']:l for l in labeled}
    for split in counts:
        expected=[r['configuration_id'] for r in rows.values() if r['split']==split]
        base='export/deepmd/'+split+'/'
        assert json.loads(labdata[base+'frame-ids.json'])==expected
        assert labdata[base+'type_map.raw'].decode().split()==['Al']
        assert list(map(int,labdata[base+'type.raw'].decode().split()))==[0]*32
        xyz=read(io.StringIO(labdata['export/'+split+'.extxyz'].decode()),format='extxyz',index=':')
        arrays={name:np.load(io.BytesIO(labdata[base+'set.000/'+name+'.npy']),allow_pickle=False) for name in ('energy','coord','box','force','virial')}
        assert arrays['energy'].shape==(len(expected),)
        for name,width in (('coord',96),('box',9),('force',96),('virial',9)):
            assert arrays[name].shape==(len(expected),width)
        assert len(xyz)==len(expected) and all(a.dtype==np.float64 and np.isfinite(a).all() for a in arrays.values())
        for i,(cid,ex) in enumerate(zip(expected,xyz)):
            label=by_id[cid]; atoms=atoms_by_id[cid]
            assert ex.info['configuration_id']==cid and ex.info['parent_id']==rows[cid]['parent_id'] and ex.info['label_source']=='DFT_VASP'
            assert ex.get_chemical_symbols()==atoms.get_chemical_symbols()==['Al']*32
            assert all(ex.pbc) and all(atoms.pbc)
            f=np.array(label['forces_eV_per_A']); s=np.array(label['stress_eV_per_A3'])
            tensor=np.array([[s[0],s[5],s[4]],[s[5],s[1],s[3]],[s[4],s[3],s[2]]])
            errors=dict(energy_eV=max(abs(arrays['energy'][i]-label['energy_free_eV']),abs(ex.get_potential_energy()-label['energy_free_eV'])),
                force_eV_per_A=max(float(np.max(np.abs(arrays['force'][i].reshape(32,3)-f))),float(np.max(np.abs(ex.get_forces()-f)))),
                stress_eV_per_A3=float(np.max(np.abs(ex.get_stress()-s))),
                virial_eV=float(np.max(np.abs(arrays['virial'][i].reshape(3,3)+atoms.get_volume()*tensor))),
                position_A=max(float(np.max(np.abs(arrays['coord'][i].reshape(32,3)-atoms.positions))),float(np.max(np.abs(ex.positions-atoms.positions)))),
                cell_A=max(float(np.max(np.abs(arrays['box'][i].reshape(3,3)-atoms.cell.array))),float(np.max(np.abs(ex.cell.array-atoms.cell.array)))))
            for key,v in errors.items(): maxima[key]=max(maxima[key],v)
    assert maxima['energy_eV']<1e-7 and maxima['force_eV_per_A']<1e-7 and maxima['stress_eV_per_A3']<1e-8
    assert maxima['virial_eV']<1e-8 and maxima['position_A']<1e-8 and maxima['cell_A']<1e-8
    repeated={cid:differences(by_id[cid],labels_by_parent[cid]['candidate'+str(protocol['gamma_mesh'])]) for cid in labels_by_parent}
    budget=[]
    for p in sorted((ROOT/'runs').glob('al32-*/manifest.json')):
        m=json.loads(p.read_text()); record=m['scheduler_record']
        assert record[1] in ('COMPLETED','FAILED','CANCELLED')
        budget.append(dict(run_id=m['run_id'],job_id=m['job_id'],state=record[1],wall_seconds=int(record[3]),allocated_CPU_seconds=int(record[5]),allocated_CPU_hours=int(record[5])/3600))
    total=sum(r['allocated_CPU_seconds'] for r in budget); assert total<=128*3600
    scheduler=json.loads((LAB/'scheduler-final.json').read_text())
    assert {x['job_id'] for x in scheduler['jobs']}=={x['job_id'] for x in budget}
    for row in budget:
        observed=next(x for x in scheduler['jobs'] if x['job_id']==row['job_id'])
        assert observed['state']==row['state'] and observed['allocated_CPU_seconds']==row['allocated_CPU_seconds']
    selected_checks=chosen['checks']
    worst_force=max(c['differences']['max_force_component_eV_per_A'] for c in selected_checks)
    assessment=dict(assessed_utc=datetime.now(timezone.utc).isoformat(),analysis_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),analysis_source_sha256=sha(Path(__file__).read_bytes()),
      calibration_passed=True,calibration_points_verified=10,protocol=protocol,reference_checks=refs,candidate_results=candidates,
      force_tolerance_margin_eV_per_A=cfg['acceptance']['max_force_component_eV_per_A']-worst_force,
      production_points_verified=40,split_counts=counts,parent_groups=len(parents),all_electronically_converged=True,
      all_raw_hashes_and_label_provenance_verified=True,exports_verified=True,export_maximum_differences=maxima,
      ready_for_pilot_finetuning=True,ready_for_production_MD=False,
      production_ranges=dict(electronic_steps=[min(d['electronic_steps'] for d in diagnostics),max(d['electronic_steps'] for d in diagnostics)],energy_eV_per_atom=[min(d['energy_eV_per_atom'] for d in diagnostics),max(d['energy_eV_per_atom'] for d in diagnostics)],max_force_norm_eV_per_A=max(d['max_force_norm_eV_per_A'] for d in diagnostics),max_net_force_norm_eV_per_A=max(d['net_force_norm_eV_per_A'] for d in diagnostics),max_abs_stress_eV_per_A3=max(d['max_abs_stress_eV_per_A3'] for d in diagnostics)),
      repeated_train_points_differences=repeated,budget=dict(jobs=budget,total_CPU_seconds=total,total_CPU_hours=total/3600,cap_CPU_hours=128,remaining_CPU_hours=128-total/3600),
      backups={CAL.name:calbackup,LAB.name:labbackup},dataset_manifest_sha256=sha(labdata['dataset-manifest.json']),labels_sha256=sha(labdata['labels.json']),export_manifest_sha256=sha(labdata['export-manifest.json']),
      limitations=['Convergence calibrated on only two training configurations','Force criterion passes narrowly; no claim of a large numerical margin','EMT proposals do not establish equilibrium DFT liquid; no interfaces/segregation/nucleation validation','Potential pretraining overlap and PAW54/PAW64 alignment remain to be handled in model evaluation','Private main-home plus workstation backups verified; institutional retention unconfirmed'])
    save(LAB/'quality-assessment.json',assessment)
    save(CAL/'quality-assessment.json',{k:assessment[k] for k in ('analysis_code_commit','analysis_source_sha256','calibration_passed','calibration_points_verified','protocol','reference_checks','candidate_results','force_tolerance_margin_eV_per_A')})
    for directory,name,ds in ((CAL,'calibration-diagnostics.csv',caldiagnostics),(LAB,'label-diagnostics.csv',diagnostics)):
        with (directory/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(ds[0])); w.writeheader(); w.writerows(ds)
    print(json.dumps({k:assessment[k] for k in ('calibration_passed','protocol','force_tolerance_margin_eV_per_A','production_points_verified','split_counts','export_maximum_differences','production_ranges','repeated_train_points_differences','budget')},indent=2))


if __name__=='__main__': main()
