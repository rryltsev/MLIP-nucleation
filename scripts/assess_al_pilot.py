"""Audit archived Al32 VASP data without recomputing DFT or extracting POTCAR.

Uses only stdlib; all reported comparisons derive from checksum-verified raw XML.
The diagnostic calibration labels are never exported as a training dataset.
"""
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import tarfile
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
CAL=ROOT/'runs/al32-calibration-20261018-01'
LABEL=ROOT/'runs/al32-labeling-20261018-01'


def sha(data): return hashlib.sha256(data).hexdigest()


def save(path,obj):
    with path.open('w',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(obj,indent=2,sort_keys=True)+'\n')


def compare(a,b):
    return dict(energy_eV_per_atom=abs(a['energy_free_eV']-b['energy_free_eV'])/32,
      max_force_component_eV_per_A=max(abs(x-y) for u,v in zip(a['forces_eV_per_A'],b['forces_eV_per_A']) for x,y in zip(u,v)),
      max_stress_component_eV_per_A3=max(abs(x-y) for x,y in zip(a['stress_eV_per_A3'],b['stress_eV_per_A3'])))


def main():
    cfg=json.loads((ROOT/'datasets/al32-pilot-v1/system.json').read_text())
    manifest=json.loads((CAL/'manifest.json').read_text())
    frozen=json.loads((ROOT/'datasets/al32-pilot-v1/dataset-manifest.json').read_text())
    rows={r['configuration_id']:r for r in frozen['configurations']}
    calibration=json.loads((CAL/'calibration.json').read_text())
    backups={}
    members_by_run={}
    for directory in (CAL,LABEL):
        record=json.loads((directory/'artifact-backup.json').read_text())
        archive=Path(record['local_archive'])
        if not archive.is_file():
            archive=ROOT/'artifacts/dft-run-backups'/Path(record['remote_archive']).name
        assert record['verified_retrieval'] and sha(archive.read_bytes())==record['sha256']
        with tarfile.open(archive) as t:
            members=t.getmembers()
            assert all(m.isfile() for m in members)
            assert len({m.name for m in members})==len(members)
            assert {m.name for m in members}==set(record['members'])
            data={m.name:t.extractfile(m).read() for m in members}
        for name,content in data.items():
            assert sha(content)==record['members'][name]['sha256']
        members_by_run[directory.name]=data
        backups[directory.name]={k:record[k] for k in ('sha256','bytes','remote_archive','local_archive','verified_retrieval')}
        backups[directory.name]['members_verified']=len(data)
    data=members_by_run[CAL.name]
    assert sha(data['dataset-manifest.json'])==manifest['input_sha256']['dataset-manifest.json']
    assert sha(data['al-pilot.py'])==manifest['input_sha256']['al-pilot.py']
    labels={}
    diagnostics=[]
    for cid in cfg['calibration_configuration_ids']:
        assert rows[cid]['split']=='train'
        for case,encut,k,sigma in cfg['calibration_cases']:
            name=cid+'-'+case
            base='calculations/'+name+'/'
            label=json.loads(data[base+'label.json'])
            assert (CAL/base/'label.json').read_bytes()==data[base+'label.json']
            assert label['job_id']==manifest['job_id'] and label['code_commit']==manifest['code_commit']
            assert label['configuration_id']==cid and label['split']=='train'
            assert label['parent_id']==rows[cid]['parent_id']
            for filename,h in label['source_hashes'].items(): assert sha(data[base+filename])==h
            assert label['source_hashes']['POTCAR']==cfg['reference_protocol']['potcar_sha256']
            assert label['source_hashes']['POSCAR']==rows[cid]['poscar_sha256']
            inputs=json.loads(data[base+'input-manifest.json'])
            for filename,h in inputs['input_sha256'].items(): assert sha(data[base+filename])==h
            provenance=json.loads(data[base+'potcar-provenance.json'])
            assert provenance['family']=='PBE' and provenance['ordered_potentials'][0]['variant']=='Al'
            assert provenance['potcar_sha256']==label['source_hashes']['POTCAR']
            incar=dict(line.split('=',1) for line in data[base+'INCAR'].decode().splitlines())
            incar={x.strip():y.strip() for x,y in incar.items()}
            assert float(incar['ENCUT'])==encut and float(incar['SIGMA'])==sigma
            assert [int(x) for x in data[base+'KPOINTS'].decode().splitlines()[3].split()]==[k]*3
            xml=ET.fromstring(data[base+'vasprun.xml'])
            calcs=xml.findall('calculation'); assert len(calcs)==1
            calc=calcs[0]
            pars={e.get('name'):(e.text or '').strip() for e in xml.findall('parameters//i')}
            assert int(pars['NSW'])==0 and int(pars['ISPIN'])==1 and int(pars['ISYM'])==0
            # VASP6.6 reports the effective cutoff as parameters/ENMAX;
            # the explicitly requested ENCUT remains under incar/ENCUT.
            requested={e.get('name'):(e.text or '').strip() for e in xml.findall('incar//i')}
            assert float(requested['ENCUT'])==encut and float(pars['ENMAX'])==encut
            assert float(pars['SIGMA'])==sigma
            assert float(pars['EDIFF'])==1e-7 and int(pars['ISMEAR'])==0
            steps=calc.findall('scstep')
            assert len(steps)==label['electronic_steps'] and 0<len(steps)<int(pars['NELM'])
            out=data[base+'OUTCAR'].decode(errors='replace')
            assert 'aborting loop because EDIFF is reached' in out
            assert 'General timing and accounting informations for this job' in out
            energy=float(calc.find("energy/i[@name='e_fr_energy']").text)
            assert abs(energy-label['energy_free_eV'])<1e-8
            last=steps[-1].find('energy')
            sigma0=energy+float(last.find("i[@name='e_0_energy']").text)-float(last.find("i[@name='e_fr_energy']").text)
            assert abs(sigma0-label['energy_sigma0_eV'])<1e-8
            species=[e[0].text.strip() for e in xml.findall("atominfo/array[@name='atoms']/set/rc")]
            assert species==['Al']*32
            basis=[[float(x) for x in e.text.split()] for e in calc.findall("structure/crystal/varray[@name='basis']/v")]
            a,b,c=basis
            volume=a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
            assert math.isclose(volume,label['volume_A3'],rel_tol=1e-9,abs_tol=1e-6)
            force=[[float(x) for x in e.text.split()] for e in calc.findall("varray[@name='forces']/v")]
            assert len(force)==32 and all(len(f)==3 for f in force)
            assert max(abs(x-y) for f,g in zip(force,label['forces_eV_per_A']) for x,y in zip(f,g))<1e-10
            pressure=[[float(x) for x in e.text.split()] for e in calc.findall("varray[@name='stress']/v")]
            # Independent SI conversion, exact 2019 elementary-charge constant.
            stress=[-pressure[i][j]*1e8/1.602176634e11 for i,j in ((0,0),(1,1),(2,2),(1,2),(0,2),(0,1))]
            assert all(math.isclose(x,y,rel_tol=1e-7,abs_tol=1e-10) for x,y in zip(stress,label['stress_eV_per_A3']))
            numbers=[energy,label['volume_A3']]+[x for f in force for x in f]+stress
            assert all(math.isfinite(x) for x in numbers) and label['volume_A3']>0
            net=[sum(f[j] for f in force) for j in range(3)]
            diagnostics.append(dict(name=name,configuration_id=cid,case=case,encut_eV=encut,gamma_mesh=k,sigma_eV=sigma,
                electronic_steps=len(steps),energy_eV_per_atom=energy/32,
                maximum_force_norm_eV_per_A=max(math.sqrt(sum(x*x for x in f)) for f in force),
                net_force_norm_eV_per_A=math.sqrt(sum(x*x for x in net)),
                pressure_tensor_antisymmetry_kbar=max(abs(pressure[i][j]-pressure[j][i]) for i in range(3) for j in range(3)),
                wall_seconds=label['wall_seconds']))
            labels[name]=label
    assert len(labels)==16
    expected=[]
    for cid in cfg['calibration_configuration_ids']:
        for case in cfg['acceptance']['required_against_reference']:
            diff=compare(labels[cid+'-'+case],labels[cid+'-reference'])
            expected.append(dict(configuration_id=cid,case=case,differences=diff,
                 passed=all(v<=cfg['acceptance'][k] for k,v in diff.items())))
    assert expected==calibration['checks']
    assert calibration['passed']==all(c['passed'] for c in expected)
    cancelled_data=members_by_run[LABEL.name]
    assert not any(n.startswith('calculations/') for n in cancelled_data)
    assert 'labels.json' not in cancelled_data and 'export-manifest.json' not in cancelled_data
    lm=json.loads((LABEL/'manifest.json').read_text())
    assert lm['scheduler_record'][1]=='CANCELLED' and int(lm['scheduler_record'][5])==0
    cpu_seconds=int(manifest['scheduler_record'][5])+int(lm['scheduler_record'][5])
    assessment=dict(assessed_utc=datetime.now(timezone.utc).isoformat(),
      analysis_source_sha256=sha(Path(__file__).read_bytes()),
      analysis_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      raw_label_quality_audit_passed=True,checks_recomputed_from_verified_raw_outputs=True,
      electronically_converged_calibration_points=len(labels),
      calibration_protocol_accepted=calibration['passed'],
      gate_checks=expected,failed_gate_checks=sum(not c['passed'] for c in expected),
      production_labels=0,ready_for_finetuning=False,
      decision='Reject draft 450eV / Gamma4 / sigma0.05 protocol. Retain calibration diagnostics only; do not merge settings or fit models.',
      next_calibration='Test denser k meshes on the same train-only parents with fixed sigma and unchanged tolerances; Gamma5 is not established as converged by this run.',
      budget=dict(allocated_CPU_seconds=cpu_seconds,allocated_CPU_hours=cpu_seconds/3600,
                  maximum_authorized_CPU_hours=128,new_jobs_submitted_during_collection=0),
      backups=backups,
      ranges=dict(electronic_steps=[min(x['electronic_steps'] for x in diagnostics),max(x['electronic_steps'] for x in diagnostics)],
         energy_eV_per_atom=[min(x['energy_eV_per_atom'] for x in diagnostics),max(x['energy_eV_per_atom'] for x in diagnostics)],
         maximum_force_norm_eV_per_A=max(x['maximum_force_norm_eV_per_A'] for x in diagnostics),
         maximum_net_force_norm_eV_per_A=max(x['net_force_norm_eV_per_A'] for x in diagnostics),
         maximum_pressure_antisymmetry_kbar=max(x['pressure_tensor_antisymmetry_kbar'] for x in diagnostics)),
      limitations=['Only two training-parent configurations were calibrated','No production labels; no DFT equilibrium liquid or interfacial validation','Private main-home and workstation copies verified; institutional backup/retention unconfirmed'])
    save(CAL/'quality-assessment.json',assessment)
    with (CAL/'calibration-diagnostics.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(diagnostics[0])); writer.writeheader(); writer.writerows(diagnostics)
    print(json.dumps({k:v for k,v in assessment.items() if k not in ('gate_checks','backups')},indent=2))


if __name__=='__main__': main()
