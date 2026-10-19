"""Finite Slurm kpLib installation, generation and paired VASP benchmark."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

ROOT=Path.cwd()
CFG=json.loads((ROOT/'system.json').read_text())


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,obj):
    with Path(path).open('w',encoding='utf-8',newline='\n') as f:f.write(json.dumps(obj,indent=2,sort_keys=True)+'\n')


def setup():
    prefix=Path(CFG['kplib']['environment'])
    if prefix.exists():raise FileExistsError('Preserve existing environment: '+str(prefix))
    home=ROOT/'setup-home';home.mkdir()
    os.environ.update(HOME=str(home),PIP_CACHE_DIR=str(home/'pip-cache'),XDG_CACHE_HOME=str(home/'cache'))
    subprocess.run([sys.executable,'-m','venv',str(prefix)],check=True)
    py=str(prefix/'bin/python'); wheels=ROOT/'wheelhouse';wheels.mkdir()
    pins=['numpy==1.26.4','spglib==2.7.0','pybind11==2.13.6','setuptools==75.8.0','wheel==0.45.1','typing_extensions==4.14.1','packaging==25.0']
    subprocess.run([py,'-m','pip','download','--only-binary=:all:','--no-deps','--dest',str(wheels)]+pins,check=True)
    subprocess.run([py,'-m','pip','install','--no-index','--find-links',str(wheels),'--no-deps']+pins,check=True)
    source=ROOT/'kpLib-1.1.1.tar.gz'
    assert sha(source)==CFG['kplib']['source_sha256']
    subprocess.run([py,'-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',str(wheels),str(source)],check=True)
    built=next(wheels.glob('kp*1.1.1*.whl'))
    subprocess.run([py,'-m','pip','install','--no-deps',str(built)],check=True)
    subprocess.run([py,'-m','pip','check'],check=True)
    versions=subprocess.check_output([py,'-m','pip','freeze','--all'],text=True)
    compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0]
    record=dict(python=sys.version,base_interpreter=sys.executable,platform=platform.platform(),environment=str(prefix),
        installed_packages=versions.splitlines(),compiler=compiler,source_sha256=sha(source),
        wheel_files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in wheels.iterdir() if p.is_file()})
    save(ROOT/'software.json',record)
    (ROOT/'requirements.lock.txt').write_text('\n'.join(p.name+' '+sha(p) for p in sorted(wheels.glob('*.whl')))+'\n')
    (prefix/'activate-kplib.sh').write_text('source '+str(prefix/'bin/activate')+'\nexport PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1\n')
    # Smoke test imports and both old interface dependencies in the new prefix.
    subprocess.run([py,'-c','from kpLib.interface import get_kpoints; import spglib,numpy; print(numpy.__version__,spglib.__version__)'],check=True)


def grids():
    import numpy as np
    import importlib.metadata
    import kpLib.lib
    from kplib_grid import generate,reduce_time_reversal,kpoints
    save(ROOT/'runtime.json',dict(python=sys.version,kplib=importlib.metadata.version('kpLib'),
        numpy=np.__version__,spglib=importlib.metadata.version('spglib'),
        extension_path=kpLib.lib.__file__,extension_sha256=sha(kpLib.lib.__file__)))
    # Direct representation tests, including self-inverse and shifted points.
    for n,shift,expected in ((3,0,14),(2,0,8),(2,0.5,4)):
        coords=np.array([[(i+shift)/n,(j+shift)/n,(k+shift)/n] for i in range(n) for j in range(n) for k in range(n)])
        tr,w=reduce_time_reversal(coords);assert len(tr)==expected and sum(w)==n**3
        for vector in ((1,2,3),(4,-2,1)):
            full=np.cos(2*np.pi*coords.dot(vector)).mean()
            reduced=np.average(np.cos(2*np.pi*np.array(tr).dot(vector)),weights=w)
            assert abs(full-reduced)<1e-12
    refs=json.loads((ROOT/'references.json').read_text());result={}
    for cid,ref in refs.items():
        result[cid]={}
        for distance in CFG['kplib']['min_distance_A']:
            grid=generate(ROOT/ref['row']['poscar'],distance,CFG['kplib'])
            path=ROOT/'grids'/cid/str(distance);path.mkdir(parents=True)
            (path/'KPOINTS').write_text(kpoints(grid['tr_coords'],grid['tr_weights'],f'kpLib d={distance}A time reversal only'))
            save(path/'grid.json',grid)
            result[cid][str(distance)]=grid
            save(ROOT/'grids.json',result)
    save(ROOT/'grid-summary.json',{cid:{d:{k:v for k,v in g.items() if 'coords' not in k and 'weights' not in k} for d,g in grids.items()} for cid,grids in result.items()})


def benchmark():
    import numpy as np
    import xml.etree.ElementTree as ET
    from kplib_grid import kpoints
    spec=importlib.util.spec_from_file_location('vasp_api',ROOT/'al-pilot.py'); api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    refs=json.loads((ROOT/'references.json').read_text());grids=json.loads((ROOT/'grids.json').read_text())
    program=Path(shutil.which('vasp_std'))
    save(ROOT/'runtime.json',dict(vasp_executable=str(program),vasp_sha256=sha(program),
        python=sys.version,host=platform.node(),ntasks=int(os.environ['SLURM_NTASKS']),
        reference_labels_sha256=sha(ROOT/'references.json'),grids_sha256=sha(ROOT/'grids.json')))
    cases={};root=ROOT/'calculations';root.mkdir()
    def run(cid,name,grid=None,full=False):
        path=root/(cid+'-'+name);path.mkdir()
        row=refs[cid]['row'];(path/'POSCAR').write_bytes((ROOT/row['poscar']).read_bytes())
        assert sha(path/'POSCAR')==row['poscar_sha256']
        (path/'INCAR').write_text(api.incar(450,0.05,cid))
        if grid is None: content='Fresh Gamma7 baseline\n0\nGamma\n7 7 7\n0 0 0\n'
        else: content=kpoints(grid['full_coords'] if full else grid['tr_coords'],grid['full_weights'] if full else grid['tr_weights'],name)
        (path/'KPOINTS').write_text(content)
        subprocess.run([sys.executable,os.environ['VASP_PAW_ROOT']+'/vasp_potentials.py','assemble',os.environ['VASP_PAW_REGISTRY'],'--family','PBE','--variants','Al','--output',str(path/'POTCAR')],check=True)
        assert sha(path/'POTCAR')=='1b07010218b572e3404ce1eea8fed39e24b52859ebef1517bffa2ef69b8d1315'
        save(path/'input-manifest.json',dict(configuration_id=cid,parent_id=row['parent_id'],split='train',job_id=os.environ['SLURM_JOB_ID'],code_commit=os.environ['PILOT_CODE_COMMIT'],input_sha256={n:sha(path/n) for n in ('POSCAR','INCAR','KPOINTS','POTCAR')}))
        started=time.perf_counter()
        with (path/'vasp.stdout').open('w') as f:
            subprocess.run(['mpirun','-np',os.environ['SLURM_NTASKS'],'vasp_std'],cwd=path,stdout=f,stderr=subprocess.STDOUT,check=True)
        elapsed=time.perf_counter()-started
        label=api.parse_label(path)
        xml=ET.parse(path/'vasprun.xml')
        label.update(configuration_id=cid,case=name,parent_id=row['parent_id'],split='train',wall_seconds=elapsed,
            actual_kpoints=len(xml.findall("kpoints/varray[@name='kpointlist']/v")),job_id=os.environ['SLURM_JOB_ID'],code_commit=os.environ['PILOT_CODE_COMMIT'],host=platform.node())
        save(path/'label.json',label);cases.setdefault(cid,{})[name]=label
        save(ROOT/'benchmark-progress.json',dict(complete=False,cases=cases))
        return label
    def diff(a,b):return api.differences(a,b)
    def passes(a,b,tol):return all(v<=tol[k] for k,v in diff(a,b).items())
    distances=CFG['kplib']['min_distance_A']; anchor=CFG['kplib']['anchor_min_distance_A']
    # Opposite orders reduce systematic drift; every run starts without WAVECAR.
    for rep in range(CFG['timing']['repeats_per_method_per_parent']):
        for cid in (list(refs) if rep%2==0 else list(refs)[::-1]):run(cid,'gamma7-r'+str(rep))
    for di,d in enumerate([anchor]+[d for d in distances if d!=anchor]):
        for cid in (list(refs) if di%2==0 else list(refs)[::-1]):run(cid,'d'+str(d),grids[cid][str(d)])
    checks=[]
    for d in distances:
        for cid in refs:
            label=cases[cid]['d'+str(d)]
            checks.append(dict(min_distance_A=d,configuration_id=cid,versus_gamma8=diff(label,refs[cid]['label']),
                versus_anchor=diff(label,cases[cid]['d'+str(anchor)]),
                passed=passes(label,refs[cid]['label'],CFG['tolerances']) and passes(label,cases[cid]['d'+str(anchor)],CFG['tolerances'])))
    accepted=[d for d in distances if d<anchor and all(c['passed'] for c in checks if c['min_distance_A']>=d)]
    chosen=min(accepted) if accepted else None
    timing={};representation={}
    if chosen is not None:
        for cid in list(refs)[::-1]:run(cid,'selected-repeat',grids[cid][str(chosen)])
        for cid in refs:
            full=run(cid,'selected-full',grids[cid][str(chosen)],full=True)
            reduced=cases[cid]['d'+str(chosen)]
            representation[cid]=dict(differences=diff(full,reduced),passed=passes(full,reduced,CFG['representation_tolerances']),full_actual_kpoints=full['actual_kpoints'],reduced_actual_kpoints=reduced['actual_kpoints'])
            baseline=[cases[cid]['gamma7-r'+str(i)]['wall_seconds'] for i in range(2)]
            selected=[reduced['wall_seconds'],cases[cid]['selected-repeat']['wall_seconds']]
            bm=statistics.median(baseline);km=statistics.median(selected);generation=grids[cid][str(chosen)]['generation_seconds']
            timing[cid]=dict(gamma7_wall_samples_seconds=baseline,kplib_wall_samples_seconds=selected,
                gamma7_median_seconds=bm,kplib_median_seconds=km,grid_generation_seconds=generation,
                end_to_end_speedup=bm/(km+generation),at_least10percent_faster=km+generation<=0.9*bm,
                gamma7_actual_kpoints=cases[cid]['gamma7-r0']['actual_kpoints'],kplib_actual_kpoints=reduced['actual_kpoints'],
                baseline_error=diff(cases[cid]['gamma7-r0'],refs[cid]['label']),kplib_error=diff(reduced,refs[cid]['label']))
    baseline_ok=all(passes(cases[cid]['gamma7-r0'],refs[cid]['label'],CFG['tolerances']) for cid in refs)
    recommend=chosen is not None and baseline_ok and all(x['passed'] for x in representation.values()) and all(t['at_least10percent_faster'] for t in timing.values())
    assessment=dict(complete=True,configuration_ids=list(refs),chosen_min_distance_A=chosen,checks=checks,
        baseline_accuracy_passed=baseline_ok,timing=timing,representation_checks=representation,
        recommend_for_this_pilot=recommend,
        decision='Pilot adoption supported by predefined accuracy and paired cost gates' if recommend else 'Retain validated Gamma7; generalized-grid adoption not supported by all predefined gates',
        limitations=['Two train parents only; two cold repeats per timed method','Reference Gamma8 and d80 are finite-grid local references','No universal optimality or production throughput claim'],cases=cases)
    save(ROOT/'benchmark.json',assessment)


if __name__=='__main__':
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run on Slurm compute node')
    phase=sys.argv[1];result=dict(phase=phase,job_id=os.environ['SLURM_JOB_ID'],passed=False)
    try:
        {'setup':setup,'grids':grids,'benchmark':benchmark}[phase]()
        result['passed']=True
    except Exception as e:
        result['error']=repr(e);raise
    finally:save(ROOT/'result.json',result)
