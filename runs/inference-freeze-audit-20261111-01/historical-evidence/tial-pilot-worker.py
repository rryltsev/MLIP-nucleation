"""Development and harmonic evaluation of the fixed final checkpoint; test remains closed."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
import numpy as np
from ase.io import read
from deepmd.infer import DeepPot
ROOT=Path(__file__).resolve().parent.parent;WORK=Path.cwd()
sys.path.insert(0,str(ROOT))
from baseline_worker import evaluate,differences,dump,sha
def main():
    assert os.environ.get('SLURM_JOB_ID')
    s=json.loads((WORK/'training-summary.json').read_text());assert s['status']=='PASS'
    model=s['final_checkpoint'];xc=s['case']['XC'];dp=DeepPot(model);out=WORK/'evaluation';out.mkdir()
    with (out/'freeze.log').open('w') as f:subprocess.run([str(Path(sys.executable).parent/'dp'),'--pt','freeze','-c',model,'-o',str(out/'final')],stdout=f,stderr=subprocess.STDOUT,check=True)
    export=DeepPot(str(out/'final.pt2'));checks=[]
    for p in json.loads((ROOT/'probes.json').read_text()):
        d=differences(evaluate(dp,p['positions'],p['cell'],p['symbols']),evaluate(export,p['positions'],p['cell'],p['symbols']),len(p['symbols']))
        assert d['energy_eV_atom']<1e-5 and d['force_max_eV_A']<2e-4 and d['virial_max_eV']<.02,d
        checks.append(dict(id=p['id'],**d))
    del export
    for part in ['train','validation']:
        target=out/'predictions'/xc/part;target.mkdir(parents=True)
        for src in sorted((ROOT/'data'/xc/part).iterdir()):
            ids=json.loads((src/'geometry_ids.json').read_text());types=np.loadtxt(src/'type.raw',dtype=int,ndmin=1);names=(src/'type_map.raw').read_text().split();symbols=[names[t] for t in types]
            es=[];fs=[];vs=[]
            for c,b in zip(np.load(src/'set.000/coord.npy'),np.load(src/'set.000/box.npy')):
                e,f,v=evaluate(dp,c,b,symbols);es.append(e);fs.append(f);vs.append(v)
            np.savez_compressed(target/(src.name+'.npz'),ids=ids,energy=es,force=fs,virial=vs)
    for key in ['force000','force001','force002']:
        a=read(ROOT/'harmonic_inputs'/xc/key/'POSCAR',format='vasp');e,f,v=evaluate(dp,a.positions,a.cell.array,a.get_chemical_symbols())
        target=out/'harmonic_forces'/key;target.mkdir(parents=True);np.save(target/'forces.npy',f)
        dump(target/'prediction.json',dict(energy_eV=e,virial_eV=v.tolist()))
    dump(out/'summary.json',dict(status='PASS',model_sha256=sha(model),export_sha256=sha(out/'final.pt2'),export_checks=checks,train_evaluated=200,validation_evaluated=149,test_access=False,harmonic_geometry='same XC-specific DFT geometry',job_id=os.environ['SLURM_JOB_ID']))
if __name__=='__main__':main()
