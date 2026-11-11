"""Slurm-only, native/export and differential checks; fixed original-model evaluation."""
from pathlib import Path
import hashlib, importlib.metadata, json, os, subprocess, sys, time
import numpy as np
from ase.io import read
from deepmd.infer import DeepPot

ROOT=Path(__file__).resolve().parent
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+'\n')
def evaluate(dp,c,b,s):
    e,f,v=dp.eval(np.asarray(c).reshape(1,-1),np.asarray(b).reshape(1,9),[dp.get_type_map().index(x) for x in s])[:3]
    e=float(e.ravel()[0]);f=f.reshape(-1,3);v=v.reshape(3,3)
    assert np.isfinite(e) and np.isfinite(f).all() and np.isfinite(v).all()
    return e,f,v
def differences(a,b,n):return dict(energy_eV_atom=abs(a[0]-b[0])/n,force_max_eV_A=float(np.max(np.abs(a[1]-b[1]))),virial_max_eV=float(np.max(np.abs(a[2]-b[2]))))
def main():
    assert os.environ.get('SLURM_JOB_ID')
    cfg=json.loads((ROOT/'protocol.json').read_text());out=ROOT/'baseline';out.mkdir(exist_ok=False)
    for name,h in json.loads((ROOT/'FILES_SHA256.json').read_text()).items():assert sha(ROOT/name)==h,name
    model=cfg['model'];assert sha(model['path'])==model['sha256']
    dp=DeepPot(model['path']);assert dp.get_dim_fparam()==dp.get_dim_aparam()==0
    checks=[];probes=json.loads((ROOT/'probes.json').read_text())
    for probe in probes:
        c=np.array(probe['positions']);b=np.array(probe['cell']);s=probe['symbols'];n=len(s);base=evaluate(dp,c,b,s)
        rng=np.random.default_rng(103);order=rng.permutation(n)
        perm=evaluate(dp,c[order],b,[s[i] for i in order]);perm=(perm[0],perm[1][np.argsort(order)],perm[2])
        shift=c+np.array([.73,-.29,.41])@b
        shift[::3]+=b[1];shift[1::3]-=b[2]
        pbc=evaluate(dp,shift,b,s)
        q,_=np.linalg.qr(rng.normal(size=(3,3)))
        if np.linalg.det(q)<0:q[:,0]*=-1
        rot=evaluate(dp,c@q,b@q,s);rot=(rot[0],rot[1]@q.T,q@rot[2]@q.T)
        sym={k:differences(base,v,n) for k,v in [('permutation',perm),('translation_PBC',pbc),('rotation',rot)]}
        idx=np.unravel_index(np.argmax(np.abs(base[1])),base[1].shape);fd=[]
        for h in [.01,.003,.001]:
            cp=c.copy();cm=c.copy();cp[idx]+=h;cm[idx]-=h
            numeric=-(evaluate(dp,cp,b,s)[0]-evaluate(dp,cm,b,s)[0])/(2*h)
            fd.append(dict(h_A=h,analytic=float(base[1][idx]),numeric=numeric,error=abs(numeric-base[1][idx])))
        vd=[]
        for h in [.003,.001,.0003]:
            plus=np.eye(3);minus=np.eye(3);plus[0,0]+=h;minus[0,0]-=h
            numeric=-(evaluate(dp,c@plus,b@plus,s)[0]-evaluate(dp,c@minus,b@minus,s)[0])/(2*h)
            vd.append(dict(strain=h,analytic=float(base[2][0,0]),numeric=numeric,error=abs(numeric-base[2][0,0])))
        ok=all(v['energy_eV_atom']<1e-5 and v['force_max_eV_A']<2e-4 and v['virial_max_eV']<.01 for v in sym.values()) and min(v['error'] for v in fd)<.01 and min(v['error'] for v in vd)<.1
        checks.append(dict(id=probe['id'],symmetries=sym,force_FD=fd,virial_FD=vd,pass_checks=ok))
        dump(out/'numerical_checks.json',checks)
    # Freeze in the same pinned environment; this is a fresh export of the hashed parent.
    with (out/'freeze.log').open('w') as f:
        subprocess.run([str(Path(sys.executable).parent/'dp'),'--pt','freeze','-c',model['path'],'-o',str(out/'original')],stdout=f,stderr=subprocess.STDOUT,check=True)
    exported=out/'original.pt2';assert exported.exists();edp=DeepPot(str(exported));export=[]
    for p in probes:export.append(dict(id=p['id'],**differences(evaluate(dp,p['positions'],p['cell'],p['symbols']),evaluate(edp,p['positions'],p['cell'],p['symbols']),len(p['symbols']))))
    export_ok=all(x['energy_eV_atom']<1e-5 and x['force_max_eV_A']<2e-4 and x['virial_max_eV']<.01 for x in export)
    dump(out/'export_checks.json',dict(export_sha256=sha(exported),differences=export,status='PASS' if export_ok else 'FAIL'))
    del edp
    # Native DFT geometries per XC are preserved (no transfer of rounding errors between XCs).
    for xc in ['PBE','PBEsol','r2SCAN']:
        for part in ['train','validation','test']:
            target=out/'predictions'/xc/part;target.mkdir(parents=True)
            for source in sorted((ROOT/'data'/xc/part).iterdir()):
                ids=json.loads((source/'geometry_ids.json').read_text());types=np.loadtxt(source/'type.raw',dtype=int,ndmin=1)
                names=(source/'type_map.raw').read_text().split();s=[names[t] for t in types]
                coords=np.load(source/'set.000/coord.npy');boxes=np.load(source/'set.000/box.npy');es=[];fs=[];vs=[]
                for c,b in zip(coords,boxes):
                    e,f,v=evaluate(dp,c,b,s);es.append(e);fs.append(f);vs.append(v)
                np.savez_compressed(target/(source.name+'.npz'),ids=ids,energy=es,force=fs,virial=vs)
            print('predicted',xc,part,flush=True)
    harmonic={}
    for xc in ['PBE','PBEsol','r2SCAN']:
        harmonic[xc]={}
        for key in ['force000','force001','force002']:
            p=ROOT/'harmonic_inputs'/xc/key/'POSCAR';a=read(p,format='vasp')
            e,f,v=evaluate(dp,a.positions,a.cell.array,a.get_chemical_symbols())
            target=out/'harmonic_forces'/xc/key;target.mkdir(parents=True);np.save(target/'forces.npy',f)
            harmonic[xc][key]=dict(POSCAR_sha256=sha(p),energy_eV=e,virial_eV=v.tolist(),max_force_eV_A=float(np.linalg.norm(f,axis=1).max()))
    dump(out/'harmonic_predictions.json',harmonic)
    passed=export_ok and all(x['pass_checks'] for x in checks)
    dump(out/'summary.json',dict(status='PASS' if passed else 'FAIL',job_id=os.environ['SLURM_JOB_ID'],model=model,versions={n:importlib.metadata.version(n) for n in ['deepmd-kit','torch','numpy','ase']},worker_sha256=sha(__file__),evaluated_frames_per_XC=498,harmonic_frames=9))
    assert passed,'Numerical gate B0 failed; predictions retained for diagnosis'
if __name__=='__main__':main()
