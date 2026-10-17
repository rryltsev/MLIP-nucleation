"""True restart and checkpoint/export/ASE/LAMMPS equivalence on synthetic structures."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

SOURCE=Path('/home6/u3129/mlip-research/runs/fm-finetune-20261017-01')
TOOLS=Path(__file__).parent
BACKEND=sys.argv[1]
ROOT=Path.cwd()
TOLERANCES={
 'dpa4':{'energy':1e-4,'forces':1e-4,'stress':1e-6,'rtol':1e-4},
 'grace':{'energy':1e-7,'forces':1e-7,'stress':1e-9,'rtol':1e-7}}


def save(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(command,name,seconds=180,env=None,cwd=None,check=True):
    started=time.monotonic(); samples=[]
    with (ROOT/(name+'.log')).open('wb') as stream:
        proc=subprocess.Popen(command,cwd=cwd or ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        while proc.poll() is None:
            if time.monotonic()-started>seconds:
                import signal
                os.killpg(proc.pid,signal.SIGKILL);proc.wait();raise TimeoutError(name+' exceeded '+str(seconds)+' seconds')
            q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
            samples.extend({'seconds':time.monotonic()-started,'process':s} for s in q.stdout.splitlines() if s.split(',')[0].strip()==str(proc.pid))
            time.sleep(1)
    record={'command':command,'exitcode':proc.returncode,'seconds':time.monotonic()-started,'pid':proc.pid,'gpu_samples':samples}
    save(ROOT/(name+'-process.json'),record)
    if check:assert proc.returncode==0,(ROOT/(name+'.log')).read_text(errors='replace')[-5000:]
    return record


def compare_predictions(reference,prediction,name):
    import numpy as np
    tol=TOLERANCES[BACKEND]; errors={}
    for key in ['energy','forces','stress']:
        a=np.asarray([x[key] for x in reference]);b=np.asarray([x[key] for x in prediction])
        assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        errors[key]={'max_abs':float(np.max(np.abs(a-b))),'atol':tol[key],'rtol':tol['rtol'],
            'passed':bool(np.allclose(a,b,atol=tol[key],rtol=tol['rtol']))}
    save(ROOT/(name+'.json'),errors)
    assert all(v['passed'] for v in errors.values()), name+': '+str(errors)
    return errors


def frames():
    from ase.io import read,write
    import numpy as np
    a=read(SOURCE/'grace/data/validation.extxyz',index=':')[:2]
    for x in a:x.calc=None
    b=a[0].copy(); b.set_cell([[12.,0,0],[.4,11.7,0],[.1,-.2,12.3]],scale_atoms=True)
    c=a[1].copy();c.set_cell([7.2,7.4,7.6]);c.positions[0]=[.2,.3,.4];c.wrap()
    a.extend([b,c]);write(ROOT/'comparison.extxyz',a)
    save(ROOT/'comparison-provenance.json',{'label_source':'geometry-only synthetic export diagnostic; NOT DFT',
        'frames':['validation-0','validation-1','validation-0 sheared/scaled cell','validation-1 periodic images'],
        'sha256':digest(ROOT/'comparison.extxyz'),'units':{'energy':'eV total','forces':'eV/angstrom','stress':'eV/angstrom^3 tensile, Voigt xx yy zz yz xz xy'},
        'tolerances_fixed_before_submission':TOLERANCES[BACKEND]})
    return a


def ase_predictions(calculator,structures):
    values=[]
    for structure in structures:
        atoms=structure.copy();atoms.calc=calculator
        values.append({'energy':float(atoms.get_potential_energy()),'forces':atoms.get_forces().tolist(),'stress':atoms.get_stress().tolist()})
    return values


def lammps_predictions(binary,model,structures,plugin=None):
    import numpy as np
    from ase.io import write
    from ase.units import bar
    values=[]
    for i,atoms in enumerate(structures):
        folder=ROOT/('lammps-'+str(i));folder.mkdir()
        write(folder/'atoms.data',atoms,format='lammps-data',atom_style='atomic',specorder=['Al'])
        init=('plugin load '+str(plugin)+'\n') if plugin else ''
        pair=('deepmd '+str(model)) if BACKEND=='dpa4' else 'grace'
        coeff='pair_coeff * * Al' if BACKEND=='dpa4' else 'pair_coeff * * '+str(model)+' Al'
        text=init+'''units metal
atom_style atomic
atom_modify map yes
boundary p p p
read_data atoms.data
mass 1 26.9815385
newton on
pair_style @PAIR@
@COEFF@
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
thermo_style custom step pe pxx pyy pzz pyz pxz pxy
thermo_modify format float %.17g
run 0
print "$(pe:%.17g) $(pxx:%.17g) $(pyy:%.17g) $(pzz:%.17g) $(pyz:%.17g) $(pxz:%.17g) $(pxy:%.17g)" file prediction.txt
write_dump all custom forces.dump id type x y z fx fy fz modify sort id format float %.17g
'''.replace('@PAIR@',pair).replace('@COEFF@',coeff)
        (folder/'in.lammps').write_text(text)
        env=dict(os.environ,LAMMPS_PLUGIN_PATH='')
        record=execute([str(binary),'-in','in.lammps'], 'lammps-'+str(i),seconds=120,env=env,cwd=folder)
        output=np.loadtxt(folder/'prediction.txt',ndmin=1)
        dump=(folder/'forces.dump').read_text().splitlines();start=next(j for j,line in enumerate(dump) if line.startswith('ITEM: ATOMS'))
        forces=np.array([[float(v) for v in line.split()] for line in dump[start+1:]])[:,-3:]
        values.append({'energy':float(output[0]),'forces':forces.tolist(),'stress':(-output[1:]*bar).tolist()})
    return values


def restore_check_pt(source,loaded):
    import torch
    import numpy as np
    a=torch.load(source,map_location='cpu',weights_only=False);b=torch.load(loaded,map_location='cpu',weights_only=False)
    matched=[];scheduler_changes=[]
    def equal(x,y,path):
        if isinstance(x,torch.Tensor):
            assert torch.equal(x,y),path; matched.append(path)
        elif isinstance(x,dict):
            # Model metadata records the new stop step; parameters/buffers must match exactly.
            for k,v in x.items():
                if k=='_extra_state':continue
                equal(v,y[k],path+'/'+str(k))
        elif isinstance(x,(list,tuple)):
            assert len(x)==len(y)
            for i,(u,v) in enumerate(zip(x,y)):equal(u,v,path+'/'+str(i))
        elif path.startswith('optimizer/param_groups/') and path.endswith('/lr'):
            if x!=y:scheduler_changes.append({'field':path,'checkpoint':x,'restart':y,'reason':'LR scheduler recomputed for the extended stopping target'})
        else: assert x==y,(path,x,y)
    equal(a['model'],b['model'],'model');equal(a['optimizer'],b['optimizer'],'optimizer')
    assert json.loads(loaded.with_suffix('.json').read_text())['step']==20
    return {'exactly_restored_tensor_count':len(matched),'optimizer_tensor_count':sum(k.startswith('optimizer/') for k in matched),'restored_step':20,'scheduler_changes':scheduler_changes}


def dpa4(structures,result):
    import numpy as np
    import torch
    from deepmd.infer import DeepPot
    from deepmd.calculator import DP
    source=SOURCE/'dpa4/model.ckpt-20.pt'
    config=json.loads((SOURCE/'dpa4/input.json').read_text());config['training'].update(numb_steps=24,save_freq=24,disp_freq=1)
    save(ROOT/'input.json',config)
    env=dict(os.environ,RESTART_SNAPSHOT=str(ROOT/'restored/model.pt'))
    result['restart_process']=execute([sys.executable,str(TOOLS/'restart-child.py'),'dpa4','--pt','train','input.json','--restart',str(source)],'restart',env=env)
    result['restart_state']=restore_check_pt(source,ROOT/'restored/model.pt')
    final=ROOT/'model.ckpt-24.pt';state=torch.load(final,map_location='cpu',weights_only=False)
    result['final_step']=state['model']['_extra_state']['train_infos']['step'];assert result['final_step']==24
    assert any(not torch.equal(v,state['model'][k]) for k,v in torch.load(source,map_location='cpu',weights_only=False)['model'].items() if isinstance(v,torch.Tensor) and v.is_floating_point())
    engine=DeepPot(str(final));types=engine.get_type_map(); reference=[]
    for a in structures:
        energy,force,virial=engine.eval(a.positions.reshape(1,-1),a.cell.array.reshape(1,9),[types.index(s) for s in a.get_chemical_symbols()])
        stress=-virial.reshape(3,3)/a.get_volume()
        reference.append({'energy':float(energy.ravel()[0]),'forces':force.reshape(-1,3).tolist(),'stress':stress.flat[[0,4,8,5,2,1]].tolist()})
    save(ROOT/'checkpoint-predictions.json',reference)
    result['export_process']=execute([str(Path(sys.prefix)/'bin/dp'),'--pt','freeze','-c',str(final),'-o',str(ROOT/'exported')],'export',seconds=900)
    exported=ROOT/'exported.pt2';assert exported.exists()
    after=ase_predictions(DP(model=str(exported)),structures);save(ROOT/'ase-export-predictions.json',after)
    result['ase_equivalence']=compare_predictions(reference,after,'ase-equivalence')
    package=Path(sys.prefix)/'lib/python3.12/site-packages'
    binary=package/'lammps/lmp'; plugin=package/'deepmd/lib/libdeepmd_lmpplugin.so'
    result['lammps_binary_sha256']=digest(binary);result['lammps_plugin_sha256']=digest(plugin)
    after=lammps_predictions(binary,exported,structures,plugin);save(ROOT/'lammps-export-predictions.json',after)
    result['lammps_equivalence']=compare_predictions(reference,after,'lammps-equivalence')
    result['export_sha256']=digest(exported)


def grace(structures,result):
    import numpy as np
    import yaml
    import tensorflow as tf
    from tensorpotential import TensorPotential
    from tensorpotential.instructions.base import load_instructions
    from tensorpotential.calculator import TPCalculator
    seed=ROOT/'seed/42';(seed/'checkpoints').mkdir(parents=True)
    source=SOURCE/'grace/seed/42'
    shutil.copyfile(source/'model.yaml',seed/'model.yaml')
    for suffix in ['index','data-00000-of-00001']:shutil.copyfile(source/('checkpoints/checkpoint.'+suffix),seed/('checkpoints/checkpoint.'+suffix))
    config=yaml.safe_load((SOURCE/'grace/input.yaml').read_text())
    config['potential']={'filename':str(seed/'model.yaml'),'shift':False,'reduce_elements':False}
    config['fit'].update(maxiter=6,reset_epoch_and_step=False,reset_optimizer=False)
    (ROOT/'input.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    env=dict(os.environ,RESTART_SNAPSHOT=str(ROOT/'restored/model'))
    result['restart_process']=execute([sys.executable,str(TOOLS/'restart-child.py'),'grace','input.yaml','-rl'],'restart',env=env)
    restored=json.loads((ROOT/'restored/model.json').read_text());assert restored=={'step':20,'epoch':5,'optimizer_iterations':20},restored
    a=tf.train.load_checkpoint(str(source/'checkpoints/checkpoint'));b=tf.train.load_checkpoint(str(ROOT/'restored/model'))
    keys=[k for k in a.get_variable_to_dtype_map() if k!='_CHECKPOINTABLE_OBJECT_GRAPH' and not k.startswith('save_counter/')]
    for k in keys:assert np.array_equal(a.get_tensor(k),b.get_tensor(k)),k
    result['restart_state']=dict(restored,exactly_restored_tensor_count=len(keys),optimizer_tensor_count=sum(k.startswith('optimizer/') for k in keys))
    checkpoint=seed/'checkpoints/checkpoint';reader=tf.train.load_checkpoint(str(checkpoint))
    result['final_step']=int(reader.get_tensor('step/.ATTRIBUTES/VARIABLE_VALUE'));assert result['final_step']==24
    assert int(reader.get_tensor('optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE'))==24
    model=TensorPotential(potential=load_instructions(str(seed/'model.yaml')),param_dtype=tf.float64)
    model.load_checkpoint(checkpoint_name=str(checkpoint),expect_partial=True,assert_existing_objects_matched=True,model_only=True)
    model.model.decorate_compute_function(jit_compile=False)
    reference=ase_predictions(TPCalculator(model.model,enable_uq_if_available=False),structures)
    save(ROOT/'checkpoint-predictions.json',reference)
    result['export_process']=execute([str(Path(sys.prefix)/'bin/gracemaker'),'input.yaml','-rl','-s'],'export',seconds=180)
    exported=seed/'saved_model';assert (exported/'saved_model.pb').exists()
    after=ase_predictions(TPCalculator(str(exported),enable_uq_if_available=False),structures);save(ROOT/'ase-export-predictions.json',after)
    result['ase_equivalence']=compare_predictions(reference,after,'ase-equivalence')
    build=json.loads((TOOLS/'grace-lammps-build.json').read_text());binary=Path(build['binary'])
    assert digest(binary)==build['binary_sha256']
    result['lammps_binary_sha256']=digest(binary)
    after=lammps_predictions(binary,exported,structures);save(ROOT/'lammps-export-predictions.json',after)
    result['lammps_equivalence']=compare_predictions(reference,after,'lammps-equivalence')
    result['export_sha256']={p.relative_to(exported).as_posix():digest(p) for p in exported.rglob('*') if p.is_file()}


def main():
    result={'status':'running','backend':BACKEND,'job_id':os.environ['SLURM_JOB_ID'],'host':os.uname().nodename,
        'environment':sys.prefix,'started_utc':datetime.now(timezone.utc).isoformat(),'scope':'synthetic software restart/deployment check only',
        'tolerances':TOLERANCES[BACKEND]}
    try:
        backup=json.loads((TOOLS/'source-backup.json').read_text())
        source_hashes={n:r['sha256'] for n,r in backup['members'].items() if n.startswith(BACKEND+'/')}
        for n,want in source_hashes.items():assert digest(SOURCE/n)==want,n
        result['source_input_sha256']=source_hashes
        structures=frames()
        (dpa4 if BACKEND=='dpa4' else grace)(structures,result)
        assert result['restart_process']['gpu_samples'],'Restart training process not observed on GPU'
        result['status']='passed'
    except Exception:
        result['status']='failed';result['error']=traceback.format_exc()
    result['finished_utc']=datetime.now(timezone.utc).isoformat();save(ROOT/'result.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':sys.exit(main())
