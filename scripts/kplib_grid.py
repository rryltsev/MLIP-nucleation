"""Auditable kpLib-to-VASP adapter. No spatial force/stress symmetry assumptions."""
import json
from pathlib import Path
import time

import numpy as np


def poscar(path):
    lines=Path(path).read_text().splitlines()
    assert float(lines[1])==1.0 and lines[5].split()==['Al'] and lines[6].split()==['32']
    cell=np.array([list(map(float,l.split())) for l in lines[2:5]])
    start=7
    if lines[start].lower().startswith('s'): start+=1
    assert lines[start].lower().startswith('d')
    frac=np.array([list(map(float,l.split()[:3])) for l in lines[start+1:start+33]])
    return cell,frac,[13]*32


def reduce_time_reversal(coords):
    coords=np.mod(np.asarray(coords,dtype=float),1.0)
    assert coords.ndim==2 and coords.shape[1]==3 and np.isfinite(coords).all()
    n=len(coords); used=set(); reduced=[]; weights=[]
    for i,k in enumerate(coords):
        if i in used: continue
        delta=coords-k; delta-=np.rint(delta)
        assert np.count_nonzero(np.max(np.abs(delta),axis=1)<1e-8)==1,'Duplicate grid point'
        inverse=coords+k; inverse-=np.rint(inverse)
        partners=np.flatnonzero(np.max(np.abs(inverse),axis=1)<1e-8)
        assert len(partners)==1,'Grid not closed under time reversal'
        j=int(partners[0]); assert j not in used
        used.update((i,j)); reduced.append(k.tolist()); weights.append(1 if i==j else 2)
    assert len(used)==n and sum(weights)==n
    return reduced,weights


def kpoints(coords,weights,title):
    assert len(coords)==len(weights) and all(w>0 for w in weights)
    return title+'\n'+str(len(coords))+'\nReciprocal\n'+''.join(' '.join(f'{x:.16g}' for x in k)+f' {w}\n' for k,w in zip(coords,weights))


def generate(path,distance,settings):
    from kpLib.interface import get_kpoints
    cell,frac,numbers=poscar(path)
    started=time.perf_counter()
    result=get_kpoints(cell,frac,numbers,min_distance=distance,min_total_kpoints=1,
        include_gamma=settings['include_gamma'],symprec=settings['symprec_A'],use_scale_factor=False)
    elapsed=time.perf_counter()-started
    full=result['coords']; n=int(result['num_total_kpts'])
    assert len(full)==n and result['min_periodic_distance']>=distance-1e-6
    reduced,weights=reduce_time_reversal(full)
    # kpLib weights describe spatial orbits, and contain zeros on other full
    # points. Uniform full weights must be reconstructed before TR reduction.
    assert len(result['weights'])==n and sum(result['weights'])==n
    return dict(requested_min_distance_A=distance,effective_min_distance_A=float(result['min_periodic_distance']),
        full_coords=full,full_weights=[1]*n,tr_coords=reduced,tr_weights=weights,
        num_total_kpts=n,num_time_reversal_kpts=len(reduced),
        library_num_distinct_kpts=int(result['num_distinct_kpts']),generation_seconds=elapsed)
