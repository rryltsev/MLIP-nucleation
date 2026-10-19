"""Fixed original-FM error metrics and train-only elemental energy alignment."""
import numpy as np


def statistics(values):
    x=np.asarray(values,dtype=float)
    assert x.size and np.isfinite(x).all()
    return dict(mae=float(np.abs(x).mean()),rmse=float(np.sqrt(np.mean(x*x))),
                signed_mean=float(x.mean()),max_absolute=float(np.abs(x).max()),components=int(x.size))


def evaluate(rows,predictions,labels):
    ids=[r['configuration_id'] for r in rows]
    assert len(set(ids))==len(ids) and set(ids)==set(predictions)==set(labels)
    train=[r['configuration_id'] for r in rows if r['split']=='train'];assert train
    n={cid:len(labels[cid]['forces_eV_per_A']) for cid in ids}
    delta=float(np.mean([(labels[c]['energy_free_eV']-predictions[c]['energy_eV'])/n[c] for c in train]))
    residuals={}
    for row in rows:
        cid=row['configuration_id'];pred=predictions[cid];ref=labels[cid]
        f=np.asarray(pred['forces_eV_per_A'])-np.asarray(ref['forces_eV_per_A'])
        s=np.asarray(pred['stress_eV_per_A3'])-np.asarray(ref['stress_eV_per_A3'])
        assert f.shape==(n[cid],3) and s.shape==(6,)
        raw=(pred['energy_eV']-ref['energy_free_eV'])/n[cid]
        residuals[cid]=dict(energy_raw_eV_per_atom=float(raw),energy_aligned_eV_per_atom=float(raw+delta),force=f,stress=s)
    def summarize(chosen):
        rr=[residuals[c] for c in chosen]
        return dict(configurations=len(chosen),atoms=sum(n[c] for c in chosen),
            energy_raw_eV_per_atom=statistics([r['energy_raw_eV_per_atom'] for r in rr]),
            energy_aligned_eV_per_atom=statistics([r['energy_aligned_eV_per_atom'] for r in rr]),
            forces_eV_per_A=statistics(np.concatenate([r['force'].ravel() for r in rr])),
            stress_eV_per_A3=statistics(np.concatenate([r['stress'] for r in rr])))
    by_split={s:summarize([r['configuration_id'] for r in rows if r['split']==s]) for s in ('train','validation','test') if any(r['split']==s for r in rows)}
    by_parent={p:dict(split=next(r['split'] for r in rows if r['parent_id']==p),
        metrics=summarize([r['configuration_id'] for r in rows if r['parent_id']==p])) for p in sorted({r['parent_id'] for r in rows})}
    for p in by_parent:assert len({r['split'] for r in rows if r['parent_id']==p})==1
    return dict(delta_Al_eV_per_atom=delta,offset_fit_ids=train,metrics_by_split=by_split,metrics_by_parent=by_parent,
        pooled_descriptive=summarize(ids),per_frame={cid:{k:v for k,v in r.items() if k.startswith('energy')} for cid,r in residuals.items()})
