"""Shared fixed validation-only checkpoint selection; test rows are forbidden."""
from baseline_metrics import evaluate


def candidate_metrics(rows,predictions,labels,settings):
    assert {r['split'] for r in rows}=={'train','validation'},'Selection accepts train/validation only'
    metrics=evaluate(rows,predictions,labels)
    val=metrics['metrics_by_split']['validation']
    score=sum((val[key]['rmse']/settings[scale])**2 for key,scale in (
        ('energy_aligned_eV_per_atom','energy_RMSE_scale_eV_per_atom'),
        ('forces_eV_per_A','force_RMSE_scale_eV_per_A'),
        ('stress_eV_per_A3','stress_RMSE_scale_eV_per_A3')))/3
    return dict(score=score,metrics=metrics)


def choose(candidates):
    assert candidates and all(c['score']>=0 and c['score']<float('inf') for c in candidates)
    return min(candidates,key=lambda c:(c['score'],c['progress']))
