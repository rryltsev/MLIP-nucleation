"""Describe existing DPA4 failure evidence without choosing another checkpoint."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/fm-al32-finetune-20261019-01'
OUT=ROOT/'results/al32-foundation-finetune-20261020/dpa4-diagnosis.json'
SCALES={'energy_aligned_eV_per_atom':.01,'forces_eV_per_A':.05,'stress_eV_per_A3':.001}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    selection=json.loads((RUN/'dpa4/selection.json').read_text())
    assessment=json.loads((RUN/'assessment.json').read_text());assert assessment['status']=='passed'
    rows=json.loads((RUN/'dataset-manifest.json').read_text())['configurations']
    history=selection['all_candidates'];base=history[0]['metrics']['metrics_by_split']['validation']
    chosen=selection['metrics']['metrics_by_split']['validation']
    evidence=[]
    for item in history:
        val=item['metrics']['metrics_by_split']['validation']
        components={k:(val[k]['rmse']/scale)**2/3 for k,scale in SCALES.items()}
        assert abs(sum(components.values())-item['score'])<1e-12
        evidence.append(dict(progress=item['progress'],score=item['score'],validation_score_components=components,
            validation_RMSE={k:val[k]['rmse'] for k in SCALES},
            training_RMSE={k:item['metrics']['metrics_by_split']['train'][k]['rmse'] for k in SCALES}))
    final=assessment['metrics']['dpa4']
    contrasts={}
    for state,m in final.items():
        p=m['metrics_by_parent']
        contrasts[state]=p['hcp-prototype-001']['metrics']['energy_raw_eV_per_atom']['signed_mean']-p['fcc-prototype-001']['metrics']['energy_raw_eV_per_atom']['signed_mean']
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),analysis_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        analysis_source_sha256=sha(Path(__file__)),
        inputs={str(p.relative_to(ROOT)):sha(p) for p in [RUN/'assessment.json',RUN/'dpa4/selection.json',RUN/'dataset-manifest.json',RUN/'dpa4/weight-update-check.json']},
        status='Descriptive diagnosis; no causal attribution or new model selection',
        split_kind_counts={split:dict(Counter(r['kind'] for r in rows if r['split']==split)) for split in ('train','validation','test')},
        parent_counts={split:len({r['parent_id'] for r in rows if r['split']==split}) for split in ('train','validation','test')},
        selected_progress_unchanged=selection['progress'],
        selected_validation_score_delta_components={k:((chosen[k]['rmse']/scale)**2-(base[k]['rmse']/scale)**2)/3 for k,scale in SCALES.items()},
        trained_candidates_with_worse_validation={k:sum(item['metrics']['metrics_by_split']['validation'][k]['rmse']>base[k]['rmse'] for item in history[1:]) for k in SCALES},
        history=evidence,
        HCP_minus_FCC_mean_energy_residual_eV_per_atom=contrasts,
        contrast_interpretation='Energy-zero invariant contrast between residual means of unmatched ensembles; NOT a matched-volume HCP-FCC phase energy difference',
        causal_gaps=['No E/F/stress evaluation of loaded-initial snapshot before optimizer updates',
            'No alternative LR, frozen-layer, LoRA, loss or precision ablation',
            'One seed; correlated frames; validation has only two parent trajectories',
            'No inference that DFT convergence tolerance equals the RMSE noise floor'],
        test_policy='Existing test inspected descriptively only; no re-ranking, re-splitting or test-driven acquisition. Future method selection uses train/validation and requires a new independent final control.')
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['split_kind_counts','parent_counts','trained_candidates_with_worse_validation','selected_validation_score_delta_components']}))


if __name__=='__main__':main()
