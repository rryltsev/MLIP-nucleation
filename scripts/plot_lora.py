"""Reproduce validation-only DPA4 original/full/LoRA comparisons."""
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/fm-al32-lora-20261020-01'
PRIOR=ROOT/'runs/fm-al32-finetune-20261019-01/dpa4/selection.json'
OUT=ROOT/'results/al32-dpa4-lora-20261021'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    a=json.loads((RUN/'assessment.json').read_text());assert a['status']=='passed'
    current=json.loads((RUN/'dpa4/selection.json').read_text());prior=json.loads(PRIOR.read_text())
    OUT.mkdir(parents=True,exist_ok=True)
    models=[('original',0,current['all_candidates'][0]),('full',prior['progress'],prior),('LoRA',current['progress'],current)]
    keys=['energy_raw_eV_per_atom','energy_aligned_eV_per_atom','forces_eV_per_A','stress_eV_per_A3']
    factors=[1000,1000,1000,160.2176634]
    lines=[r'\begin{center}',r'\small',r'\begin{tabular}{lrrrrrr}',r'\toprule',
        r'Model & Step & $E_{raw}$ & $E_{aligned}$ & $F$ & $\sigma$ & $S$\\',r'\midrule']
    records=[]
    for name,step,selection in models:
        m=selection['metrics']['metrics_by_split']['validation'];values=[m[k]['rmse']*f for k,f in zip(keys,factors)]
        lines.append(name+' & '+str(step)+' & '+' & '.join(f'{v:.4f}' for v in values)+f" & {selection['score']:.6f}"+r'\\')
        records.append(dict(model=name,progress=step,score=selection['score'],delta_Al_eV_per_atom=selection['metrics']['delta_Al_eV_per_atom'],
            **{k+'_'+stat:m[k][stat] for k in keys for stat in ['mae','rmse','signed_mean','max_absolute']}))
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{center}']
    (OUT/'validation-table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (OUT/'validation-metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.7),layout='constrained')
    colors=['#a0a9b5','#d27619','#086eac']
    for ax,key,factor,label in zip(axes,keys[1:],factors[1:],['Aligned energy RMSE (meV/atom)','Force RMSE (meV/Angstrom)','Stress RMSE (GPa)']):
        values=[s['metrics']['metrics_by_split']['validation'][key]['rmse']*factor for _,_,s in models]
        ax.bar(range(3),values,color=colors);ax.set(xticks=range(3),xticklabels=['Original','Full:100','LoRA:500'],ylabel=label)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        digits=5 if key=='stress_eV_per_A3' else 3
        for i,v in enumerate(values):ax.text(i,v,f'{v:.{digits}f}',ha='center',va='bottom',fontsize=9)
        ax.set_ylim(0,max(values)*1.16)
    fig.savefig(OUT/'validation-rmse.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.3,3.8),layout='constrained')
    for name,color,s in [('Full',colors[1],prior),('LoRA',colors[2],current)]:
        history=s['all_candidates'];ax.plot([x['progress'] for x in history],[x['score'] for x in history],'o-',color=color,label=name)
        ax.scatter([s['progress']],[s['score']],marker='*',s=150,color=color,zorder=4)
    ax.axhline(current['all_candidates'][0]['score'],ls='--',lw=.8,color=colors[0],label='Original')
    ax.set(xlabel='Optimizer updates',ylabel='Full-validation score S (lower is better)');ax.grid(alpha=.2);ax.legend()
    fig.savefig(OUT/'validation-score.png',dpi=180);plt.close(fig)
    manifest=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),analysis_source_sha256=sha(Path(__file__)),
        inputs={str(p.relative_to(ROOT)):sha(p) for p in [RUN/'assessment.json',RUN/'dpa4/selection.json',PRIOR]},
        packages={p:importlib.metadata.version(p) for p in ['numpy','matplotlib']},
        settings=dict(scope='Validation6 only; no test',alignment='Train-only fixed per-model shift',force_components='Cartesian',stress_components='Voigt6',stress_GPa_per_eV_A3=160.2176634),
        outputs={p.name:sha(p) for p in OUT.iterdir() if p.name!='figure-manifest.json'})
    (OUT/'figure-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({r['model']:dict(progress=r['progress'],score=r['score']) for r in records}))


if __name__=='__main__':main()
