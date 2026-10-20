"""Reproduce Al32 fine-tuning comparisons from the independent archive audit."""
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
RUN=ROOT/'runs/fm-al32-finetune-20261019-01'
OUT=ROOT/'results/al32-foundation-finetune-20261020'
NAMES={'dpa4':'DPA4-Mini','grace':'GRACE-1L'}
KEYS=['energy_aligned_eV_per_atom','forces_eV_per_A','stress_eV_per_A3']
FACTORS=[1000,1000,160.2176634]
TITLES=['Aligned energy RMSE (meV/atom)','Force RMSE (meV/Angstrom)','Stress RMSE (GPa)']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assessment=json.loads((RUN/'assessment.json').read_text());assert assessment['status']=='passed'
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(11.5,6.6),layout='constrained')
    csvrows=[]
    lines=[r'\begin{center}',r'\small',r'\begin{tabular}{lllrrrr}',r'\toprule',
           r'Model & State & Split & $E_{raw}$ & $E_{aligned}$ & $F$ & $\sigma$\\',r'\midrule']
    for i,(backend,name) in enumerate(NAMES.items()):
        for kind,color,shift in [('original','#a0a9b5',-.18),('selected','#086eac',.18)]:
            m=assessment['metrics'][backend][kind]
            for j,(key,factor,title) in enumerate(zip(KEYS,FACTORS,TITLES)):
                axes[i,j].bar(np.arange(3)+shift,[m['metrics_by_split'][s][key]['rmse']*factor for s in ('train','validation','test')],width=.36,label=kind,color=color)
                axes[i,j].set(xticks=range(3),xticklabels=['train (21)','val (6)','test (13)'],ylabel=title,title=name)
                axes[i,j].set_axisbelow(True);axes[i,j].grid(axis='y',alpha=.2)
            for split,sm in m['metrics_by_split'].items():
                values=[sm['energy_raw_eV_per_atom']['rmse']*1000]+[sm[k]['rmse']*f for k,f in zip(KEYS,FACTORS)]
                lines.append(name+' & '+kind+' & '+split+' & '+' & '.join(f'{v:.4f}' for v in values)+r'\\')
                csvrows.append(dict(model=backend,state=kind,split=split,configurations=sm['configurations'],delta_Al_eV_per_atom=m['delta_Al_eV_per_atom'],
                    **{k+'_'+stat:sm[k][stat] for k in ['energy_raw_eV_per_atom']+KEYS for stat in ('mae','rmse','signed_mean','max_absolute')}))
        lines.append(r'\midrule')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=2)
    fig.savefig(OUT/'split-rmse.png',dpi=180);plt.close(fig)
    lines[-1]=r'\bottomrule';lines += [r'\end{tabular}',r'\end{center}']
    (OUT/'rmse-table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (OUT/'split-metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    fig,axes=plt.subplots(1,2,figsize=(9.7,3.9),layout='constrained')
    histories={}
    for ax,(backend,name) in zip(axes,NAMES.items()):
        selection=json.loads((RUN/backend/'selection.json').read_text());history=selection['all_candidates']
        histories[backend]=[dict(progress=h['progress'],score=h['score']) for h in history]
        ax.plot([h['progress'] for h in history],[h['score'] for h in history],'o-',label='Full validation score')
        ax.scatter([selection['progress']],[selection['score']],marker='*',s=150,color='#d27619',zorder=4,label='Locked selection')
        ax.set(title=name,xlabel='Updates' if backend=='dpa4' else 'Epochs',ylabel='Validation score S (lower is better)')
        ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2)
    fig.savefig(OUT/'validation-selection.png',dpi=180);plt.close(fig)
    (OUT/'selection-history.json').write_text(json.dumps(histories,indent=2)+'\n')
    manifest=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),analysis_source_sha256=sha(Path(__file__)),
        inputs={str(p.relative_to(ROOT)):sha(p) for p in [RUN/'assessment.json',RUN/'dpa4/selection.json',RUN/'grace/selection.json']},
        packages={p:importlib.metadata.version(p) for p in ('matplotlib','numpy')},
        settings=dict(energy_meV=1000,force_meV=1000,stress_GPa=160.2176634,alignment='Frozen train-only shifts; no test refit'),
        outputs={p.name:sha(p) for p in OUT.iterdir() if p.name!='figure-manifest.json'})
    (OUT/'figure-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(histories))


if __name__=='__main__':main()
