"""Reproduce split tables and baseline-error figures from the audited evaluation."""
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
RUN=ROOT/'runs/fm-al32-baseline-20261019-02'
OUT=ROOT/'results/al32-foundation-baseline-20261019'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    a=json.loads((RUN/'assessment.json').read_text());assert a['status']=='passed'
    rows=json.loads((RUN/'dataset-manifest.json').read_text())['configurations']
    metrics=a['metrics'];OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    splits=['train','validation','test'];names={'dpa4':'DPA4-Mini','grace':'GRACE-1L'}
    colors={'train':'#086eac','validation':'#d27619','test':'#6b5695'}
    fig,axes=plt.subplots(2,2,figsize=(10.4,6.0),sharex=True,sharey=True,layout='constrained')
    for i,backend in enumerate(names):
        for j,key in enumerate(['energy_raw_eV_per_atom','energy_aligned_eV_per_atom']):
            ax=axes[i,j]
            for split,marker in zip(splits,['o','^','s']):
                chosen=[(ix,r) for ix,r in enumerate(rows) if r['split']==split]
                ax.scatter([ix for ix,r in chosen],[metrics[backend]['per_frame'][r['configuration_id']][key]*1000 for ix,r in chosen],
                           color=colors[split],marker=marker,s=23,label=split)
            ax.axhline(0,color='black',ls='--',lw=.8);ax.axvspan(-.5,8.5,color='#aaaaaa',alpha=.12);ax.axvspan(32.5,39.5,color='#aaaaaa',alpha=.12)
            ax.set(title=names[backend]+(' raw' if j==0 else ' + train-only shift'),ylabel='Model - DFT energy (meV/atom)',xlim=(-1,40))
            ax.grid(alpha=.15)
            if i==1:ax.set_xlabel('Configuration index: FCC 0-8 | MD 9-32 | HCP 33-39')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=3)
    fig.savefig(OUT/'energy-residuals.png');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11.2,3.8),layout='constrained')
    for ax,key,factor,title in zip(axes,['energy_aligned_eV_per_atom','forces_eV_per_A','stress_eV_per_A3'],
        [1000,1000,160.2176634],['Aligned energy RMSE (meV/atom)','Force-component RMSE (meV/Angstrom)','Stress-component RMSE (GPa)']):
        for i,backend in enumerate(names):
            ax.bar(np.arange(3)+(i-.5)*.34,[metrics[backend]['metrics_by_split'][s][key]['rmse']*factor for s in splits],width=.34,label=names[backend],color=['#086eac','#d27619'][i])
        ax.set(xticks=np.arange(3),xticklabels=['train (21)','val (6)','test (13)'],ylabel=title);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2)
    fig.savefig(OUT/'split-rmse.png');plt.close(fig)
    lines=[r'\begin{center}',r'\small',r'\begin{tabular}{llrrrr}',r'\toprule',
        r'Модель & Split & $E_{\rm raw}$ MAE & $E_{\rm raw}$ RMSE & $E_{\rm aligned}$ MAE & $E_{\rm aligned}$ RMSE\\',r'\midrule']
    csvrows=[]
    for backend,title in names.items():
        for split in splits:
            m=metrics[backend]['metrics_by_split'][split]
            values=[m[k][stat]*1000 for k in ['energy_raw_eV_per_atom','energy_aligned_eV_per_atom'] for stat in ['mae','rmse']]
            lines.append(title+' & '+split+' & '+' & '.join(f'{v:.4f}' for v in values)+r'\\')
            csvrows.append(dict(model=backend,split=split,configurations=m['configurations'],delta_Al_eV_per_atom=metrics[backend]['delta_Al_eV_per_atom'],
                **{k+'_'+stat:m[k][stat] for k in ('energy_raw_eV_per_atom','energy_aligned_eV_per_atom','forces_eV_per_A','stress_eV_per_A3') for stat in ('mae','rmse','signed_mean','max_absolute')}))
        lines.append(r'\midrule')
    lines[-1]=r'\bottomrule';lines += [r'\end{tabular}',r'\end{center}']
    (OUT/'energy-table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    lines=[r'\begin{center}',r'\small',r'\begin{tabular}{llrrrr}',r'\toprule',
        r'Модель & Split & $F$ MAE & $F$ RMSE & $\sigma$ MAE & $\sigma$ RMSE\\',r'\midrule']
    for backend,title in names.items():
        for split in splits:
            m=metrics[backend]['metrics_by_split'][split]
            values=[m['forces_eV_per_A'][s]*1000 for s in ('mae','rmse')]+[m['stress_eV_per_A3'][s]*160.2176634 for s in ('mae','rmse')]
            lines.append(title+' & '+split+' & '+' & '.join(f'{v:.4f}' for v in values)+r'\\')
        lines.append(r'\midrule')
    lines[-1]=r'\bottomrule';lines += [r'\end{tabular}',r'\end{center}']
    (OUT/'derivative-table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (OUT/'split-metrics.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    manifest=dict(analysis_source_sha256=sha(Path(__file__)),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        inputs={'assessment.json':sha(RUN/'assessment.json'),'dataset-manifest.json':sha(RUN/'dataset-manifest.json')},
        packages={p:importlib.metadata.version(p) for p in ('matplotlib','numpy')},
        settings=dict(energy_scale=1000,force_scale=1000,stress_GPa_per_eV_A3=160.2176634,offset='Frozen train-only per-model additive shift; no phase fits'),
        outputs={p.name:sha(p) for p in OUT.iterdir() if p.suffix in ('.png','.tex','.csv')})
    (OUT/'figure-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest['outputs'],indent=2))


if __name__=='__main__':main()
