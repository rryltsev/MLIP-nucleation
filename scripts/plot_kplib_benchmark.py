"""Produce scientific figures and LaTeX tables from the audited kpLib run."""
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
RUN=ROOT/'runs/al32-kplib-benchmark-20261019-01'
OUT=ROOT/'results/al32-kplib-20261019'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    a=json.loads((RUN/'quality-assessment.json').read_text())
    assert a['raw_label_quality_audit_passed'] and a['raw_cases_verified']==14
    cfg=json.loads((RUN/'system.json').read_text())
    for n,h in a['source_sha256'].items():assert sha(RUN/n)==h
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    metrics=list(cfg['tolerances']);distances=cfg['kplib']['min_distance_A']
    names={'fcc-04':'FCC (fcc-04)','md-00-f1':'Disordered (md-00-f1)'}
    labels=['Energy error / tolerance','Max force-component error / tolerance','Max stress-component error / tolerance']
    fig,axes=plt.subplots(2,3,figsize=(11.4,6.4),layout='constrained',sharex=True)
    for row,cid in enumerate(names):
        cc=[c for c in a['checks'] if c['configuration_id']==cid]
        for j,key in enumerate(metrics):
            ax=axes[row,j];tol=cfg['tolerances'][key]
            for reference,color,marker,title in [('gamma8','#086eac','o','vs 550 eV Gamma8'),('anchor','#d27619','s','vs 450 eV kpLib d80')]:
                ax.plot(distances,[c['versus_'+reference][key]/tol for c in cc],color=color,marker=marker,label=title)
            ax.axhline(1,color='black',ls='--',lw=1,label='Acceptance threshold')
            ax.axhline(a['baseline_summary'][cid]['differences_vs_gamma8'][key]/tol,color='#777777',ls=':',lw=1.5,label='Gamma7 vs Gamma8')
            ax.set(title=names[cid] if j==1 else '',ylabel=labels[j],ylim=(0,None),xticks=distances)
            ax.grid(alpha=.18)
            if row==1:ax.set_xlabel('min_distance (Angstrom)')
    handles,legend=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='outside lower center',ncol=4,fontsize=9)
    fig.savefig(OUT/'convergence.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.8),layout='constrained',sharey=True)
    for ax,cid in zip(axes,names):
        times=[t for t in a['exploratory_timing'] if t['configuration_id']==cid]
        baseline=a['baseline_summary'][cid]
        wall=[baseline['median_seconds']]+[t['wall_seconds'] for t in times]
        gen=[0]+[t['generation_seconds'] for t in times]
        x=np.arange(len(wall))
        ax.bar(x,wall,color=['#747474']+['#086eac']*len(times),label='VASP wall time')
        ax.bar(x,gen,bottom=wall,color='#d27619',label='Grid generation (separate job)')
        for i,(v,g) in enumerate(zip(wall,gen)):ax.text(i,v+g+13,f'{v+g:.1f}',ha='center',fontsize=9)
        ax.set(xticks=x,xticklabels=['Gamma7']+['d'+str(d) for d in distances],title=names[cid],ylim=(0,960),ylabel='Wall time (s)')
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,fontsize=9)
    fig.savefig(OUT/'timing.png');plt.close(fig)
    lines=[r'\begin{center}',r'\small',r'\begin{tabular}{llrrrrrr}',r'\toprule',
           r'Система & Сетка & $N_k$ & $t$, s & $\Delta E$ & $\Delta F$ & $\Delta\sigma$ & $R_{80}$\\',r'\midrule']
    for cid,title in [('fcc-04','FCC'),('md-00-f1','MD')]:
        base=a['baseline_summary'][cid];err=base['differences_vs_gamma8']
        errors=' & '.join(f'{err[k]*1000:.4f}' for k in metrics)
        lines.append(title+r' & $\Gamma7$ & 172 & '+f'{base["median_seconds"]:.2f} & '+errors+r' & ---\\')
        for t in a['exploratory_timing']:
            if t['configuration_id']!=cid:continue
            d=t['min_distance_A'];c=next(c for c in a['checks'] if c['configuration_id']==cid and c['min_distance_A']==d)
            errors=' & '.join(f'{c["versus_gamma8"][k]*1000:.4f}' for k in metrics)
            ratio=max(c['versus_anchor'][k]/cfg['tolerances'][k] for k in metrics)
            lines.append(title+' & '+str(d)+' & '+str(t['actual_kpoints'])+' & '+f'{t["end_to_end_wall_seconds"]:.2f} & '+errors+f' & {ratio:.3f}'+r'\\')
        lines.append(r'\midrule')
    lines[-1]=r'\bottomrule';lines += [r'\end{tabular}',r'\end{center}']
    (OUT/'summary-tables.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    record=dict(analysis_source_sha256=sha(Path(__file__)),
        code_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        inputs={'quality-assessment.json':sha(RUN/'quality-assessment.json'),'system.json':sha(RUN/'system.json')},
        packages={p:importlib.metadata.version(p) for p in ('matplotlib','numpy','contourpy','cycler','fonttools','kiwisolver','packaging','pillow','pyparsing','python-dateutil','six')},
        settings=dict(error_normalization='Absolute component maxima / unchanged tolerance',timing='Gamma7 median of n2; each kpLib grid n1, including separately measured generation',dpi=180),
        output_sha256={p.name:sha(p) for p in OUT.iterdir() if p.name in ('convergence.png','timing.png','summary-tables.tex')})
    (OUT/'figure-manifest.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record['output_sha256'],indent=2))


if __name__=='__main__':main()
