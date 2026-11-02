"""Empirical component-loading distributions for the selected Figure 3 groups."""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.figure3 import ROOT,sha,style,write

RESULTS=ROOT/'results/diagnostics/fig3_weight_distributions_2026-11-02'
IMAGES=ROOT/'img/diagnostics/fig3_weight_distributions_2026-11-02'
COMPONENTS=('sensory','sustain','motor')
LABELS=('Sensory','Sustained','Motor')
COLORS=('#2369BD','#C4A35A','#A9373B')
GROUPS=(('picture','no_detected_difference',21,14,'Sound–Image\nNo detected difference'),
        ('lexical','first_only',27,16,'Lexical Delay\nDecision > Repeat'))


def render():
    assert os.environ.get('SLURM_JOB_ID'),'Run through Slurm'
    RESULTS.mkdir(parents=True,exist_ok=True);IMAGES.mkdir(parents=True,exist_ok=True)
    original_paths=[p for folder in ('results/nmf','results/fig3','img/fig3')
        for p in (ROOT/folder).rglob('*') if p.is_file()]
    original_paths += [ROOT/f'vizpub/fig{i}.ipynb' for i in range(1,7)]
    original_paths += [ROOT/'vizpub/supplementary/fig3_supplementary.ipynb']
    protected={str(p):sha(p) for p in original_paths}
    assignments=ROOT/'results/nmf/channel_assignments.csv'
    w=pd.read_csv(assignments,float_precision='round_trip')
    columns=[f'loading_{c}' for c in COMPONENTS]
    with np.load(ROOT/'results/nmf/factors.npz',allow_pickle=False) as factors:
        np.testing.assert_allclose(np.linalg.norm(factors['H'],axis=1),1,rtol=0,atol=1e-12)
        assert w.channel.tolist()==factors['channels'].tolist()
        for index,name in enumerate(factors['component_names'].tolist()):
            np.testing.assert_allclose(w[f'loading_{name}'],factors['W'][:,index],rtol=0,atol=1e-12)
    style()
    fig,axes=plt.subplots(1,2,figsize=(15.5/2.54,9.1/2.54),sharey=True)
    fig.subplots_adjust(left=.12,right=.985,bottom=.24,top=.73,wspace=.22)
    records=[];summary=[];inputs={str(assignments):sha(assignments)}
    for axis,(key,region,n,subjects,title) in zip(axes,GROUPS):
        source=ROOT/'results/fig3/regions'/key/f'{region}_electrodes.csv'
        selected=pd.read_csv(source,float_precision='round_trip')
        assert len(selected)==n and selected.subject.nunique()==subjects and selected.electrode.is_unique
        frame=selected[['electrode','subject','channel']].merge(w[['subject','channel',*columns]],
            on=['subject','channel'],how='left',validate='one_to_one')
        loadings=frame[columns].to_numpy(float)
        assert np.isfinite(loadings).all() and (loadings>=0).all() and (loadings.sum(axis=1)>0).all()
        proportions=loadings/loadings.sum(axis=1,keepdims=True)
        np.testing.assert_allclose(proportions.sum(axis=1),1,rtol=0,atol=1e-12)
        jitter=np.random.default_rng(42).uniform(-.10,.10,size=n)
        for index in range(n):
            axis.plot([1+jitter[index],2+jitter[index]],proportions[index,1:],
                color='#808080',alpha=.23,lw=.45,zorder=1)
        boxes=axis.boxplot([proportions[:,i] for i in range(3)],positions=np.arange(3),
            widths=.43,patch_artist=True,showfliers=False,whis=1.5,
            medianprops=dict(color='black',linewidth=1),
            whiskerprops=dict(linewidth=.7),capprops=dict(linewidth=.7),zorder=2)
        for index,(component,label,color,box) in enumerate(zip(COMPONENTS,LABELS,COLORS,boxes['boxes'])):
            box.set_facecolor(color);box.set_alpha(.16);box.set_edgecolor(color)
            for item in boxes['whiskers'][2*index:2*index+2]+boxes['caps'][2*index:2*index+2]:item.set_color(color)
            axis.scatter(index+jitter,proportions[:,index],s=15,color=color,
                edgecolors='white',linewidths=.3,alpha=.85,zorder=3)
            for i,row in enumerate(frame.itertuples()):
                records.append(dict(group=key,electrode=row.electrode,subject=row.subject,channel=row.channel,
                    component=component,original_loading=float(loadings[i,index]),
                    loading_sum=float(loadings[i].sum()),proportion=float(proportions[i,index]),
                    display_x=float(index+jitter[i])))
            summary.append(dict(group=key,component=component,n_electrodes=n,n_participants=subjects,
                mean=float(proportions[:,index].mean()),median=float(np.median(proportions[:,index])),
                q25=float(np.quantile(proportions[:,index],.25)),q75=float(np.quantile(proportions[:,index],.75))))
        axis.set_title(f'{title}\n{n} electrodes / {subjects} participants',fontsize=9,pad=12)
        axis.set_xticks(np.arange(3),LABELS,fontsize=8)
        for tick,color in zip(axis.get_xticklabels(),COLORS):tick.set_color(color)
        axis.set(xlim=(-.4,2.4),ylim=(-.035,1.035),yticks=np.linspace(0,1,5))
        axis.tick_params(axis='both',labelsize=8,length=3)
        axis.spines[['top','right']].set_visible(False)
        inputs[str(source)]=sha(source)
    axes[0].set_ylabel('Relative NMF loading',fontsize=9)
    fig.text(.5,.105,'Each dot is one electrode; gray lines pair its Sustained and Motor weights.',
        ha='center',fontsize=7)
    fig.text(.5,.055,'Relative loading = component loading / sum of all three loadings',ha='center',fontsize=7)
    svg=IMAGES/'fig3_component_weight_distributions.svg'
    fig.savefig(svg,format='svg',bbox_inches=None);plt.close(fig)
    pd.DataFrame(records).to_csv(RESULTS/'electrode_weights_long.csv',index=False)
    pd.DataFrame(summary).to_csv(RESULTS/'descriptive_summary.csv',index=False)
    assert all(sha(p)==digest for p,digest in protected.items())
    write(RESULTS/'manifest.json',dict(passed=True,job_id=os.environ['SLURM_JOB_ID'],
        svg=str(svg),svg_sha256=sha(svg),source_hashes=inputs,
        n_electrodes=dict(picture=21,lexical=27),n_participants=dict(picture=14,lexical=16),
        proportion_definition='Each frozen W loading divided by the sum of the three W loadings for that electrode',
        box_definition='Median and interquartile range; whiskers 1.5 IQR; every electrode shown as a dot, including outliers',
        pairing='Gray lines connect Sustained/Motor values of the same electrode within each group',
        jitter_seed=42,common_proportion_range=[0,1],common_display_y_limits=[-.035,1.035],
        no_inferential_tests=True,no_NMF_refit=True,
        original_files_unchanged=len(protected),main_and_supplementary_figures_unchanged=True))
    return svg
