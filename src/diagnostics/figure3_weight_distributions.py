"""Empirical component-loading distributions for the selected Figure 3 groups."""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

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
    fig.subplots_adjust(left=.11,right=.985,bottom=.29,top=.73,wspace=.22)
    records=[];summary=[];curves=[];maximum_density=0
    inputs={str(assignments):sha(assignments)}
    grid=np.linspace(0,1,512)
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
        for index,(component,label,color) in enumerate(zip(COMPONENTS,LABELS,COLORS)):
            kde=gaussian_kde(proportions[:,index],bw_method='scott')
            # Relative loadings are bounded. Reflect the same Gaussian-kernel
            # estimate at 0 and 1, then normalize over its displayed support.
            density=kde(grid)+kde(-grid)+kde(2-grid)
            density/=np.trapz(density,grid)
            np.testing.assert_allclose(np.trapz(density,grid),1,rtol=0,atol=1e-12)
            maximum_density=max(maximum_density,float(density.max()))
            axis.fill_between(grid,0,density,color=color,alpha=.10,linewidth=0)
            axis.plot(grid,density,color=color,lw=1.5,label=label)
            bandwidth=float(np.sqrt(kde.covariance[0,0]))
            curves.extend(dict(group=key,component=component,relative_weight=float(x),density=float(y),
                bandwidth=bandwidth) for x,y in zip(grid,density))
            for i,row in enumerate(frame.itertuples()):
                records.append(dict(group=key,electrode=row.electrode,subject=row.subject,channel=row.channel,
                    component=component,original_loading=float(loadings[i,index]),
                    loading_sum=float(loadings[i].sum()),proportion=float(proportions[i,index])))
            summary.append(dict(group=key,component=component,n_electrodes=n,n_participants=subjects,
                mean=float(proportions[:,index].mean()),median=float(np.median(proportions[:,index])),
                q25=float(np.quantile(proportions[:,index],.25)),q75=float(np.quantile(proportions[:,index],.75)),
                gaussian_kernel_bandwidth=bandwidth))
        axis.set_title(f'{title}\n{n} electrodes / {subjects} participants',fontsize=9,pad=12)
        axis.set(xlim=(0,1),xticks=np.linspace(0,1,5))
        axis.set_xlabel('Relative NMF loading',fontsize=9)
        axis.tick_params(axis='both',labelsize=8,length=3)
        axis.spines[['top','right']].set_visible(False)
        inputs[str(source)]=sha(source)
    axes[0].set_ylabel('Probability density',fontsize=9)
    axes[0].set_ylim(0,maximum_density*1.05)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.105),
        ncol=3,frameon=False,fontsize=8)
    fig.text(.5,.055,'Relative loading = component loading / sum of all three loadings',ha='center',fontsize=7)
    svg=IMAGES/'fig3_component_weight_distributions.svg'
    fig.savefig(svg,format='svg',bbox_inches=None);plt.close(fig)
    pd.DataFrame(records).to_csv(RESULTS/'electrode_weights_long.csv',index=False)
    pd.DataFrame(summary).to_csv(RESULTS/'descriptive_summary.csv',index=False)
    pd.DataFrame(curves).to_csv(RESULTS/'density_curves.csv',index=False)
    assert all(sha(p)==digest for p,digest in protected.items())
    write(RESULTS/'manifest.json',dict(passed=True,job_id=os.environ['SLURM_JOB_ID'],
        svg=str(svg),svg_sha256=sha(svg),source_hashes=inputs,
        n_electrodes=dict(picture=21,lexical=27),n_participants=dict(picture=14,lexical=16),
        proportion_definition='Each frozen W loading divided by the sum of the three W loadings for that electrode',
        density_method='Gaussian kernel density, Scott bandwidth; reflect at 0 and 1; unit integral on [0,1]',
        normal_distribution_assumed=False,grid_points=len(grid),common_proportion_range=[0,1],
        common_density_y_limits=[0,maximum_density*1.05],
        no_inferential_tests=True,no_NMF_refit=True,
        original_files_unchanged=len(protected),main_and_supplementary_figures_unchanged=True))
    return svg
