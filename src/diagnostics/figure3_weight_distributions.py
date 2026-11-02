"""Empirical component-loading distributions for the selected Figure 3 groups."""
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    fig,axes=plt.subplots(1,2,figsize=(17/2.54,10/2.54))
    fig.subplots_adjust(left=.04,right=.96,bottom=.23,top=.73,wspace=.25)
    records=[];summary=[];coordinates=[];group_means=[]
    inputs={str(assignments):sha(assignments)}
    height=np.sqrt(3)/2
    # Columns follow COMPONENTS: Sensory, Sustained, Motor.
    vertices=np.array([[.5,height],[0.,0.],[1.,0.]])
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
        xy=proportions@vertices
        recovered_sensory=xy[:,1]/height
        recovered_motor=xy[:,0]-.5*recovered_sensory
        recovered=np.column_stack([recovered_sensory,1-recovered_sensory-recovered_motor,recovered_motor])
        np.testing.assert_allclose(recovered,proportions,rtol=0,atol=1e-12)
        outline=np.vstack([vertices,vertices[0]])
        axis.plot(outline[:,0],outline[:,1],color='#737B85',lw=.8,zorder=1)
        for fraction in (.25,.5,.75):
            for index in range(3):
                others=[i for i in range(3) if i!=index]
                ends=fraction*vertices[index]+(1-fraction)*vertices[others]
                axis.plot(ends[:,0],ends[:,1],color='#E0E3E7',lw=.55,zorder=0)
            axis.text(.5*fraction-.025,height*fraction,f'{fraction:.0%}',
                ha='right',va='center',fontsize=5.5,color='#9299A2')
            axis.text(1-fraction,-.025,f'{fraction:.0%}',
                ha='center',va='top',fontsize=5.5,color='#9299A2')
            axis.text(.5+.5*fraction+.025,height*(1-fraction),f'{fraction:.0%}',
                ha='left',va='center',fontsize=5.5,color='#9299A2')
        axis.plot([.5,.5],[0,height],color='#A2A9B2',lw=.8,ls=(0,(3,3)),zorder=1)
        axis.scatter(xy[:,0],xy[:,1],s=21,facecolor='#4D5562',alpha=.7,
            edgecolor='white',linewidth=.45,zorder=3,clip_on=False)
        mean=proportions.mean(axis=0);mean_xy=mean@vertices
        axis.scatter(*mean_xy,s=48,marker='D',facecolor='white',edgecolor='#23262B',linewidth=1.15,zorder=4)
        group_means.append(dict(group=key,sensory=float(mean[0]),sustain=float(mean[1]),motor=float(mean[2]),
            ternary_x=float(mean_xy[0]),ternary_y=float(mean_xy[1])))
        for i,row in enumerate(frame.itertuples()):
            coordinates.append(dict(group=key,electrode=row.electrode,subject=row.subject,channel=row.channel,
                sensory=float(proportions[i,0]),sustain=float(proportions[i,1]),motor=float(proportions[i,2]),
                ternary_x=float(xy[i,0]),ternary_y=float(xy[i,1])))
        for index,(component,label,color) in enumerate(zip(COMPONENTS,LABELS,COLORS)):
            for i,row in enumerate(frame.itertuples()):
                records.append(dict(group=key,electrode=row.electrode,subject=row.subject,channel=row.channel,
                    component=component,original_loading=float(loadings[i,index]),
                    loading_sum=float(loadings[i].sum()),proportion=float(proportions[i,index])))
            summary.append(dict(group=key,component=component,n_electrodes=n,n_participants=subjects,
                mean=float(proportions[:,index].mean()),median=float(np.median(proportions[:,index])),
                q25=float(np.quantile(proportions[:,index],.25)),q75=float(np.quantile(proportions[:,index],.75))))
        axis.text(.5,height+.055,'Sensory\n100%',ha='center',va='bottom',fontsize=8,color=COLORS[0])
        axis.text(-.005,-.10,'Sustained\n100%',ha='center',va='top',fontsize=8,color=COLORS[1])
        axis.text(1.005,-.10,'Motor\n100%',ha='center',va='top',fontsize=8,color=COLORS[2])
        axis.set_title(f'{title}\n{n} electrodes / {subjects} participants',fontsize=9,pad=16)
        axis.set(xlim=(-.15,1.15),ylim=(-.20,height+.20),aspect='equal')
        axis.axis('off')
        inputs[str(source)]=sha(source)
    handles=[Line2D([],[],marker='o',linestyle='none',markersize=4,color='#4D5562',label='Electrode'),
        Line2D([],[],marker='D',linestyle='none',markersize=5,markerfacecolor='white',markeredgecolor='#23262B',label='Mean composition'),
        Line2D([],[],color='#A2A9B2',ls=(0,(3,3)),lw=.8,label='Sustained = Motor')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.115),ncol=3,frameon=False,fontsize=7.5)
    fig.text(.5,.065,'Each electrode retains all three component proportions; grid spacing = 25%.',ha='center',fontsize=7)
    svg=IMAGES/'fig3_component_weight_ternary.svg'
    fig.savefig(svg,format='svg',bbox_inches=None);plt.close(fig)
    pd.DataFrame(records).to_csv(RESULTS/'electrode_weights_long.csv',index=False)
    pd.DataFrame(summary).to_csv(RESULTS/'descriptive_summary.csv',index=False)
    pd.DataFrame(coordinates).to_csv(RESULTS/'ternary_coordinates.csv',index=False)
    pd.DataFrame(group_means).to_csv(RESULTS/'mean_compositions.csv',index=False)
    assert all(sha(p)==digest for p,digest in protected.items())
    write(RESULTS/'manifest.json',dict(passed=True,job_id=os.environ['SLURM_JOB_ID'],
        svg=str(svg),svg_sha256=sha(svg),source_hashes=inputs,
        n_electrodes=dict(picture=21,lexical=27),n_participants=dict(picture=14,lexical=16),
        proportion_definition='Each frozen W loading divided by the sum of the three W loadings for that electrode',
        visualization='Ternary composition scatter; one neutral-color point per electrode, no argmax labels',
        triangle_vertices=dict(zip(COMPONENTS,vertices.tolist())),grid_spacing=.25,
        dashed_line='Sustained and Motor proportions are equal',mean_marker='Open diamond, electrode-equal mean',
        projection_inverted_to_original_proportions=True,no_coordinate_jitter=True,no_smoothing=True,
        common_proportion_range=[0,1],
        no_inferential_tests=True,no_NMF_refit=True,
        original_files_unchanged=len(protected),main_and_supplementary_figures_unchanged=True))
    (IMAGES/'fig3_component_weight_distributions.svg').unlink(missing_ok=True)
    (RESULTS/'density_curves.csv').unlink(missing_ok=True)
    return svg
