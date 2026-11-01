"""One original NMF weight vector per electrode, three PN response regions."""
import json
import os

from src.supplementary.picture_fixed_template_expression import ROOT,BASE,DATA,IMG,COLORS,LABELS
from src.supplementary.picture_component_proportions import _old_gaussian_renderer,_save,EXPORTS

AUDIT=ROOT/'results/audit/picture_region_original_nmf_weights_2026-10-27'
OUTPUT=BASE/'region_original_nmf_weights_2026-10-27'
GROUPS=('Sound-only','Both','Picture-only')
DISPLAY={'Sound-only':'Sound-only','Both':'Both','Picture-only':'Image-only'}
COMPONENTS=('sensory','sustain','motor')


def notebook_cells():
    import nbformat as nbf
    return [nbf.v4.new_markdown_cell('''## 5. Three response regions: the original NMF component weights

**Exactly three bilateral maps:** Sound-only, Both and Image-only. Image-only is the unchanged 11-site Picture-only set. Response-set membership remains the native Stimulus **0–1 s** baseline classification, with **25 / 13 / 11 sites** respectively. There is no second Sound/Image-condition split within a region.

Each electrode uses its **one original frozen NMF weight vector**: `loading_sensory`, `loading_sustain`, `loading_motor` from `results/nmf/channel_assignments.csv`. These are the existing cross-task Sound-discovery weights; the condition-specific PN NNLS coefficients from the preceding exploratory sections are not used here. No template fitting, new projection, condition averaging, HGA multiplication, label assignment or statistical test is performed.

Original weights exist for **44 of the 49 responsive sites**: Sound-only **23/25**, Both **12/13**, Image-only **9/11**. The other **2 / 1 / 2 sites** are separately listed as **Unassigned: no original NMF W**. They remain in the response-set counts and lists, with no invented weights or map contribution. Group and hemisphere labels explicitly distinguish all responsive sites from sites with available original W.

For the already agreed relative-weight display, normalize each available electrode's three original weights as **p_k = W_k / sum_j W_j**. The stacked bar for each response region is the electrode-equal mean of these proportions among sites with original W; the three bars and three maps use the exact same 44-site weight source. This is a continuous functional composition, not hard-label electrode counts or percentages of explained variance.

Reuse the original Gaussian surface/RGB helpers, views and colors: **Sensory blue, Sustained gold, Motor red; sigma 5 mm; nearest-electrode support 6 mm; alpha P95**. Each electrode contributes its three proportions to a single mixed-color bilateral map. Mixed hues summarize the local component mixture; opacity represents kernel sampling support. No significance boundary or nonsignificant background-point layer is added.

The preceding six-map section and its exports were archived before replacement. All original NMF files, native response masks, condition-specific PN projections and statistical results remain unchanged.
''',id='picture-component-proportions-note'),nbf.v4.new_code_cell('''from pathlib import Path
import sys
from IPython.display import SVG,display
_pn_region_root=Path('/cwork/ns458/insula-paper')
if str(_pn_region_root) not in sys.path:
    sys.path.insert(0,str(_pn_region_root))
from src.supplementary.picture_region_nmf_weights import render,OUTPUT as region_weight_data
region_weight_paths=render()
display(SVG(filename=str(region_weight_paths['composition'])))
display(SVG(filename=str(region_weight_paths['all_maps'])))
import pandas as pd
display(pd.read_csv(region_weight_data/'region_counts.csv'))
display(pd.read_csv(region_weight_data/'unassigned_sites.csv'))
print('Original NMF-weight site list:',region_weight_data/'electrode_weights.csv')
''',id='picture-component-proportions-display')]


def _prepare():
    import numpy as np
    import pandas as pd
    from scripts.refresh_lexical_condition_sets import sha,write
    nmf=ROOT/'results/nmf/channel_assignments.csv'
    sources=[DATA/'sites.csv',nmf]
    hashes={str(p):sha(p) for p in sources}
    sites=pd.read_csv(DATA/'sites.csv',float_precision='round_trip')
    assignments=pd.read_csv(nmf,float_precision='round_trip')
    columns=[f'loading_{c}' for c in COMPONENTS]
    assert not assignments.duplicated(['subject','channel']).any()
    sites=sites.merge(assignments[['subject','channel']+columns],on=['subject','channel'],how='left',validate='one_to_one',indicator='original_nmf_match')
    sites['has_original_nmf_weight']=sites.original_nmf_match.eq('both')
    available=sites.has_original_nmf_weight
    assert len(sites)==49 and sites.electrode.is_unique and available.sum()==44
    assert sites.loc[~available,'functional_cluster'].eq('Unassigned').all()
    assert not sites.loc[available,'functional_cluster'].eq('Unassigned').any()
    assert sites.loc[~available,columns].isna().all().all()
    W=sites.loc[available,columns].to_numpy(float)
    assert np.isfinite(W).all() and (W>=0).all() and (W.sum(axis=1)>0).all()
    p=W/W.sum(axis=1,keepdims=True)
    np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-14,rtol=0)
    sites['weight_status']=np.where(available,'Original frozen NMF W','Unassigned: no original NMF W')
    for k,c in enumerate(COMPONENTS):
        sites[f'proportion_{c}']=np.nan
        sites.loc[available,f'proportion_{c}']=p[:,k]
    sites['response_region']=sites.response_set.map(DISPLAY)
    counts=[];means=[]
    for group,expected in zip(GROUPS,[(25,23,2),(13,12,1),(11,9,2)]):
        selected=sites.loc[sites.response_set.eq(group)]
        known=selected.loc[selected.has_original_nmf_weight]
        assert (len(selected),len(known),len(selected)-len(known))==expected
        counts.append(dict(response_region=DISPLAY[group],response_set_source=group,n_sites=len(selected),
            n_participants=selected.subject.nunique(),n_original_weights=len(known),
            n_weight_participants=known.subject.nunique(),n_unassigned=len(selected)-len(known)))
        for c in COMPONENTS:
            means.append(dict(response_region=DISPLAY[group],response_set_source=group,component=c,
                mean_proportion=known[f'proportion_{c}'].mean(),n_original_weights=len(known),n_all_sites=len(selected)))
    summary=pd.DataFrame(means)
    np.testing.assert_allclose(summary.groupby('response_region').mean_proportion.sum(),1,atol=1e-14,rtol=0)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    sites.to_csv(OUTPUT/'electrode_weights.csv',index=False)
    sites.loc[~available].to_csv(OUTPUT/'unassigned_sites.csv',index=False)
    pd.DataFrame(counts).to_csv(OUTPUT/'region_counts.csv',index=False)
    summary.to_csv(OUTPUT/'mean_proportions.csv',index=False)
    write(OUTPUT/'input_sources.json',hashes)
    return sites,summary,hashes


def _bars(sites,summary):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig,ax=plt.subplots(figsize=(7.1,4.0))
    for gi,group in enumerate(GROUPS):
        bottom=0.
        for c in COMPONENTS:
            value=100*float(summary.loc[summary.response_set_source.eq(group)&summary.component.eq(c),'mean_proportion'].iloc[0])
            ax.bar(gi,value,width=.50,bottom=bottom,color=COLORS[c],edgecolor='white',linewidth=.6)
            if value>=8:ax.text(gi,bottom+value/2,f'{value:.1f}%',ha='center',va='center',fontsize=9)
            bottom+=value
    labels=[]
    for group in GROUPS:
        selected=sites.loc[sites.response_set.eq(group)];n=int(selected.has_original_nmf_weight.sum())
        labels.append(f'{DISPLAY[group]}\n{n}/{len(selected)} with NMF W\nUnassigned: {len(selected)-n}')
    ax.set_xticks(range(3),labels,fontsize=9);ax.tick_params(axis='x',length=0)
    ax.set_ylim(0,100);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Mean relative component weight (%)')
    ax.set_xlim(-.7,2.7);ax.spines[['top','right']].set_visible(False)
    fig.legend(handles=[Patch(facecolor=COLORS[c],label=LABELS[c]) for c in COMPONENTS],
        loc='upper center',bbox_to_anchor=(.55,.91),frameon=False,ncol=3,fontsize=9)
    fig.suptitle('Three response regions · original NMF functional weights',fontsize=11,y=.99)
    fig.text(.5,.025,'One fixed weight vector per electrode. Means use the 44 sites with original NMF W.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.12,right=.98,bottom=.24,top=.78)
    return _save(fig,EXPORTS['composition'])


def _maps(sites,images):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    handles=[Patch(facecolor=COLORS[c],label=LABELS[c]) for c in COMPONENTS]
    paths={}
    for group,key in zip(GROUPS,['sound_only','both','picture_only']):
        selected=sites.loc[sites.response_set.eq(group)];known=selected.loc[selected.has_original_nmf_weight]
        fig,axes=plt.subplots(1,2,figsize=(5.0,3.5))
        for ax,hemi,hemilabel in zip(axes,['lh','rh'],['Left','Right']):
            ax.imshow(images[(group,hemi)]);ax.axis('off')
            a=int((selected.x.lt(0) if hemi=='lh' else selected.x.gt(0)).sum())
            n=int((known.x.lt(0) if hemi=='lh' else known.x.gt(0)).sum())
            ax.set_title(f'{hemilabel} · NMF W {n}/{a}',fontsize=9,pad=0)
        fig.suptitle(f'{DISPLAY[group]} · {len(known)} original weights / {len(selected)} sites',fontsize=10,y=.99)
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.09),frameon=False,ncol=3,fontsize=8)
        fig.text(.5,.025,f'Unassigned: {len(selected)-len(known)} · original NMF W · relative component mixture',ha='center',fontsize=8)
        fig.subplots_adjust(left=.01,right=.99,bottom=.18,top=.87,wspace=.01)
        paths[key]=_save(fig,EXPORTS[key])
    fig,axes=plt.subplots(3,2,figsize=(6.4,7.9))
    fig.subplots_adjust(left=.27,right=.99,bottom=.12,top=.91,hspace=.02,wspace=.01)
    for gi,group in enumerate(GROUPS):
        selected=sites.loc[sites.response_set.eq(group)];known=selected.loc[selected.has_original_nmf_weight]
        for hi,(hemi,hemilabel) in enumerate(zip(['lh','rh'],['Left insula','Right insula'])):
            ax=axes[gi,hi];ax.imshow(images[(group,hemi)]);ax.axis('off')
            a=int((selected.x.lt(0) if hemi=='lh' else selected.x.gt(0)).sum())
            n=int((known.x.lt(0) if hemi=='lh' else known.x.gt(0)).sum())
            ax.set_title(f'{hemilabel} · NMF W {n}/{a}',fontsize=8,pad=0)
        pos=axes[gi,0].get_position()
        fig.text(.015,(pos.y0+pos.y1)/2,
            f'{DISPLAY[group]}\n{len(known)}/{len(selected)} with NMF W\nUnassigned: {len(selected)-len(known)}',va='center',fontsize=9)
    fig.suptitle('Three response regions · original NMF weights',fontsize=12,y=.99)
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.62,.06),frameon=False,ncol=3,fontsize=9)
    fig.text(.5,.018,'Each region has one bilateral map. Mixed hues = local relative weights; opacity = sampling support.',ha='center',fontsize=7)
    paths['all_maps']=_save(fig,EXPORTS['all_maps'])
    return paths


def render():
    assert os.environ.get('SLURM_JOB_ID'),'Use Slurm'
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scripts.refresh_lexical_condition_sets import sha,write
    sites,summary,input_hashes=_prepare()
    plt.rcParams.update({'font.family':'Arial','font.size':8,'svg.fonttype':'none'})
    paths={'composition':_bars(sites,summary)}
    space,renderer_proof=_old_gaussian_renderer()
    images={};fields={};support=[]
    for group in GROUPS:
        selected=sites.loc[sites.response_set.eq(group)&sites.has_original_nmf_weight]
        xyz=selected[['x','y','z']].to_numpy(float)
        weights=selected[[f'proportion_{c}' for c in COMPONENTS]].to_numpy(float)
        for hemi in ['lh','rh']:
            take=xyz[:,0]<0 if hemi=='lh' else xyz[:,0]>0
            density=space['compute_cluster_densities'](space[f'{hemi}_pial_coords'],xyz[take],weights[take])
            fields[f'{DISPLAY[group]}_{hemi}']=density
            support.append(dict(response_region=DISPLAY[group],hemisphere=hemi,n_original_weights=int(take.sum()),
                n_supported_vertices=int((density.sum(axis=1)>0).sum())))
        print('Rendering ONE bilateral map:',DISPLAY[group],len(selected),'original NMF weights',flush=True)
        left,right=space['render_phase_brains_density'](selected,weights,overlay_mixed=False)
        for hemi,image in [('lh',left),('rh',right)]:
            assert image.shape[:2]==(800,800)
            images[(group,hemi)]=image
    np.savez_compressed(OUTPUT/'surface_densities.npz',**fields)
    pd.DataFrame(support).to_csv(OUTPUT/'spatial_support.csv',index=False)
    paths.update(_maps(sites,images))
    assert all(sha(p)==h for p,h in input_hashes.items())
    write(OUTPUT/'provenance.json',dict(n_regions=3,n_bilateral_maps=3,n_all_sites=49,n_original_NMF_weights=44,n_unassigned=5,
        grouping_window=[0,1],weight_source='Original frozen results/nmf/channel_assignments.csv loading_* columns',
        one_fixed_vector_per_electrode=True,no_PN_condition_split=True,no_condition_average=True,
        relative_weights='W_k/sum_j W_j within electrode',mean='Electrode-equal among available original W',
        unassigned_sites='Explicit separate list, not imputed, projected or relabeled',
        no_new_NMF_or_NNLS=True,no_new_inference=True,no_extra_HGA_multiplier=True,
        coordinates='Unchanged frozen PN response-set coordinates',renderer=renderer_proof,
        inputs=input_hashes,exports={str(p):sha(p) for p in paths.values()}))
    return paths
