"""Continuous component composition of the three PN baseline-response sets."""
from pathlib import Path
import ast
import hashlib
import json
import os

from src.supplementary.picture_fixed_template_expression import ROOT,BASE,DATA,IMG,COLORS,LABELS

AUDIT=ROOT/'results/audit/picture_component_proportions_2026-10-27'
OUTPUT=BASE/'component_proportions_2026-10-27'
GROUPS=('Sound-only','Both','Picture-only')
COMPONENTS=('sensory','sustain','motor')
CONDITIONS=('Sound','Image')
EXPORTS={
    'composition':'fig3_picture_response_sets_component_proportions.svg',
    'sound_only':'fig3_picture_sound_only_component_gaussian.svg',
    'both':'fig3_picture_both_component_gaussian.svg',
    'picture_only':'fig3_picture_picture_only_component_gaussian.svg',
    'all_maps':'fig3_picture_response_sets_component_gaussian.svg',
}


def notebook_cells():
    import nbformat as nbf
    return [nbf.v4.new_markdown_cell('''## 5. Component composition within Sound-only, Both and Picture-only

The comparison unit is now the **Venn response set**, retaining all 49 existing sites / 22 participants: Sound-only **25 / 17**, Both **13 / 10**, Picture-only **11 / 8**. These overlapping participant counts are not additive. Set membership remains the existing native baseline response in Stimulus **0–1 s**. Each response set is characterized separately under **Sound and Image**, using exactly the same sites in its two conditions.

Use the already saved PN-specific fixed-template expression coefficients, without rerunning NNLS or NMF. For each electrode and condition compute **p_k = W_k / sum_j W_j** across Sensory, Sustained and Motor. All three coefficients jointly summarize the frozen Stimulus/Delay/Response template windows (0–0.6 / 0–1 / −0.5–1 s); this is not a separate stimulus-only decomposition. The five originally Unassigned sites have continuous projection coefficients and enter these summaries while retaining their original labels elsewhere. Zero coefficient sums would remain undefined and be reported, never replaced with equal shares.

### Overall composition

Each response set has two **100% stacked bars**, Sound and Image, with Sensory blue, Sustained gold and Motor red. Bars average the **electrode-level proportions with equal electrode weight**; they are not ratios of summed raw coefficients. Every electrode therefore contributes one total unit regardless of absolute HGA amplitude. These are continuous coefficient shares, not numbers of electrodes assigned to a component or explained-variance percentages. No new confidence interval or hypothesis test is added.

### Gaussian spatial composition

Reuse the old `notebooks/functional_hga_specificity.ipynb` proportion-map helpers verbatim: Gaussian sigma **5 mm**, support restricted to vertices within **6 mm of the nearest electrode**, and the same template pial surface, anatomical labels and bilateral camera views. At each supported surface vertex, compute three weighted sums **d_k(v) = sum_i K(v,i) p_i,k**, then blend the three component colors using **d_k / sum_j d_j**. Mixed hues represent the local component mixture across neighboring electrodes. Each electrode contributes to all three continuous maps; no hard cluster or specificity suppression is applied. The old additional multiplication by phase HGA is deliberately omitted for this relative-weight question.

The original helper's opacity is total kernel support scaled to its per-map P95. It conveys local sampling support and rendering visibility, **not HGA amplitude, statistical significance or absolute strength across groups**. Because each electrode's proportions sum to one, Sound and Image have the same total support and opacity within each response set. Numeric vertex fields and support audits are saved. No nonsignificant context points or gray Unassigned-point layer is added.

The three individual exports each contain a Sound bilateral row and an Image bilateral row. The overview has three response-set rows and two condition blocks. All previous figures, statistical results, raw coefficients, labels, coordinates, Figure 1/2 and NMF remain unchanged.
''',id='picture-component-proportions-note'),nbf.v4.new_code_cell('''from pathlib import Path
import sys
from IPython.display import SVG,display
_pn_composition_root=Path('/cwork/ns458/insula-paper')
if str(_pn_composition_root) not in sys.path:
    sys.path.insert(0,str(_pn_composition_root))
from src.supplementary.picture_component_proportions import render,OUTPUT as composition_data
composition_paths=render()
display(SVG(filename=str(composition_paths['composition'])))
display(SVG(filename=str(composition_paths['all_maps'])))
import pandas as pd
display(pd.read_csv(composition_data/'mean_proportions.csv'))
print('Individual response-set maps:',{k:str(v) for k,v in composition_paths.items() if k not in ['composition','all_maps']})
print('Per-electrode proportions:',composition_data/'electrode_proportions.csv')
''',id='picture-component-proportions-display')]


def _prepare():
    import numpy as np
    import pandas as pd
    from scripts.refresh_lexical_condition_sets import sha,write
    sources=[DATA/n for n in ['sites.csv','coefficients.csv','projection_arrays.npz','projection_provenance.json']]
    hashes={str(p):sha(p) for p in sources}
    sites=pd.read_csv(DATA/'sites.csv',float_precision='round_trip')
    saved=pd.read_csv(DATA/'coefficients.csv',float_precision='round_trip')
    with np.load(DATA/'projection_arrays.npz',allow_pickle=False) as arrays:
        W=arrays['expression_coefficients'].copy()
        normalized=arrays['normalized_weights'].copy()
        names=tuple(arrays['component_names'].tolist())
        np.testing.assert_array_equal(arrays['electrodes'],sites.electrode.to_numpy())
    assert W.shape==(49,2,3) and sites.electrode.is_unique and sites.subject.nunique()==22
    assert sites.response_set.value_counts().to_dict()=={'Sound-only':25,'Both':13,'Picture-only':11}
    assert sites.functional_cluster.eq('Unassigned').sum()==5
    assert np.isfinite(W).all() and (W>=0).all()
    order=[names.index(c) for c in COMPONENTS]
    W=W[:,:,order];normalized=normalized[:,:,order]
    totals=W.sum(axis=2,keepdims=True)
    assert (totals>0).all(),'Undefined zero-sum coefficient vector; report before drawing'
    proportions=W/totals
    np.testing.assert_allclose(proportions.sum(axis=2),1,atol=1e-14,rtol=0)
    # The within-electrode shares should not depend on prior input L2 scaling.
    np.testing.assert_allclose(proportions,normalized/normalized.sum(axis=2,keepdims=True),atol=1e-9,rtol=1e-8)
    index=saved.set_index(['electrode','condition','component']).expression_coefficient
    rows=[]
    for i,site in enumerate(sites.itertuples()):
        for j,condition in enumerate(CONDITIONS):
            for k,component in enumerate(COMPONENTS):
                assert W[i,j,k]==index.loc[(site.electrode,condition,component)]
                rows.append(dict(electrode=site.electrode,subject=site.subject,channel=site.channel,
                    response_set=site.response_set,condition=condition,component=component,
                    coefficient=W[i,j,k],proportion=proportions[i,j,k],
                    frozen_functional_cluster=site.functional_cluster,x=site.x,y=site.y,z=site.z))
    long=pd.DataFrame(rows)
    summary=long.groupby(['response_set','condition','component'],sort=False).agg(
        mean_proportion=('proportion','mean'),n_sites=('electrode','nunique'),
        n_participants=('subject','nunique')).reset_index()
    np.testing.assert_allclose(summary.groupby(['response_set','condition']).mean_proportion.sum(),1,atol=1e-14,rtol=0)
    assert len(long)==294 and len(summary)==18
    OUTPUT.mkdir(parents=True,exist_ok=True)
    long.to_csv(OUTPUT/'electrode_proportions.csv',index=False)
    summary.to_csv(OUTPUT/'mean_proportions.csv',index=False)
    write(OUTPUT/'input_sources.json',hashes)
    return sites,proportions,summary,hashes


def _save(fig,name):
    import matplotlib.pyplot as plt
    path=IMG/name;fig.savefig(path,format='svg',bbox_inches=None,pad_inches=0);plt.close(fig)
    return path


def _composition(sites,summary):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig,ax=plt.subplots(figsize=(7.4,4.2))
    for gi,group in enumerate(GROUPS):
        for ci,condition in enumerate(CONDITIONS):
            x=gi+(-.17 if ci==0 else .17);bottom=0.
            for component in COMPONENTS:
                value=100*float(summary.loc[summary.response_set.eq(group)&summary.condition.eq(condition)&summary.component.eq(component),'mean_proportion'].iloc[0])
                ax.bar(x,value,width=.28,bottom=bottom,color=COLORS[component],edgecolor='white',linewidth=.6)
                if value>=8:
                    ax.text(x,bottom+value/2,f'{value:.1f}%',ha='center',va='center',fontsize=8,color='black')
                bottom+=value
            np.testing.assert_allclose(bottom,100,atol=1e-10,rtol=0)
            ax.text(x,-4,condition,ha='center',va='top',fontsize=8)
    ticks=[]
    for group in GROUPS:
        selected=sites.loc[sites.response_set.eq(group)]
        ticks.append(f'{group}\n{len(selected)} sites · {selected.subject.nunique()} participants')
    ax.set_xticks(range(3),ticks,fontsize=9);ax.tick_params(axis='x',pad=25,length=0)
    ax.set_ylim(0,100);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Mean relative component weight (%)')
    ax.set_xlim(-.6,2.6);ax.spines[['top','right']].set_visible(False)
    handles=[Patch(facecolor=COLORS[c],label=LABELS[c]) for c in COMPONENTS]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,.91),ncol=3,frameon=False,fontsize=9)
    fig.suptitle('Picture Naming · component composition within each response set',fontsize=11,y=.99)
    fig.text(.5,.035,'Each electrode contributes one unit. Groups: Stimulus 0–1 s baseline responses. Frozen NMF templates.',
        ha='center',fontsize=8)
    fig.subplots_adjust(left=.12,right=.98,bottom=.25,top=.78)
    return _save(fig,EXPORTS['composition'])


def _old_gaussian_renderer():
    """Extract only the reference's functions; never execute its analyses."""
    import numpy as np
    import pandas as pd
    import matplotlib.colors as mcolors
    import mne
    import pyvista as pv
    from mne.viz import Brain
    from scipy.spatial.distance import cdist
    from scripts.refresh_lexical_condition_sets import sha
    path=ROOT/'notebooks/functional_hga_specificity.ipynb'
    notebook=json.loads(path.read_text())
    sources={c.get('id'):''.join(c['source']) for c in notebook['cells']}
    wanted={'get_insula_center','make_insula_brains','compute_cluster_densities','densities_to_rgba',
            'add_density_rgba_mesh','render_phase_brains_density'}
    functions=[];proofs={}
    for cell_id in ['brain-setup','render']:
        source=sources[cell_id]
        for node in ast.parse(source).body:
            if isinstance(node,ast.FunctionDef) and node.name in wanted:
                functions.append(node)
                proofs[node.name]=dict(cell_id=cell_id,sha256=hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest())
    assert {f.name for f in functions}==wanted
    config=ast.parse(sources['config']);constants={}
    for node in config.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            if node.targets[0].id in ['DENSITY_BANDWIDTH','DENSITY_MAX_DISTANCE','recon_dir']:
                constants[node.targets[0].id]=ast.literal_eval(node.value)
    assert constants['DENSITY_BANDWIDTH']==5 and constants['DENSITY_MAX_DISTANCE']==6
    pv.OFF_SCREEN=True;mne.viz.set_3d_backend('notebook')
    labels=mne.read_labels_from_annot(subject='cvs_avg35_inMNI152',parc='aparc.a2009s',surf_name='pial',
                                    hemi='both',subjects_dir=constants['recon_dir'])
    patterns=['G_insular_short','G_Ins_lg_and_S_cent_ins','S_circular_insula_ant','S_circular_insula_inf','S_circular_insula_sup']
    space=dict(np=np,pd=pd,pv=pv,Brain=Brain,cdist=cdist,CLUSTERS=COMPONENTS,
        CLUSTER_RGB=np.array([mcolors.to_rgb(COLORS[c]) for c in COMPONENTS]),labels=labels,insula_patterns=patterns,**constants)
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path)+':reference functions','exec'),space)
    surface_sources={}
    for hemi in ['lh','rh']:
        surface=Path(constants['recon_dir'])/'cvs_avg35_inMNI152/surf'/f'{hemi}.pial'
        pial,faces=mne.read_surface(surface)
        space[f'{hemi}_pial_coords']=pial
        space[f'{hemi}_insula_center']=space['get_insula_center'](hemi,pial)
        surface_sources[str(surface)]=sha(surface)
    proof=dict(source_notebook=str(path),source_sha256=sha(path),functions=proofs,copied_verbatim=True,
        source_analysis_cells_executed=False,constants=constants,alpha_percentile=95,
        surfaces=surface_sources,colors=COLORS,component_order=COMPONENTS,
        original_HGA_multiplier_omitted=True,specificity_not_used=True)
    return space,proof


def _map_figures(sites,images):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    handles=[Patch(facecolor=COLORS[c],label=LABELS[c]) for c in COMPONENTS]
    paths={}
    for group,key in zip(GROUPS,['sound_only','both','picture_only']):
        selected=sites.loc[sites.response_set.eq(group)]
        fig,axes=plt.subplots(2,2,figsize=(4.7,5.4))
        for ci,condition in enumerate(CONDITIONS):
            for hi,hemi in enumerate(['lh','rh']):
                ax=axes[ci,hi];ax.imshow(images[(group,condition,hemi)]);ax.axis('off')
                n=int((selected.x.lt(0) if hemi=='lh' else selected.x.gt(0)).sum())
                ax.set_title(f'{condition} · {"Left" if hemi=="lh" else "Right"} · n={n}',fontsize=8,pad=0)
        fig.suptitle(f'{group} · {len(selected)} sites / {selected.subject.nunique()} participants',fontsize=10,y=.99)
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.065),frameon=False,ncol=3,fontsize=8)
        fig.text(.5,.025,'Relative weights · mixed hues = local component mixture\nOpacity reflects sampling support; no HGA multiplier.',ha='center',fontsize=7)
        fig.subplots_adjust(left=.01,right=.99,bottom=.13,top=.91,hspace=.02,wspace=.01)
        paths[key]=_save(fig,EXPORTS[key])
    fig,axes=plt.subplots(3,4,figsize=(10.6,8.0))
    for gi,group in enumerate(GROUPS):
        selected=sites.loc[sites.response_set.eq(group)]
        for ci,condition in enumerate(CONDITIONS):
            for hi,hemi in enumerate(['lh','rh']):
                ax=axes[gi,ci*2+hi];ax.imshow(images[(group,condition,hemi)]);ax.axis('off')
                if gi==0:ax.set_title(f'{condition} · {"Left" if hemi=="lh" else "Right"}',fontsize=9,pad=2)
        pos=axes[gi,0].get_position()
        # A stable figure-coordinate row label is added after final layout below.
    fig.suptitle('Picture Naming · Gaussian maps of relative component weights',fontsize=12,y=.99)
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.55,.058),frameon=False,ncol=3,fontsize=9)
    fig.text(.55,.017,'Mixed hues = local component mixture. Opacity = kernel sampling support. Same sites in Sound and Image.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.16,right=.99,bottom=.12,top=.92,hspace=.01,wspace=.01)
    for gi,group in enumerate(GROUPS):
        selected=sites.loc[sites.response_set.eq(group)];p=axes[gi,0].get_position()
        fig.text(.01,(p.y0+p.y1)/2,f'{group}\nn={len(selected)}\n{selected.subject.nunique()} participants',fontsize=9,va='center')
    paths['all_maps']=_save(fig,EXPORTS['all_maps'])
    return paths


def render():
    assert os.environ.get('SLURM_JOB_ID'),'Use Slurm'
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scripts.refresh_lexical_condition_sets import sha,write
    from scipy.spatial import cKDTree
    OUTPUT.mkdir(parents=True,exist_ok=True);IMG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Arial','font.size':8,'svg.fonttype':'none'})
    sites,proportions,summary,input_hashes=_prepare()
    paths={'composition':_composition(sites,summary)}
    space,renderer_proof=_old_gaussian_renderer()
    images={};surface_fields={};support_records=[];distance_records=[]
    for group in GROUPS:
        selected=sites.loc[sites.response_set.eq(group)]
        indices=selected.index.to_numpy();weights=proportions[indices]
        xyz=selected[['x','y','z']].to_numpy(float)
        assert np.isfinite(xyz).all() and (xyz[:,0]!=0).all()
        fields={}
        for hemi in ['lh','rh']:
            take=xyz[:,0]<0 if hemi=='lh' else xyz[:,0]>0
            pial=space[f'{hemi}_pial_coords']
            fields[hemi]=[space['compute_cluster_densities'](pial,xyz[take],weights[take,j]) for j in range(2)]
            np.testing.assert_allclose(fields[hemi][0].sum(axis=1),fields[hemi][1].sum(axis=1),atol=1e-12,rtol=1e-12)
            for j,condition in enumerate(CONDITIONS):
                rgba=space['densities_to_rgba'](fields[hemi][j]);surface_fields[f'{group}_{condition}_{hemi}']=fields[hemi][j]
                if j==0:alpha=rgba[:,3]
                else:np.testing.assert_allclose(alpha,rgba[:,3],atol=1e-12,rtol=1e-12)
                support_records.append(dict(response_set=group,condition=condition,hemisphere=hemi,
                    n_sites=int(take.sum()),n_supported_vertices=int((fields[hemi][j].sum(axis=1)>0).sum()),
                    n_vertices=len(pial),alpha_percentile=95))
            for site,distance in zip(selected.loc[take].itertuples(),cKDTree(pial).query(xyz[take])[0]):
                distance_records.append(dict(response_set=group,electrode=site.electrode,hemisphere=hemi,
                    nearest_pial_mm=float(distance),within_6mm_nearest_pial=bool(distance<6),
                    electrode_retained_in_proportion_summary=True))
        for j,condition in enumerate(CONDITIONS):
            print('Rendering',group,condition,len(selected),'sites',flush=True)
            # Both the mathematical blending and camera/surface renderer are the old helper.
            left,right=space['render_phase_brains_density'](selected,weights[:,j],overlay_mixed=False)
            for hemi,image in [('lh',left),('rh',right)]:
                assert image.shape[:2]==(800,800)
                images[(group,condition,hemi)]=image
    np.savez_compressed(OUTPUT/'surface_densities.npz',**surface_fields)
    pd.DataFrame(support_records).to_csv(OUTPUT/'spatial_support.csv',index=False)
    pd.DataFrame(distance_records).to_csv(OUTPUT/'electrode_surface_distances.csv',index=False)
    paths.update(_map_figures(sites,images))
    assert all(sha(p)==h for p,h in input_hashes.items()),'An original coefficient or source changed'
    write(OUTPUT/'provenance.json',dict(n_sites=49,n_participants=22,groups={g:int(sites.response_set.eq(g).sum()) for g in GROUPS},
        grouping_window=[0,1],coefficient_input='Saved PN-only fixed-H NNLS coefficients; three phases jointly',
        coefficient_proportions='W_k / sum_j W_j within each electrode and condition',
        bar_summary='Electrode-equal mean of per-electrode proportions',new_NMF_fit=False,new_NNLS_fit=False,
        new_bootstrap=False,new_hypothesis_test=False,extra_HGA_multiplier=False,specificity_transform=False,
        original_unassigned_sites_retained=5,sound_image_same_sites_same_support_and_opacity=True,
        renderer=renderer_proof,inputs=input_hashes,exports={str(p):sha(p) for p in paths.values()}))
    return paths
