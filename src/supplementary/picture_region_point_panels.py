"""Independent panels connecting PN response sets with frozen functional labels."""
import ast
import hashlib
import json
import os

from src.supplementary.picture_fixed_template_expression import ROOT,BASE,IMG,COLORS,LABELS

SOURCE=BASE/'region_original_nmf_weights_2026-10-27'
OUTPUT=BASE/'region_point_panels_2026-10-27'
AUDIT=ROOT/'results/audit/picture_region_point_panels_2026-10-27'
GROUPS=('Sound-only','Both','Picture-only')
DISPLAY={'Sound-only':'Sound-only','Both':'Both','Picture-only':'Image-only'}
KEYS={'Sound-only':'sound_only','Both':'both','Picture-only':'image_only'}
COMPONENTS=('sensory','sustain','motor')
UNASSIGNED_COLOR='#858585'
POINT_SIZE=24.
EXPORTS={'venn':'fig3_picture_response_regions_neutral_venn.svg'}
for _group in GROUPS:
    EXPORTS[f'{KEYS[_group]}_brain']=f'fig3_picture_{KEYS[_group]}_functional_points.svg'
    EXPORTS[f'{KEYS[_group]}_bar']=f'fig3_picture_{KEYS[_group]}_functional_weight_bar.svg'


def notebook_cells():
    import nbformat as nbf
    return [nbf.v4.new_markdown_cell('''## 6. Standalone panels: input response sets, functional composition and anatomy

This draft exports **independent subpanels only**, with no Venn/bar/brain assembly and no combined three-map sheet. The neutral Venn defines **Sound-only 25, Both 13, Image-only 11**, with 79 neither among the unchanged 128 eligible sites / 27 participants. Each region then has its own bilateral insula point panel and its own component-weight bar.

**Spatial panels:** use the exact Figure 1 local-insula pial surface, cameras, surface projection and 4-mm display lift; keep the accepted 24-pixel point size. Colors reuse each electrode's **existing frozen Figure 2 functional label**: Sensory blue, Sustained gold, Motor red. No new argmax rule, mixed category, clustering or threshold is introduced. The five responsive electrodes without an original NMF assignment are shown as **open gray circles** (2 / 1 / 2 per set), not as nonsignificant points. All 49 responsive sites retain their positions. No Gaussian overlay, RGB blending or density contour/fill is rendered. The original Figure 2 A–P / D–V orientation helper is reused, with anterior direction verified from projection of the positive-y anatomical direction.

**Composition panels:** each region's one 100% stacked horizontal bar reads the **already saved original-NMF mean proportions**. Its denominator is the 23 / 12 / 9 sites with original W, explicitly separated from the full 25 / 13 / 11 responsive-site denominator. Bars retain the same blue/gold/red component palette. The Venn uses neutral grays so that component colors keep one meaning throughout these subpanels.

Only display changes are made: no NMF/NNLS fit, response test, label or membership change, weight aggregation or bootstrap is run. Earlier drafts, scientific results and main figures remain unchanged. The point-projection audit checks all 49 centers against the previously accepted coordinate export.
''',id='picture-region-point-panels-note'),nbf.v4.new_code_cell('''from pathlib import Path
import sys
from IPython.display import SVG,display
_pn_point_root=Path('/cwork/ns458/insula-paper')
if str(_pn_point_root) not in sys.path:
    sys.path.insert(0,str(_pn_point_root))
from src.supplementary.picture_region_point_panels import render,OUTPUT as point_panel_data
point_panel_paths=render()
display(SVG(filename=str(point_panel_paths['venn'])))
for _group_key in ['sound_only','both','image_only']:
    display(SVG(filename=str(point_panel_paths[_group_key+'_bar'])))
    display(SVG(filename=str(point_panel_paths[_group_key+'_brain'])))
print('All panels are separate SVGs:',{k:str(v) for k,v in point_panel_paths.items()})
print('Displayed site list:',point_panel_data/'displayed_sites.csv')
''',id='picture-region-point-panels-display')]


def _save(fig,key):
    import matplotlib.pyplot as plt
    path=IMG/EXPORTS[key]
    fig.savefig(path,format='svg',bbox_inches=None,pad_inches=0)
    plt.close(fig)
    return path


def _orientation_helper():
    import numpy as np
    path=ROOT/'vizpub/fig2.ipynb'
    n=json.loads(path.read_text())
    source=''.join(next(c for c in n['cells'] if c.get('id')=='fig2-spatial-plot')['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='draw_ap_dv')
    space=dict(np=np,fontsize=6.)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path)+':orientation','exec'),space)
    proof=dict(source_notebook=str(path),source_cell='fig2-spatial-plot',copied_verbatim=True,
        function_sha256=hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest())
    return space['draw_ap_dv'],proof


def _neutral_venn(table):
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2,venn2_circles
    from matplotlib.patches import PathPatch
    sound=set(table.loc[table.sound_responsive,'electrode'])
    image=set(table.loc[table.image_responsive,'electrode'])
    counts=(len(sound-image),len(image-sound),len(sound&image))
    assert counts==(25,11,13) and len(table)==128
    cm=1/2.54
    fig,ax=plt.subplots(figsize=(8.6*cm,6.2*cm))
    fig.subplots_adjust(left=.08,right=.92,bottom=.21,top=.82)
    diagram=venn2(subsets=counts,set_labels=('Sound (n=38)','Image (n=24)'),ax=ax)
    for key,color,text in [('10','#E5E5E5','Sound-only\n25'),('01','#D8D8D8','Image-only\n11'),('11','#B8B8B8','Both\n13')]:
        patch=diagram.get_patch_by_id(key);patch.set_facecolor(color);patch.set_alpha(1)
        label=diagram.get_label_by_id(key);label.set_text(text);label.set_fontsize(7)
    for circle in venn2_circles(subsets=counts,ax=ax,lw=.8):circle.set_edgecolor('.35')
    shared=diagram.get_patch_by_id('11')
    ax.add_patch(PathPatch(shared.get_path(),transform=shared.get_transform(),facecolor='none',edgecolor='white',lw=3,zorder=2.5))
    for label in diagram.set_labels:label.set_fontsize(7);label.set_color('black')
    fig.suptitle('Picture Naming · Stimulus 0–1 s',fontsize=9,y=.98)
    fig.text(.5,.035,'128 sampled sites / 27 participants\nNeither detected: 79',ha='center',fontsize=7)
    return _save(fig,'venn')


def _weight_bar(group,summary,counts):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    cm=1/2.54
    fig,ax=plt.subplots(figsize=(8.6*cm,4.6*cm))
    left=0.
    for component in COMPONENTS:
        value=100*float(summary.loc[summary.response_set_source.eq(group)&summary.component.eq(component),'mean_proportion'].iloc[0])
        ax.barh(0,value,left=left,height=.7,color=COLORS[component],edgecolor='white',linewidth=.6)
        ax.text(left+value/2,0,f'{value:.1f}%',ha='center',va='center',fontsize=7)
        left+=value
    row=counts.loc[counts.response_set_source.eq(group)].iloc[0]
    ax.set_xlim(0,100);ax.set_ylim(-.65,.65);ax.set_yticks([]);ax.set_xticks([0,25,50,75,100])
    ax.tick_params(axis='x',labelsize=6,length=2)
    ax.spines[['top','left','right']].set_visible(False)
    ax.set_xlabel('Mean relative component weight (%)',fontsize=7,labelpad=3)
    fig.suptitle(DISPLAY[group],fontsize=9,y=.98)
    fig.legend(handles=[Patch(facecolor=COLORS[c],label=LABELS[c]) for c in COMPONENTS],
        loc='upper center',bbox_to_anchor=(.5,.86),ncol=3,frameon=False,fontsize=6.5,
        handletextpad=.3,columnspacing=.8)
    fig.text(.5,.035,f'{int(row.n_original_weights)}/{int(row.n_sites)} sites with original NMF W · Unassigned: {int(row.n_unassigned)}',
        ha='center',fontsize=6.5)
    fig.subplots_adjust(left=.10,right=.96,bottom=.31,top=.68)
    return _save(fig,KEYS[group]+'_bar')


def _point_map(group,sites,renderer,crop,draw_orientation,reference_positions):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    cm=1/2.54
    selected=sites.loc[sites.response_set.eq(group)].copy()
    fig,axes=plt.subplots(1,2,figsize=(8.6*cm,5.5*cm))
    fig.subplots_adjust(left=.03,right=.98,bottom=.24,top=.77,wspace=.03)
    fig.suptitle(f'{DISPLAY[group]} · {len(selected)} sites / {selected.subject.nunique()} participants\nPicture Naming · Stimulus 0–1 s',
        fontsize=8,y=.99)
    audit=[];pending=[];orientation=[]
    for ax,hemi,label in zip(axes,['lh','rh'],['Left insula','Right insula']):
        frame=selected.loc[selected.x.lt(0) if hemi=='lh' else selected.x.gt(0)]
        known=frame.has_original_nmf_weight.to_numpy(bool)
        xyz=frame[['x','y','z']].to_numpy(float)
        colors=np.array([COLORS[c] for c in frame.loc[known,'functional_cluster']],dtype=object)
        brain=None
        try:
            image,project,brain=renderer['_render_insula_hemi'](hemi,xyz[known],colors,np.full(int(known.sum()),POINT_SIZE))
            camera=renderer['_camera_xyz'](brain)
            surface=renderer[f'{hemi}_surf'];pial=renderer[f'{hemi}_pial_coords']
            displayed=renderer['_project_to_insula'](xyz,surface,pial,camera)
            raw_pixels=project(displayed)
            toward=camera-displayed;toward/=np.linalg.norm(toward,axis=1,keepdims=True)
            snapped=displayed-4.*toward
            residual,_=renderer['cKDTree'](pial[surface['verts']]).query(snapped)
            assert (residual<1e-8).all()
            center=surface['center']
            probes=project(np.stack([center+[0,5,0],center-[0,5,0],center+[0,0,5],center-[0,0,5]]))
            anterior_left=bool(probes[0,0]<probes[1,0])
            assert probes[2,1]<probes[3,1],'Verify dorsal/ventral orientation'
        finally:
            if brain is not None:brain.close()
        image,pixels=crop(image,raw_pixels)
        ax.imshow(image);ax.set_xlim(0,image.shape[1]);ax.set_ylim(image.shape[0],0);ax.axis('off')
        ax.set_title(f'{label} · n={len(frame)}',fontsize=7,pad=2)
        draw_orientation(ax,anterior_is_left=anterior_left,corner='lower left' if hemi=='lh' else 'lower right')
        orientation.append(dict(response_region=DISPLAY[group],hemisphere=hemi,anterior_is_left=anterior_left))
        pending.append((ax,pixels[~known],image.shape[1]))
        for i,row in enumerate(frame.itertuples()):
            audit.append(dict(response_region=DISPLAY[group],electrode=row.electrode,subject=row.subject,
                hemisphere=hemi,functional_cluster=row.functional_cluster,
                has_original_nmf_weight=bool(known[i]),display_style='filled original Figure 2 color' if known[i] else 'open gray circle',
                pixel_x=float(raw_pixels[i,0]),pixel_y=float(raw_pixels[i,1]),
                cropped_pixel_x=float(pixels[i,0]),cropped_pixel_y=float(pixels[i,1]),
                surface_residual_mm=float(residual[i]),display_lift_mm=4.))
    fig.canvas.draw()
    for ax,shift in zip(axes,(.025,-.025)):
        box=ax.get_position();ax.set_position([box.x0+shift,box.y0,box.width,box.height])
    fig.canvas.draw()
    for ax,pixels,image_width in pending:
        if len(pixels):
            # Match the 24 source-pixel marker diameter at the saved canvas scale.
            diameter_pt=POINT_SIZE*(ax.get_window_extent().width/image_width)*72/fig.dpi
            ax.scatter(pixels[:,0],pixels[:,1],s=diameter_pt**2,facecolors='none',edgecolors=UNASSIGNED_COLOR,
                linewidths=.8,zorder=6)
    handles=[Line2D([],[],ls='none',marker='o',ms=4,mfc=COLORS[c],mec=COLORS[c],
        label=f'{LABELS[c]} ({int(selected.functional_cluster.eq(c).sum())})') for c in COMPONENTS]
    unknown=int((~selected.has_original_nmf_weight).sum())
    handles.append(Line2D([],[],ls='none',marker='o',ms=4,mfc='none',mec=UNASSIGNED_COLOR,label=f'Unassigned ({unknown})'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.105),frameon=False,ncol=4,fontsize=6,
        handletextpad=.25,columnspacing=.6)
    fig.text(.5,.025,'Open gray: responsive sites outside original NMF',ha='center',fontsize=6.5)
    audit=pd.DataFrame(audit)
    assert len(audit)==len(selected) and set(audit.electrode)==set(selected.electrode)
    matched=audit.merge(reference_positions[['electrode','pixel_x','pixel_y']],on='electrode',validate='one_to_one',suffixes=('_new','_old'))
    for coord in ['pixel_x','pixel_y']:
        np.testing.assert_allclose(matched[coord+'_new'],matched[coord+'_old'],atol=1e-8,rtol=0)
    path=_save(fig,KEYS[group]+'_brain')
    return path,audit,orientation


def render():
    assert os.environ.get('SLURM_JOB_ID'),'Use Slurm'
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scripts.refresh_lexical_condition_sets import sha,write
    from src.supplementary.decision_repeat_maps import _figure1_local_insula_renderer
    from src.supplementary.selected_brain_contours import _figure2_contour_helpers
    sources=[SOURCE/n for n in ['electrode_weights.csv','mean_proportions.csv','region_counts.csv']]
    sources += [BASE/'electrodes.csv',ROOT/'results/nmf/channel_assignments.csv',BASE/'baseline_classes_display_coordinates.csv']
    hashes={str(p):sha(p) for p in sources}
    sites=pd.read_csv(SOURCE/'electrode_weights.csv',float_precision='round_trip')
    means=pd.read_csv(SOURCE/'mean_proportions.csv',float_precision='round_trip')
    counts=pd.read_csv(SOURCE/'region_counts.csv')
    table=pd.read_csv(BASE/'electrodes.csv')
    reference_positions=pd.read_csv(BASE/'baseline_classes_display_coordinates.csv')
    assignments=pd.read_csv(ROOT/'results/nmf/channel_assignments.csv')
    labels=assignments.set_index(['subject','channel']).functional_cluster
    assert len(sites)==49 and sites.electrode.is_unique and sites.has_original_nmf_weight.sum()==44
    assert set(sites.electrode)==set(table.loc[table.sound_responsive|table.image_responsive,'electrode'])
    for row in sites.itertuples():
        if row.has_original_nmf_weight:assert row.functional_cluster==labels.loc[(row.subject,row.channel)]
        else:assert row.functional_cluster=='Unassigned'
    plt.rcParams.update({'font.family':'Arial','font.size':7,'svg.fonttype':'none'})
    OUTPUT.mkdir(parents=True,exist_ok=True)
    paths={'venn':_neutral_venn(table)}
    renderer,renderer_proof=_figure1_local_insula_renderer(ROOT)
    helpers,crop_proof=_figure2_contour_helpers(ROOT)
    draw_orientation,orientation_proof=_orientation_helper()
    audits=[];orientations=[]
    for group in GROUPS:
        paths[KEYS[group]+'_bar']=_weight_bar(group,means,counts)
        paths[KEYS[group]+'_brain'],audit,ori=_point_map(group,sites,renderer,helpers['crop_to_content'],draw_orientation,reference_positions)
        audits.append(audit);orientations.extend(ori)
    audit=pd.concat(audits,ignore_index=True)
    assert len(audit)==49 and audit.electrode.is_unique and audit.has_original_nmf_weight.sum()==44
    assert audit.display_style.eq('open gray circle').sum()==5
    audit.to_csv(OUTPUT/'displayed_sites.csv',index=False)
    counts.to_csv(OUTPUT/'region_counts.csv',index=False)
    pd.DataFrame(orientations).to_csv(OUTPUT/'anatomical_orientation.csv',index=False)
    assert all(sha(p)==h for p,h in hashes.items()),'An original data file or saved mean changed'
    write(OUTPUT/'provenance.json',dict(n_response_regions=3,n_bilateral_brain_panels=3,n_bar_panels=3,n_venn_panels=1,
        no_composite_created=True,n_displayed_sites=49,n_existing_assignments=44,n_open_gray_unassigned=5,
        grouping_window=[0,1],colors=COLORS,labels='Original frozen Figure 2 functional_cluster',
        weights='Already saved mean proportions of original NMF W',
        no_Gaussian_or_RGB_overlay=True,no_KDE_contours_or_fills=True,no_reclassification=True,
        no_model_fit_or_new_statistics=True,point_pixels=POINT_SIZE,analytic_coordinates_unchanged=True,
        all_49_pixel_centers_match_previous_accepted_export=True,renderer=renderer_proof,
        crop_reference=crop_proof,orientation=orientation_proof,inputs=hashes,
        exports={str(p):sha(p) for p in paths.values()}))
    return paths
