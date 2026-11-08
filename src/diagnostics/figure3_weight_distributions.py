"""Empirical component-loading distributions for the selected Figure 3 groups."""
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap,Normalize,to_rgb,to_hex
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
DENSITY_SIGMA=.09
NMF_DIR=ROOT/'results/nmf'
GROUP_COUNTS=None  # Explicit current-Figure3 override; legacy diagnostics stay versioned.


def group_counts(task,region,n,subjects):
    return (n,subjects) if GROUP_COUNTS is None else GROUP_COUNTS[(task,region)]


def composition_density(xy,xx,yy,sigma=DENSITY_SIGMA):
    """Equal-electrode Gaussian display density in the unchanged simplex plane."""
    density=np.zeros_like(xx)
    for x,y in xy:
        density+=np.exp(-((xx-x)**2+(yy-y)**2)/(2*sigma**2))
    return density/(len(xy)*2*np.pi*sigma**2)


def render(*,sound_gt_image=False):
    assert os.environ.get('SLURM_JOB_ID'),'Run through Slurm'
    results=RESULTS/'sound_gt_image' if sound_gt_image else RESULTS
    svg=IMAGES/('fig3_sound_gt_image_weight_ternary.svg' if sound_gt_image else 'fig3_component_weight_ternary.svg')
    n_sound,s_sound=group_counts('picture','first_only',18,12)
    groups=(('picture','first_only',n_sound,s_sound,'Sound–Image\nSound > Image only'),) if sound_gt_image else GROUPS
    reference_peak=None
    if sound_gt_image:
        import json
        reference_peak=json.loads((RESULTS/'manifest.json').read_text())['common_density_peak']
    results.mkdir(parents=True,exist_ok=True);IMAGES.mkdir(parents=True,exist_ok=True)
    original_paths=[p for folder in ('results/nmf','results/fig3','img/fig3')
        for p in (ROOT/folder).rglob('*') if p.is_file() and p!=svg and results not in p.parents]
    original_paths += [ROOT/f'vizpub/fig{i}.ipynb' for i in range(1,7)]
    original_paths += [ROOT/'vizpub/supplementary/fig3_supplementary.ipynb']
    protected={str(p):sha(p) for p in original_paths}
    assignments=NMF_DIR/'channel_assignments.csv'
    w=pd.read_csv(assignments,float_precision='round_trip')
    columns=[f'loading_{c}' for c in COMPONENTS]
    with np.load(NMF_DIR/'factors.npz',allow_pickle=False) as factors:
        np.testing.assert_allclose(np.linalg.norm(factors['H'],axis=1),1,rtol=0,atol=1e-12)
        assert w.channel.tolist()==factors['channels'].tolist()
        for index,name in enumerate(factors['component_names'].tolist()):
            np.testing.assert_allclose(w[f'loading_{name}'],factors['W'][:,index],rtol=0,atol=1e-12)
    style()
    fig,axes=plt.subplots(1,len(groups),squeeze=False,figsize=(8.8*len(groups)/2.54,10.4/2.54))
    axes=axes[0]
    fig.subplots_adjust(left=.035,right=.965,bottom=.27,top=.77,wspace=.23)
    records=[];summary=[];coordinates=[];group_means=[];color_records=[]
    inputs={str(assignments):sha(assignments)}
    height=np.sqrt(3)/2
    # Columns follow COMPONENTS: Sensory, Sustained, Motor.
    vertices=np.array([[.5,height],[0.,0.],[1.,0.]])
    center=vertices.mean(axis=0)
    component_rgb=np.array([to_rgb(color) for color in COLORS])
    xx,yy=np.meshgrid(np.linspace(0,1,301),np.linspace(0,height,261))
    sensory_grid=yy/height;motor_grid=xx-.5*sensory_grid
    valid=(sensory_grid>=0)&(motor_grid>=0)&(sensory_grid+motor_grid<=1)
    clouds=[];density_cache=dict(x=xx,y=yy,valid_simplex=valid)
    for axis,(key,region,n,subjects,title) in zip(axes,groups):
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
        density=composition_density(xy,xx,yy)
        clouds.append((axis,key,density))
        density_cache[key+'_density']=density
        outline=np.vstack([vertices,vertices[0]])
        axis.fill(outline[:,0],outline[:,1],facecolor='#FAFBFD',edgecolor='none',zorder=-1)
        axis.plot(outline[:,0],outline[:,1],color='#CBD2DD',lw=.7,zorder=2)
        for component,vertex in zip(COMPONENTS,vertices):
            spoke,=axis.plot([center[0],vertex[0]],[center[1],vertex[1]],
                color='#A2A9B2',lw=.7,zorder=2)
            spoke.set_gid(key+'_center_to_'+component+'_vertex')
        for fraction in (.5,):
            axis.text(.5*fraction-.025,height*fraction,f'{fraction:.0%}',
                ha='right',va='center',fontsize=5.5,color='#9299A2')
            axis.text(1-fraction,-.025,f'{fraction:.0%}',
                ha='center',va='top',fontsize=5.5,color='#9299A2')
            axis.text(.5+.5*fraction+.025,height*(1-fraction),f'{fraction:.0%}',
                ha='left',va='center',fontsize=5.5,color='#9299A2')
        point_rgb=proportions@component_rgb
        points=axis.scatter(xy[:,0],xy[:,1],s=25,facecolor=point_rgb,alpha=.95,
            edgecolor='white',linewidth=.6,zorder=3,clip_on=False)
        points.set_gid(key+'_original_electrodes')
        mean=proportions.mean(axis=0);mean_xy=mean@vertices
        group_means.append(dict(group=key,sensory=float(mean[0]),sustain=float(mean[1]),motor=float(mean[2]),
            ternary_x=float(mean_xy[0]),ternary_y=float(mean_xy[1])))
        for i,row in enumerate(frame.itertuples()):
            color_records.append(dict(group=key,electrode=row.electrode,
                sensory=float(proportions[i,0]),sustain=float(proportions[i,1]),motor=float(proportions[i,2]),
                red=float(point_rgb[i,0]),green=float(point_rgb[i,1]),blue=float(point_rgb[i,2]),
                color_hex=to_hex(point_rgb[i])))
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
        axis.text(.5,height+.055,'Sensory',ha='center',va='bottom',fontsize=8,color=COLORS[0])
        axis.text(-.005,-.10,'Sustained',ha='center',va='top',fontsize=8,color=COLORS[1])
        axis.text(1.005,-.10,'Motor',ha='center',va='top',fontsize=8,color=COLORS[2])
        axis.set_title(f'{title}\n{n} electrodes / {subjects} participants',fontsize=9,pad=16)
        axis.set(xlim=(-.15,1.15),ylim=(-.20,height+.20),aspect='equal')
        axis.axis('off')
        inputs[str(source)]=sha(source)
    density_peak=reference_peak if reference_peak is not None else max(float(density[valid].max()) for _,_,density in clouds)
    cmap=LinearSegmentedColormap.from_list('composition_cloud',
        ['#FAFBFC','#F0F1F3','#DEE1E6','#C5CBD4','#A2ACBB'])
    for axis,key,density in clouds:
        relative=density/density_peak
        density_cache[key+'_relative_density']=relative
        masked=np.ma.masked_where(~valid,relative)
        fill=axis.contourf(xx,yy,masked,levels=np.linspace(.06,1,17),cmap=cmap,
            vmin=0,vmax=1,zorder=.5,antialiased=True,extend='max' if relative[valid].max()>1 else 'neither')
        fill.set_gid(key+'_composition_density_cloud')
        contours=axis.contour(xx,yy,masked,levels=[.2,.4,.6,.8],
            colors='#858F9E',linewidths=.55,alpha=.40,zorder=1)
        contours.set_gid(key+'_descriptive_density_contours')
    color_axis=fig.add_axes([.22 if sound_gt_image else .36,.215 if sound_gt_image else .175,.56 if sound_gt_image else .28,.012])
    colorbar=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap=cmap),cax=color_axis,orientation='horizontal')
    colorbar.set_ticks([0,.5,1]);colorbar.ax.tick_params(labelsize=6,length=2,pad=2)
    colorbar.set_label('Relative electrode density',fontsize=7,labelpad=2)
    colorbar.outline.set_visible(False)
    handles=[Line2D([],[],marker='o',linestyle='none',markersize=4.5,color=color,label=label)
        for color,label in zip(COLORS,LABELS)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.055 if sound_gt_image else .065),ncol=3,frameon=False,
        fontsize=7,columnspacing=1.1,handletextpad=.4)
    footer='Point colors blend the three component weights; gray clouds show relative electrode density.'
    if sound_gt_image:footer='Point colors blend component weights; gray clouds show density.\nOne bidirectionally significant electrode is shown separately in the main figure.'
    fig.text(.5,.015 if sound_gt_image else .025,footer,
        ha='center',fontsize=5.7 if sound_gt_image else 6.7)
    svg=IMAGES/('fig3_sound_gt_image_weight_ternary.svg' if sound_gt_image else 'fig3_component_weight_ternary.svg')
    fig.savefig(svg,format='svg',bbox_inches=None);plt.close(fig)
    pd.DataFrame(records).to_csv(results/'electrode_weights_long.csv',index=False)
    pd.DataFrame(summary).to_csv(results/'descriptive_summary.csv',index=False)
    pd.DataFrame(coordinates).to_csv(results/'ternary_coordinates.csv',index=False)
    pd.DataFrame(group_means).to_csv(results/'mean_compositions.csv',index=False)
    pd.DataFrame(color_records).to_csv(results/'electrode_colors.csv',index=False)
    np.savez_compressed(results/'composition_density_grid.npz',**density_cache)
    assert all(sha(p)==digest for p,digest in protected.items())
    write(results/'manifest.json',dict(passed=True,job_id=os.environ['SLURM_JOB_ID'],
        svg=str(svg),svg_sha256=sha(svg),source_hashes=inputs,
        n_electrodes={key:n for key,_,n,_,_ in groups},n_participants={key:subjects for key,_,_,subjects,_ in groups},
        selected_regions={key:region for key,region,_,_,_ in groups},
        bidirectional_electrode_excluded=sound_gt_image,
        density_reference_manifest=str(RESULTS/'manifest.json') if sound_gt_image else None,
        proportion_definition='Each frozen W loading divided by the sum of the three W loadings for that electrode',
        visualization='Ternary composition landscape with continuous component-color electrodes and neutral-gray density clouds',
        point_color_anchors=dict(zip(COMPONENTS,COLORS)),point_color_alpha=.95,
        point_color_definition='sRGB weighted blend: RGB = proportion vector @ original component RGB anchors',
        no_argmax_color_assignment=True,density_color_is_neutral_gray=True,
        triangle_vertices=dict(zip(COMPONENTS,vertices.tolist())),grid_spacing=None,edge_proportion_labels=[.5],
        dashed_line=None,mean_marker=None,
        interior_guides='Three solid segments from the triangle centroid to its vertices',
        guide_center=center.tolist(),guide_endpoints=vertices.tolist(),
        guide_sections_are_not_argmax_functional_classes=True,
        projection_inverted_to_original_proportions=True,no_coordinate_jitter=True,point_coordinates_unsmoothed=True,
        density_sigma_in_triangle_side_units=DENSITY_SIGMA,density_bandwidth_shared_across_groups=True,
        density_definition='Equal-electrode isotropic Gaussian display kernels, clipped to valid simplex',
        density_scaling='One shared peak after dividing each group kernel sum by its electrode count',
        common_density_peak=density_peak,relative_density_contour_levels=[.2,.4,.6,.8],
        density_is_descriptive_not_inferential=True,no_manifold_learning_or_embedding=True,
        common_proportion_range=[0,1],
        no_inferential_tests=True,no_NMF_refit=True,
        original_files_unchanged=len(protected),main_and_supplementary_figures_unchanged=True))
    (IMAGES/'fig3_component_weight_distributions.svg').unlink(missing_ok=True)
    (RESULTS/'density_curves.csv').unlink(missing_ok=True)
    return svg


def _cached_weight_panel(source,group,region,n,subjects,title):
    """Reuse one already exported ternary group without recomputing its cloud."""
    coordinate_path=source/'ternary_coordinates.csv'
    color_path=source/'electrode_colors.csv'
    density_path=source/'composition_density_grid.npz'
    selection_path=ROOT/'results/fig3/regions'/('picture' if group=='picture' else 'lexical')/f'{region}_electrodes.csv'
    frame=pd.read_csv(coordinate_path,float_precision='round_trip')
    frame=frame.loc[frame.group.eq(group)].reset_index(drop=True)
    colors=pd.read_csv(color_path,float_precision='round_trip')
    colors=colors.loc[colors.group.eq(group)].reset_index(drop=True)
    selected=pd.read_csv(selection_path)
    assert frame.electrode.tolist()==selected.electrode.tolist()==colors.electrode.tolist()
    assert len(frame)==n and frame.subject.nunique()==subjects and frame.electrode.is_unique
    with np.load(density_path,allow_pickle=False) as data:
        grid={name:data[name].copy() for name in ('x','y','valid_simplex')}
        density=data[group+'_density'].copy();relative=data[group+'_relative_density'].copy()
    return dict(group=group,region=region,n=n,subjects=subjects,title=title,frame=frame,colors=colors,
        density=density,relative=relative,grid=grid,selection_path=selection_path,
        inputs={str(path):sha(path) for path in (coordinate_path,color_path,density_path,selection_path)})


def _new_lexical_no_difference(reference):
    """Frozen loadings for the 43 Lexical Delay sites, on the existing density scale."""
    region,n,subjects='no_detected_difference',43,25
    n,subjects=group_counts('lexical',region,n,subjects)
    selection_path=ROOT/'results/fig3/regions/lexical'/f'{region}_electrodes.csv'
    assignments=NMF_DIR/'channel_assignments.csv'
    selected=pd.read_csv(selection_path,float_precision='round_trip')
    w=pd.read_csv(assignments,float_precision='round_trip')
    columns=[f'loading_{c}' for c in COMPONENTS]
    with np.load(NMF_DIR/'factors.npz',allow_pickle=False) as factors:
        np.testing.assert_allclose(np.linalg.norm(factors['H'],axis=1),1,rtol=0,atol=1e-12)
        assert w.channel.tolist()==factors['channels'].tolist()
    assert len(selected)==n and selected.subject.nunique()==subjects and selected.electrode.is_unique
    frame=selected[['electrode','subject','channel']].merge(w[['subject','channel',*columns]],
        on=['subject','channel'],how='left',validate='one_to_one')
    loadings=frame[columns].to_numpy(float)
    assert np.isfinite(loadings).all() and (loadings>=0).all() and (loadings.sum(axis=1)>0).all()
    proportions=loadings/loadings.sum(axis=1,keepdims=True)
    height=np.sqrt(3)/2;vertices=np.array([[.5,height],[0.,0.],[1.,0.]])
    xy=proportions@vertices
    recovered_sensory=xy[:,1]/height;recovered_motor=xy[:,0]-.5*recovered_sensory
    recovered=np.column_stack([recovered_sensory,1-recovered_sensory-recovered_motor,recovered_motor])
    np.testing.assert_allclose(recovered,proportions,rtol=0,atol=1e-12)
    density=composition_density(xy,reference['x'],reference['y'])
    peak=reference['common_density_peak']
    point_rgb=proportions@np.array([to_rgb(color) for color in COLORS])
    coordinates=frame[['electrode','subject','channel']].copy()
    coordinates.insert(0,'group','lexical')
    for index,name in enumerate(COMPONENTS):coordinates[name]=proportions[:,index]
    coordinates['ternary_x']=xy[:,0];coordinates['ternary_y']=xy[:,1]
    colors=pd.DataFrame(dict(group='lexical',electrode=frame.electrode,sensory=proportions[:,0],
        sustain=proportions[:,1],motor=proportions[:,2],red=point_rgb[:,0],green=point_rgb[:,1],
        blue=point_rgb[:,2],color_hex=[to_hex(row) for row in point_rgb]))
    return dict(group='lexical',region=region,n=n,subjects=subjects,
        title='Lexical Delay\nNo detected difference',frame=coordinates,colors=colors,
        density=density,relative=density/peak,grid=reference,selection_path=selection_path,
        loadings=loadings,inputs={str(selection_path):sha(selection_path),str(assignments):sha(assignments)})


def _draw_weight_pair(panels,svg,reference_peak):
    """Two ternary panels with the approved component colors, guides and gray density."""
    style()
    fig,axes=plt.subplots(1,2,squeeze=False,figsize=(8.8*2/2.54,10.4/2.54))
    axes=axes[0]
    fig.subplots_adjust(left=.035,right=.965,bottom=.27,top=.77,wspace=.23)
    height=np.sqrt(3)/2;vertices=np.array([[.5,height],[0.,0.],[1.,0.]])
    center=vertices.mean(axis=0);outline=np.vstack([vertices,vertices[0]])
    cmap=LinearSegmentedColormap.from_list('composition_cloud',
        ['#FAFBFC','#F0F1F3','#DEE1E6','#C5CBD4','#A2ACBB'])
    for axis,panel in zip(axes,panels):
        key=panel['group']+'_'+panel['region']
        axis.fill(outline[:,0],outline[:,1],facecolor='#FAFBFD',edgecolor='none',zorder=-1)
        axis.plot(outline[:,0],outline[:,1],color='#CBD2DD',lw=.7,zorder=2)
        for component,vertex in zip(COMPONENTS,vertices):
            spoke,=axis.plot([center[0],vertex[0]],[center[1],vertex[1]],color='#A2A9B2',lw=.7,zorder=2)
            spoke.set_gid(key+'_center_to_'+component+'_vertex')
        axis.text(.5*.5-.025,height*.5,'50%',ha='right',va='center',fontsize=5.5,color='#9299A2')
        axis.text(.5,-.025,'50%',ha='center',va='top',fontsize=5.5,color='#9299A2')
        axis.text(.5+.5*.5+.025,height*.5,'50%',ha='left',va='center',fontsize=5.5,color='#9299A2')
        point_rgb=panel['colors'][['red','green','blue']].to_numpy(float)
        points=axis.scatter(panel['frame'].ternary_x,panel['frame'].ternary_y,s=6.25,facecolor=point_rgb,
            alpha=.95,edgecolor='white',linewidth=.3,zorder=3,clip_on=False)
        points.set_gid(key+'_original_electrodes')
        relative=panel['relative'];valid=panel['grid']['valid_simplex']
        masked=np.ma.masked_where(~valid,relative)
        fill=axis.contourf(panel['grid']['x'],panel['grid']['y'],masked,levels=np.linspace(.06,1,17),
            cmap=cmap,vmin=0,vmax=1,zorder=.5,antialiased=True,
            extend='max' if relative[valid].max()>1 else 'neither')
        fill.set_gid(key+'_composition_density_cloud')
        contours=axis.contour(panel['grid']['x'],panel['grid']['y'],masked,levels=[.2,.4,.6,.8],
            colors='#858F9E',linewidths=.55,alpha=.40,zorder=1)
        contours.set_gid(key+'_descriptive_density_contours')
        axis.text(.5,height+.055,'Sensory',ha='center',va='bottom',fontsize=8,color=COLORS[0])
        axis.text(-.005,-.10,'Sustained',ha='center',va='top',fontsize=8,color=COLORS[1])
        axis.text(1.005,-.10,'Motor',ha='center',va='top',fontsize=8,color=COLORS[2])
        axis.set_title(f"{panel['title']}\n{panel['n']} electrodes / {panel['subjects']} participants",fontsize=9,pad=16)
        axis.set(xlim=(-.15,1.15),ylim=(-.20,height+.20),aspect='equal');axis.axis('off')
    color_axis=fig.add_axes([.36,.175,.28,.012])
    colorbar=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap=cmap),cax=color_axis,orientation='horizontal')
    colorbar.set_ticks([0,.5,1]);colorbar.ax.tick_params(labelsize=6,length=2,pad=2)
    colorbar.set_label('Relative electrode density',fontsize=7,labelpad=2);colorbar.outline.set_visible(False)
    handles=[Line2D([],[],marker='o',linestyle='none',markersize=2.25,color=color,label=label)
        for color,label in zip(COLORS,LABELS)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.065),ncol=3,frameon=False,
        fontsize=7,columnspacing=1.1,handletextpad=.4)
    fig.text(.5,.025,'Point colors blend the three component weights; gray clouds show relative electrode density.',
        ha='center',fontsize=6.7)
    fig.savefig(svg,format='svg',bbox_inches=None);plt.close(fig)
    assert reference_peak>0


def render_task_pairs():
    """Picture 21|18 and Lexical 43|27 ternaries on the existing shared density scale."""
    import json
    assert os.environ.get('SLURM_JOB_ID'),'Run through Slurm'
    picture_svg=IMAGES/'fig3_picture_weight_pair.svg'
    lexical_svg=IMAGES/'fig3_lexical_weight_pair.svg'
    results=RESULTS/'task_pairs'
    results.mkdir(parents=True,exist_ok=True);IMAGES.mkdir(parents=True,exist_ok=True)
    protected_paths=[p for folder in ('results/nmf','results/fig3','img/fig3')
        for p in (ROOT/folder).rglob('*') if p.is_file() and results not in p.parents and p not in (picture_svg,lexical_svg)]
    protected_paths += [ROOT/f'vizpub/fig{i}.ipynb' for i in range(1,7)]
    protected_paths += [ROOT/'vizpub/supplementary/fig3_supplementary.ipynb']
    protected={str(p):sha(p) for p in protected_paths}
    reference=json.loads((RESULTS/'manifest.json').read_text())
    peak=reference['common_density_peak']
    picture_null=_cached_weight_panel(RESULTS,'picture','no_detected_difference',*group_counts('picture','no_detected_difference',21,14),'Sound–Image\nNo detected difference')
    picture_sound=_cached_weight_panel(RESULTS/'sound_gt_image','picture','first_only',*group_counts('picture','first_only',18,12),'Sound–Image\nSound > Image only')
    lexical_decision=_cached_weight_panel(RESULTS,'lexical','first_only',*group_counts('lexical','first_only',27,16),'Lexical Delay\nDecision > Repeat')
    assignments=NMF_DIR/'channel_assignments.csv'
    w=pd.read_csv(assignments,float_precision='round_trip')
    columns=[f'loading_{c}' for c in COMPONENTS]
    height=np.sqrt(3)/2;vertices=np.array([[.5,height],[0.,0.],[1.,0.]])
    for panel in (picture_null,picture_sound,lexical_decision):
        np.testing.assert_allclose(panel['relative'],panel['density']/peak,rtol=0,atol=1e-12)
        task='picture' if panel['group']=='picture' else 'lexical'
        selected=pd.read_csv(ROOT/'results/fig3/regions'/task/f"{panel['region']}_electrodes.csv")
        merged=selected[['electrode','subject','channel']].merge(w[['subject','channel',*columns]],
            on=['subject','channel'],how='left',validate='one_to_one')
        assert merged.electrode.tolist()==panel['frame'].electrode.tolist()
        loadings=merged[columns].to_numpy(float)
        proportions=loadings/loadings.sum(axis=1,keepdims=True)
        xy=proportions@vertices
        np.testing.assert_allclose(xy,panel['frame'][['ternary_x','ternary_y']].to_numpy(float),rtol=0,atol=1e-12)
        np.testing.assert_allclose(composition_density(xy,panel['grid']['x'],panel['grid']['y']),
            panel['density'],rtol=0,atol=1e-12)
    for name in ('x','y','valid_simplex'):
        np.testing.assert_array_equal(picture_null['grid'][name],picture_sound['grid'][name])
        np.testing.assert_array_equal(picture_null['grid'][name],lexical_decision['grid'][name])
    reference_grid=dict(picture_null['grid'],common_density_peak=peak)
    lexical_null=_new_lexical_no_difference(reference_grid)
    assert set(picture_null['frame'].electrode).isdisjoint(picture_sound['frame'].electrode)
    assert set(lexical_null['frame'].electrode).isdisjoint(lexical_decision['frame'].electrode)
    pairs=(('picture',(picture_null,picture_sound),picture_svg),
        ('lexical',(lexical_null,lexical_decision),lexical_svg))
    written={}
    for name,panels,svg in pairs:
        _draw_weight_pair(panels,svg,peak)
        out=results/name;out.mkdir(parents=True,exist_ok=True)
        frames=[];colors=[];records=[]
        cache=dict(x=panels[0]['grid']['x'],y=panels[0]['grid']['y'],valid_simplex=panels[0]['grid']['valid_simplex'])
        for panel in panels:
            frames.append(panel['frame']);colors.append(panel['colors'])
            cache[panel['region']+'_density']=panel['density']
            cache[panel['region']+'_relative_density']=panel['relative']
            if 'loadings' in panel:
                for i,row in enumerate(panel['frame'].itertuples()):
                    for index,component in enumerate(COMPONENTS):
                        records.append(dict(group=name,electrode=row.electrode,subject=row.subject,channel=row.channel,
                            component=component,original_loading=float(panel['loadings'][i,index]),
                            loading_sum=float(panel['loadings'][i].sum()),proportion=float(getattr(row,component))))
        pd.concat(frames,ignore_index=True).to_csv(out/'ternary_coordinates.csv',index=False)
        pd.concat(colors,ignore_index=True).to_csv(out/'electrode_colors.csv',index=False)
        if records:pd.DataFrame(records).to_csv(out/'new_group_weights_long.csv',index=False)
        np.savez_compressed(out/'composition_density_grid.npz',**cache)
        inputs={}
        for panel in panels:inputs.update(panel['inputs'])
        write(out/'manifest.json',dict(passed=True,comparison=name,svg=str(svg),svg_sha256=sha(svg),
            n_electrodes={panel['region']:panel['n'] for panel in panels},
            n_participants={panel['region']:panel['subjects'] for panel in panels},
            reused_cached_coordinates_and_density=[panel['region'] for panel in panels if 'loadings' not in panel],
            newly_computed=[panel['region'] for panel in panels if 'loadings' in panel],
            common_density_peak=peak,density_reference_manifest=str(RESULTS/'manifest.json'),
            density_sigma_in_triangle_side_units=DENSITY_SIGMA,
            max_relative_density={panel['region']:float(panel['relative'][panel['grid']['valid_simplex']].max()) for panel in panels},
            source_hashes=inputs,no_NMF_refit=True,no_new_inferential_tests=True,
            reverse_direction_groups_not_included=True))
        written[name]=svg
    assert all(sha(p)==digest for p,digest in protected.items())
    return written


def _density_palette(color):
    rgb=np.array(to_rgb(color));white=np.ones(3)
    return [to_hex(white*(1-amount)+rgb*amount) for amount in (0.04,0.18,0.40,0.68,1)]


OVERLAY_SIGMA=.06
OVERLAY_FILL_FLOOR=.25
OVERLAY_CLOUD_ALPHA=.35
OVERLAY_CONTOUR_ALPHA=.45
OVERLAY_POINT_SIZE=9


def render_task_pair_overlays():
    """One triangle per task. Narrower kernels and a higher floor keep the two cores apart."""
    import json
    assert os.environ.get('SLURM_JOB_ID'),'Run through Slurm'
    specs=(dict(name='picture',task='picture',svg=IMAGES/'fig3_picture_weight_density_overlay.svg',title='Sound–Image',
            groups=(('no_detected_difference',21,'No detected difference · 21 electrodes / 14 participants','#858F9E'),
                ('first_only',18,'Sound > Image only · 18 electrodes / 12 participants','#2369BD'))),
        dict(name='lexical',task='lexical',svg=IMAGES/'fig3_lexical_weight_density_overlay.svg',title='Lexical Delay',
            groups=(('no_detected_difference',43,'No detected difference · 43 electrodes / 25 participants','#858F9E'),
                ('first_only',27,'Decision > Repeat · 27 electrodes / 16 participants','#6A5ACD'))))
    if GROUP_COUNTS is not None:
        for spec in specs:
            updated=[]
            for region,_,label,color in spec['groups']:
                n,subjects=GROUP_COUNTS[(spec['task'],region)]
                updated.append((region,n,f'{label.split(" · ")[0]} · {n} electrodes / {subjects} participants',color))
            spec['groups']=tuple(updated)
    results=RESULTS/'task_pairs'
    written={}
    for spec in specs:
        svg=spec['svg']
        source=results/spec['name']
        overlay_grid=source/'overlay_density_grid.npz'
        overlay_manifest=source/'density_overlay_manifest.json'
        excluded={svg,overlay_grid,overlay_manifest}
        protected_paths=[p for folder in ('results/nmf','results/fig3','img/fig3')
            for p in (ROOT/folder).rglob('*') if p.is_file() and p not in excluded]
        protected_paths += [ROOT/f'vizpub/fig{i}.ipynb' for i in range(1,7)]
        protected_paths += [ROOT/'vizpub/supplementary/fig3_supplementary.ipynb']
        protected={str(p):sha(p) for p in protected_paths}
        coordinate_path=source/'ternary_coordinates.csv'
        frame=pd.read_csv(coordinate_path,float_precision='round_trip')
        with np.load(source/'composition_density_grid.npz',allow_pickle=False) as data:
            grid={name:data[name].copy() for name in ('x','y','valid_simplex')}
        clouds=[];cache=dict(grid)
        for region,n,label,color in spec['groups']:
            selected=pd.read_csv(ROOT/'results/fig3/regions'/spec['task']/f'{region}_electrodes.csv')
            part=frame.set_index('electrode').loc[selected.electrode.tolist()].reset_index()
            assert len(part)==n and part.subject.nunique()>0
            xy=part[['ternary_x','ternary_y']].to_numpy(float)
            density=composition_density(xy,grid['x'],grid['y'],OVERLAY_SIGMA)
            clouds.append((region,label,color,density,xy));cache[region+'_density']=density
        peak=max(float(density[grid['valid_simplex']].max()) for _,_,_,density,_ in clouds)
        drawn=[]
        for region,label,color,density,xy in clouds:
            relative=density/peak
            cache[region+'_relative_density']=relative
            drawn.append((region,label,color,relative,xy))
        style()
        fig,axis=plt.subplots(figsize=(8.8/2.54,10.4/2.54))
        fig.subplots_adjust(left=.035,right=.965,bottom=.30,top=.77)
        height=np.sqrt(3)/2;vertices=np.array([[.5,height],[0.,0.],[1.,0.]])
        center=vertices.mean(axis=0);outline=np.vstack([vertices,vertices[0]])
        axis.fill(outline[:,0],outline[:,1],facecolor='#FAFBFD',edgecolor='none',zorder=-1)
        for region,label,color,relative,xy in drawn:
            cmap=LinearSegmentedColormap.from_list(region+'_density',_density_palette(color))
            masked=np.ma.masked_where(~grid['valid_simplex']|(relative<OVERLAY_FILL_FLOOR),relative)
            fill=axis.contourf(grid['x'],grid['y'],masked,levels=np.linspace(OVERLAY_FILL_FLOOR,1,13),cmap=cmap,
                vmin=0,vmax=1,alpha=OVERLAY_CLOUD_ALPHA,zorder=.5,antialiased=True)
            fill.set_gid(region+'_density_cloud')
            contours=axis.contour(grid['x'],grid['y'],np.ma.masked_where(~grid['valid_simplex'],relative),
                levels=[.4,.6,.8],colors=color,linewidths=.8,alpha=OVERLAY_CONTOUR_ALPHA,zorder=1)
            contours.set_gid(region+'_density_contours')
            points=axis.scatter(xy[:,0],xy[:,1],s=OVERLAY_POINT_SIZE,facecolor=color,alpha=.9,
                edgecolor='white',linewidth=.3,zorder=3,clip_on=False)
            points.set_gid(region+'_electrodes')
        axis.plot(outline[:,0],outline[:,1],color='#CBD2DD',lw=.7,zorder=2)
        for component,vertex in zip(COMPONENTS,vertices):
            spoke,=axis.plot([center[0],vertex[0]],[center[1],vertex[1]],color='#A2A9B2',lw=.7,zorder=2)
            spoke.set_gid('overlay_center_to_'+component+'_vertex')
        axis.text(.225,height*.5,'50%',ha='right',va='center',fontsize=5.5,color='#9299A2')
        axis.text(.5,-.025,'50%',ha='center',va='top',fontsize=5.5,color='#9299A2')
        axis.text(.775,height*.5,'50%',ha='left',va='center',fontsize=5.5,color='#9299A2')
        axis.text(.5,height+.055,'Sensory',ha='center',va='bottom',fontsize=8,color=COLORS[0])
        axis.text(-.005,-.10,'Sustained',ha='center',va='top',fontsize=8,color=COLORS[1])
        axis.text(1.005,-.10,'Motor',ha='center',va='top',fontsize=8,color=COLORS[2])
        axis.set_title(f"{spec['title']}\nOverlaid weight density",fontsize=9,pad=16)
        axis.set(xlim=(-.15,1.15),ylim=(-.20,height+.20),aspect='equal');axis.axis('off')
        for left,(region,label,color,_,_) in zip((.14,.56),drawn):
            cax=fig.add_axes([left,.20,.30,.012])
            bar=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap=LinearSegmentedColormap.from_list(region,_density_palette(color))),
                cax=cax,orientation='horizontal',alpha=OVERLAY_CLOUD_ALPHA)
            bar.set_ticks([0,.5,1]);bar.ax.tick_params(labelsize=6,length=2,pad=2);bar.outline.set_visible(False)
        fig.text(.5,.145,f'Relative density · sigma {OVERLAY_SIGMA:g} · shared peak',ha='center',fontsize=7)
        handles=[Line2D([],[],color=color,lw=2,label=label) for _,label,color,_,_ in drawn]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.04),ncol=1,frameon=False,
            fontsize=6.3,handletextpad=.5,labelspacing=.45)
        fig.savefig(svg,format='svg',bbox_inches=None);plt.close(fig)
        np.savez_compressed(overlay_grid,**cache)
        write(overlay_manifest,dict(passed=True,svg=str(svg),svg_sha256=sha(svg),electrode_points_omitted=False,
            point_size=OVERLAY_POINT_SIZE,point_color_matches_group=True,point_alpha=.9,
            density_sigma_in_triangle_side_units=OVERLAY_SIGMA,paired_plot_sigma_unchanged=DENSITY_SIGMA,
            fill_floor_of_shared_peak=OVERLAY_FILL_FLOOR,contour_levels=[.4,.6,.8],
            cloud_alpha=OVERLAY_CLOUD_ALPHA,contour_alpha=OVERLAY_CONTOUR_ALPHA,common_density_peak=peak,
            group_colors={region:color for region,_,color,_,_ in drawn},
            coordinates_unchanged=True,coordinate_source=str(coordinate_path),
            coordinate_source_sha256=sha(coordinate_path),
            paired_plot_density_unchanged=str(source/'composition_density_grid.npz'),
            no_NMF_refit=True,no_new_inferential_tests=True))
        assert all(sha(p)==digest for p,digest in protected.items())
        written[spec['name']]=svg
    return written


def render_picture_overlay():
    """Overlay accepted PN groups using their unchanged cached densities."""
    import json

    assert os.environ.get('SLURM_JOB_ID'),'Run through Slurm'
    results=RESULTS/'picture_overlay'
    svg=IMAGES/'fig3_sound_image_weight_overlay.svg'
    protected_paths=[p for folder in ('results/nmf','results/fig3','img/fig3')
        for p in (ROOT/folder).rglob('*') if p.is_file()]
    protected_paths += [ROOT/f'vizpub/fig{i}.ipynb' for i in range(1,7)]
    protected_paths += [ROOT/'vizpub/supplementary/fig3_supplementary.ipynb']
    protected_paths += [p for p in RESULTS.rglob('*') if p.is_file() and results not in p.parents]
    protected_paths += [p for p in IMAGES.glob('*.svg') if p!=svg]
    protected={str(p):sha(p) for p in protected_paths}
    results.mkdir(parents=True,exist_ok=True);IMAGES.mkdir(parents=True,exist_ok=True)
    definitions=(('no_detected_difference','no_detected_difference',RESULTS,21,14,'#858F9E',.40),
        ('sound_gt_image','first_only',RESULTS/'sound_gt_image',18,12,'#2369BD',.40))
    entries=[];inputs={};cached={};reference_peak=None
    for key,region,source,n,subjects,color,cloud_alpha in definitions:
        manifest_path=source/'manifest.json'
        manifest=json.loads(manifest_path.read_text())
        peak=manifest['common_density_peak']
        if reference_peak is None:reference_peak=peak
        assert peak==reference_peak and manifest['density_sigma_in_triangle_side_units']==DENSITY_SIGMA
        coordinate_path=source/'ternary_coordinates.csv'
        frame=pd.read_csv(coordinate_path,float_precision='round_trip')
        frame=frame.loc[frame.group.eq('picture')].copy()
        selection_path=ROOT/'results/fig3/regions/picture'/f'{region}_electrodes.csv'
        selected=pd.read_csv(selection_path)
        assert frame.electrode.tolist()==selected.electrode.tolist()
        assert len(frame)==n and frame.subject.nunique()==subjects and frame.electrode.is_unique
        density_path=source/'composition_density_grid.npz'
        with np.load(density_path,allow_pickle=False) as data:
            grid={name:data[name].copy() for name in ('x','y','valid_simplex')}
            relative=data['picture_relative_density'].copy();raw=data['picture_density'].copy()
        if cached:
            for name,array in grid.items():np.testing.assert_array_equal(array,cached[name])
        else:cached.update(grid)
        cached[key+'_density']=raw;cached[key+'_relative_density']=relative
        entries.append((key,frame,color,cloud_alpha))
        for path in (manifest_path,coordinate_path,density_path,selection_path):inputs[str(path)]=sha(path)
    assert set(entries[0][1].electrode).isdisjoint(entries[1][1].electrode)
    style()
    fig,axis=plt.subplots(figsize=(8.8/2.54,10.4/2.54))
    fig.subplots_adjust(left=.035,right=.965,bottom=.27,top=.77)
    height=np.sqrt(3)/2;vertices=np.array([[.5,height],[0.,0.],[1.,0.]])
    center=vertices.mean(axis=0);outline=np.vstack([vertices,vertices[0]])
    axis.fill(outline[:,0],outline[:,1],facecolor='#FAFBFD',edgecolor='none',zorder=-1)
    palettes=(['#FAFBFC','#F0F1F3','#DEE1E6','#C5CBD4','#A2ACBB'],
        ['#F8FBFF','#E8F0FB','#BED4F1','#7FA9DD','#2369BD'])
    cmaps=[]
    for (key,frame,color,cloud_alpha),palette in zip(entries,palettes):
        cmap=LinearSegmentedColormap.from_list(key+'_density',palette);cmaps.append(cmap)
        relative=cached[key+'_relative_density']
        masked=np.ma.masked_where(~cached['valid_simplex'],relative)
        fill=axis.contourf(cached['x'],cached['y'],masked,levels=np.linspace(.06,1,17),
            cmap=cmap,vmin=0,vmax=1,alpha=cloud_alpha,zorder=.5,antialiased=True,
            extend='max' if relative[cached['valid_simplex']].max()>1 else 'neither')
        fill.set_gid(key+'_density_cloud')
        contours=axis.contour(cached['x'],cached['y'],masked,levels=[.2,.4,.6,.8],
            colors=color,linewidths=.65,alpha=.65,zorder=1)
        contours.set_gid(key+'_density_contours')
    axis.plot(outline[:,0],outline[:,1],color='#CBD2DD',lw=.7,zorder=2)
    for component,vertex in zip(COMPONENTS,vertices):
        spoke,=axis.plot([center[0],vertex[0]],[center[1],vertex[1]],color='#A2A9B2',lw=.7,zorder=2)
        spoke.set_gid('overlay_center_to_'+component+'_vertex')
    axis.text(.225,height*.5,'50%',ha='right',va='center',fontsize=5.5,color='#9299A2')
    axis.text(.5,-.025,'50%',ha='center',va='top',fontsize=5.5,color='#9299A2')
    axis.text(.775,height*.5,'50%',ha='left',va='center',fontsize=5.5,color='#9299A2')
    for key,frame,color,_ in entries:
        points=axis.scatter(frame.ternary_x,frame.ternary_y,s=14,facecolor=color,
            alpha=.60,edgecolor='white',linewidth=.35,zorder=3,clip_on=False)
        points.set_gid(key+'_original_electrodes')
    axis.text(.5,height+.055,'Sensory',ha='center',va='bottom',fontsize=8,color=COLORS[0])
    axis.text(-.005,-.10,'Sustained',ha='center',va='top',fontsize=8,color=COLORS[1])
    axis.text(1.005,-.10,'Motor',ha='center',va='top',fontsize=8,color=COLORS[2])
    axis.set_title('Sound–Image · Stimulus 0–1 s\nFunctional-weight distributions',fontsize=9,pad=16)
    axis.set(xlim=(-.15,1.15),ylim=(-.20,height+.20),aspect='equal');axis.axis('off')
    for left,cmap in zip((.14,.56),cmaps):
        cax=fig.add_axes([left,.215,.30,.012])
        bar=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap=cmap),cax=cax,orientation='horizontal',alpha=.40)
        bar.set_ticks([0,.5,1]);bar.ax.tick_params(labelsize=6,length=2,pad=2);bar.outline.set_visible(False)
    fig.text(.5,.16,'Relative electrode density · shared scale',ha='center',fontsize=7)
    handles=[Line2D([],[],marker='o',linestyle='none',markersize=4,color=entries[i][2],label=label)
        for i,label in enumerate(('No detected difference · 21 electrodes / 14 participants',
            'Sound > Image only · 18 electrodes / 12 participants'))]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.055),ncol=1,
        frameon=False,fontsize=6.3,handletextpad=.5,labelspacing=.65)
    fig.text(.5,.015,'Point color identifies the condition group; position shows component weights.',
        ha='center',fontsize=5.3)
    fig.savefig(svg,format='svg',bbox_inches=None);plt.close(fig)
    combined=pd.concat([frame.assign(comparison_group=key) for key,frame,_,_ in entries],ignore_index=True)
    combined.to_csv(results/'electrode_coordinates.csv',index=False)
    np.savez_compressed(results/'composition_density_grid.npz',**cached)
    assert all(sha(p)==digest for p,digest in protected.items())
    write(results/'manifest.json',dict(passed=True,job_id=os.environ['SLURM_JOB_ID'],svg=str(svg),
        svg_sha256=sha(svg),source_hashes=inputs,n_electrodes=dict(no_detected_difference=21,sound_gt_image=18),
        n_participants=dict(no_detected_difference=14,sound_gt_image=12),groups_disjoint=True,
        bidirectional_electrode_excluded=True,point_size=14,point_alpha=.60,
        group_colors={key:color for key,_,color,_ in entries},
        cloud_alpha={key:alpha for key,_,_,alpha in entries},contour_alpha=.65,
        point_color_encodes_condition_group=True,point_coordinates_identical_to_prior_exports=True,
        density_arrays_reused_without_recomputation=True,density_sigma=DENSITY_SIGMA,
        common_density_peak=reference_peak,relative_density_contour_levels=[.2,.4,.6,.8],
        interior_guides='Triangle centroid to three vertices',mean_marker=None,vertex_percentages=False,
        no_NMF_refit=True,no_new_inferential_tests=True,protected_files_unchanged=len(protected)))
    return svg
