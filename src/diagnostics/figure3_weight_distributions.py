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


def composition_density(xy,xx,yy):
    """Equal-electrode Gaussian display density in the unchanged simplex plane."""
    density=np.zeros_like(xx)
    for x,y in xy:
        density+=np.exp(-((xx-x)**2+(yy-y)**2)/(2*DENSITY_SIGMA**2))
    return density/(len(xy)*2*np.pi*DENSITY_SIGMA**2)


def render(*,sound_gt_image=False):
    assert os.environ.get('SLURM_JOB_ID'),'Run through Slurm'
    results=RESULTS/'sound_gt_image' if sound_gt_image else RESULTS
    groups=(('picture','first_only',18,12,'Sound–Image\nSound > Image only'),) if sound_gt_image else GROUPS
    reference_peak=None
    if sound_gt_image:
        import json
        reference_peak=json.loads((RESULTS/'manifest.json').read_text())['common_density_peak']
    results.mkdir(parents=True,exist_ok=True);IMAGES.mkdir(parents=True,exist_ok=True)
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
