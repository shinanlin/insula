"""Author-approved continuous-loading draft for the two selected Figure 3 groups."""
from pathlib import Path
import os

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex
import numpy as np
import pandas as pd

from src.figure3 import ROOT, sha, style, write
from src.supplementary.decision_repeat_maps import _figure1_local_insula_renderer
from src.supplementary.selected_brain_contours import _figure2_contour_helpers

COMPONENTS=('sensory','sustain','motor')
LABELS=('Sensory','Sustained','Motor')
COLORS=('#2369BD','#C4A35A','#A9373B')
RESULTS=ROOT/'results/diagnostics/fig3_continuous_weights_2026-11-02'
IMAGES=ROOT/'img/diagnostics/fig3_continuous_weights_2026-11-02'
GROUPS=(('picture','no_detected_difference',21,14,'Sound–Image · No detected difference'),
        ('lexical','first_only',27,16,'Lexical Delay · Decision > Repeat'))


def load_groups():
    """Join frozen W and use each electrode's three-loading composition."""
    path=ROOT/'results/nmf/channel_assignments.csv'
    weights=pd.read_csv(path,float_precision='round_trip')
    columns=[f'loading_{component}' for component in COMPONENTS]
    with np.load(ROOT/'results/nmf/factors.npz',allow_pickle=False) as factors:
        assert weights.channel.tolist()==factors['channels'].tolist()
        np.testing.assert_allclose(np.linalg.norm(factors['H'],axis=1),1,rtol=0,atol=1e-12)
        for index,name in enumerate(factors['component_names'].tolist()):
            np.testing.assert_allclose(weights[f'loading_{name}'],factors['W'][:,index],rtol=0,atol=1e-12)
    frames={};means=[];sources={str(path):sha(path)}
    for key,region,n,subjects,title in GROUPS:
        source=ROOT/'results/fig3/regions'/key/f'{region}_electrodes.csv'
        frame=pd.read_csv(source,float_precision='round_trip')
        assert len(frame)==n and frame.subject.nunique()==subjects and frame.electrode.is_unique
        frame=frame.merge(weights[['subject','channel',*columns]],on=['subject','channel'],
            how='left',validate='one_to_one')
        values=frame[columns].to_numpy(float)
        assert np.isfinite(values).all() and (values>=0).all() and (values.sum(axis=1)>0).all()
        relative=values/values.sum(axis=1,keepdims=True)
        np.testing.assert_allclose(relative.sum(axis=1),1,rtol=0,atol=1e-12)
        for index,component in enumerate(COMPONENTS):frame[f'proportion_{component}']=relative[:,index]
        frame.to_csv(RESULTS/f'{key}_electrode_weights.csv',index=False)
        sources[str(source)]=sha(source);frames[key]=frame
        means.append(dict(group=title,n_sites=n,n_participants=subjects,
            **{component:float(relative[:,index].mean()) for index,component in enumerate(COMPONENTS)}))
    pd.DataFrame(means).to_csv(RESULTS/'group_mean_proportions.csv',index=False)
    return frames,sources


def render():
    """Render continuous point colors, without RGB mixing or spatial smoothing."""
    assert os.environ.get('SLURM_JOB_ID'),'Run via Slurm'
    RESULTS.mkdir(parents=True,exist_ok=True);IMAGES.mkdir(parents=True,exist_ok=True)
    protected=[]
    for folder in ('results/nmf','results/fig3','img/fig3'):
        protected.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
    protected.extend(ROOT/f'vizpub/fig{i}.ipynb' for i in range(1,7))
    protected.append(ROOT/'vizpub/supplementary/fig3_supplementary.ipynb')
    frozen={str(p):sha(p) for p in protected}
    style();frames,sources=load_groups()
    renderer,renderer_proof=_figure1_local_insula_renderer(ROOT)
    crop_helpers,crop_proof=_figure2_contour_helpers(ROOT)
    cmaps={component:LinearSegmentedColormap.from_list(component,('#FFFFFF',color))
        for component,color in zip(COMPONENTS,COLORS)}
    normalizer=Normalize(0,1)
    panels={};projection=[];cropping=[]
    for key,region,n,subjects,title in GROUPS:
        points=frames[key]
        reference=pd.read_csv(ROOT/'results/fig3/regions'/key/f'{region}_display_coordinates.csv')
        for hemisphere in ('lh','rh'):
            frame=points.loc[points.x.lt(0) if hemisphere=='lh' else points.x.gt(0)].copy()
            xyz=frame[['x','y','z']].to_numpy(float)
            original_colors=np.array([COLORS[COMPONENTS.index(label)] for label in frame.functional_cluster],dtype=object)
            brain=None
            try:
                image,project,brain=renderer['_render_insula_hemi'](hemisphere,xyz,original_colors,np.full(len(frame),24.))
                assert image.shape[:2]==(800,800)
                camera=renderer['_camera_xyz'](brain)
                displayed=renderer['_project_to_insula'](xyz,renderer[f'{hemisphere}_surf'],
                    renderer[f'{hemisphere}_pial_coords'],camera)
                pixels=project(displayed)
            finally:
                if brain is not None:brain.close()
            previous=reference.set_index('electrode').loc[frame.electrode]
            np.testing.assert_allclose(pixels,previous[['pixel_x','pixel_y']].to_numpy(float),rtol=0,atol=1e-8)
            cropped,cropped_pixels=crop_helpers['crop_to_content'](image,pixels)
            offset=pixels-cropped_pixels
            np.testing.assert_allclose(offset,np.broadcast_to(offset[0],offset.shape),rtol=0,atol=1e-8)
            x0,y0=np.rint(offset[0]).astype(int);height,width=cropped.shape[:2]
            cropping.append(dict(group=key,hemisphere=hemisphere,x0=int(x0),y0=int(y0),width=width,height=height))
            for index,row in enumerate(frame.itertuples()):
                projection.append(dict(group=key,electrode=row.electrode,subject=row.subject,hemisphere=hemisphere,
                    x=row.x,y=row.y,z=row.z,pixel_x=float(pixels[index,0]),pixel_y=float(pixels[index,1]),
                    cropped_x=float(cropped_pixels[index,0]),cropped_y=float(cropped_pixels[index,1]),
                    display_lift_mm=4,point_size_pixels=24))
            # Fix cropping using the accepted hard-label reference once, so
            # weak/white continuous values cannot change framing or positions.
            for component in COMPONENTS:
                colors=np.array([to_hex(cmaps[component](value)) for value in frame[f'proportion_{component}']],dtype=object)
                brain=None
                try:
                    image,_,brain=renderer['_render_insula_hemi'](hemisphere,xyz,colors,np.full(len(frame),24.))
                    np.testing.assert_allclose(renderer['_camera_xyz'](brain),camera,rtol=0,atol=1e-8)
                    panels[f'{key}_{component}_{hemisphere}']=image[y0:y0+height,x0:x0+width]
                finally:
                    if brain is not None:brain.close()
                print('RENDERED',key,hemisphere,component,len(frame),'sites',flush=True)
    pd.DataFrame(projection).to_csv(RESULTS/'display_coordinates.csv',index=False)
    pd.DataFrame(cropping).to_csv(RESULTS/'fixed_cropping.csv',index=False)
    np.savez_compressed(RESULTS/'rendered_panels.npz',**panels)
    figure=plt.figure(figsize=(26.2/2.54,12.8/2.54))
    outer=figure.add_gridspec(2,3,left=.025,right=.985,bottom=.14,top=.82,wspace=.10,hspace=.55)
    for row,(key,region,n,subjects,title) in enumerate(GROUPS):
        top=outer[row,0].get_position(figure).y1
        figure.text(.5,top+.105,f'{title} · {n} electrodes / {subjects} participants',
            ha='center',va='top',fontsize=10)
        for column,(component,label,color) in enumerate(zip(COMPONENTS,LABELS,COLORS)):
            position=outer[row,column].get_position(figure)
            figure.text((position.x0+position.x1)/2,top+.052,label,ha='center',va='bottom',fontsize=9,color=color)
            pair=outer[row,column].subgridspec(1,2,wspace=.03)
            for index,(hemisphere,name) in enumerate((('lh','Left insula'),('rh','Right insula'))):
                axis=figure.add_subplot(pair[0,index]);axis.imshow(panels[f'{key}_{component}_{hemisphere}'])
                axis.axis('off')
                count=int((frames[key].x.lt(0) if hemisphere=='lh' else frames[key].x.gt(0)).sum())
                axis.set_title(f'{name} · n={count}',fontsize=7,pad=2)
    for column,component in enumerate(COMPONENTS):
        position=outer[1,column].get_position(figure)
        color_axis=figure.add_axes([position.x0+.035,.065,position.width-.07,.018])
        colorbar=figure.colorbar(ScalarMappable(norm=normalizer,cmap=cmaps[component]),cax=color_axis,orientation='horizontal')
        colorbar.set_ticks((0,.5,1));colorbar.ax.tick_params(labelsize=7,length=2,pad=2)
    figure.text(.5,.015,'Relative NMF loading = component loading / sum of all three loadings',
        ha='center',va='bottom',fontsize=7.5)
    svg=IMAGES/'fig3_continuous_component_weights.svg'
    figure.savefig(svg,format='svg',bbox_inches=None);plt.close(figure)
    assert all(sha(path)==digest for path,digest in frozen.items())
    write(RESULTS/'manifest.json',dict(passed=True,job_id=os.environ['SLURM_JOB_ID'],
        svg=str(svg),svg_sha256=sha(svg),source_hashes=sources,groups=[dict(key=key,n_sites=n,n_participants=subjects)
            for key,region,n,subjects,title in GROUPS],
        weight_definition='Frozen sound-only NMF W, H rows unit L2 norm; W_c / sum_c W_c per electrode',
        common_color_limits=[0,1],component_colors=dict(zip(COMPONENTS,COLORS)),point_size_pixels=24,
        no_RGB_mixing=True,no_spatial_smoothing=True,no_hard_label_selection_or_recoloring=True,
        original_figure1_renderer=renderer_proof,cropping_function_source=crop_proof,
        cropping_fixed_across_components=True,all_display_coordinates_match_existing_maps=True,
        protected_existing_files_unchanged=len(frozen),no_new_tests=True,no_NMF_refit=True,
        main_figure_outputs_unchanged=True))
    return svg
