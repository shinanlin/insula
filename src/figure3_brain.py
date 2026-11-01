"""Reviewed local-insula functional maps, without obsolete analysis entry points."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd
from src.supplementary.decision_repeat_maps import _figure1_local_insula_renderer
from src.supplementary.selected_brain_contours import _figure2_contour_helpers
ROOT=Path(__file__).resolve().parents[1]

def brain_pair(points,groups,colors,labels,*,title):
    """Same Figure 1 surface/snap/lift and Figure 2 contours; group by condition."""
    renderer,proof=_figure1_local_insula_renderer(ROOT)
    contour,contour_proof=_figure2_contour_helpers(ROOT)
    assert points.electrode.is_unique and points['display_group'].isin(groups).all()
    cm=1/2.54
    fig,axes=plt.subplots(1,2,figsize=(8.6*cm,4.7*cm))
    fig.subplots_adjust(left=.03,right=.98,bottom=.26,top=.80,wspace=.03)
    fig.suptitle(title,fontsize=7.5,y=.98)
    audit=[];rings_audit=[]
    for ax,hemi,hemilabel in zip(axes,('lh','rh'),('Left insula','Right insula')):
        frame=points.loc[points.x.lt(0) if hemi=='lh' else points.x.gt(0)]
        xyz=frame[['x','y','z']].to_numpy(float)
        palette=np.asarray([colors[g] for g in frame.display_group],dtype=object)
        brain=None
        try:
            image,project,brain=renderer['_render_insula_hemi'](hemi,xyz,palette,np.full(len(frame),24.))
            if len(frame):
                camera=renderer['_camera_xyz'](brain)
                surface=renderer[f'{hemi}_surf'];pial=renderer[f'{hemi}_pial_coords']
                displayed=renderer['_project_to_insula'](xyz,surface,pial,camera)
                direction=camera-displayed;direction/=np.linalg.norm(direction,axis=1,keepdims=True)
                snapped=displayed-4.*direction
                residual,vertices=renderer['cKDTree'](pial[surface['verts']]).query(snapped)
                assert np.max(residual)<1e-8
                pixels=project(displayed)
                for i,r in enumerate(frame.itertuples()):
                    audit.append(dict(electrode=r.electrode,subject=r.subject,hemisphere=hemi,display_group=r.display_group,
                        pixel_x=float(pixels[i,0]),pixel_y=float(pixels[i,1]),surface_residual_mm=float(residual[i]),display_lift_mm=4.))
            else:pixels=np.empty((0,2))
        finally:
            if brain is not None:brain.close()
        image,pixels=contour['crop_to_content'](image,pixels)
        ax.imshow(image);ax.set_xlim(0,image.shape[1]);ax.set_ylim(image.shape[0],0);ax.axis('off')
        ax.set_title(f'{hemilabel} · n={len(frame)}',fontsize=7,pad=2)
        for group in groups:
            group_pixels=pixels[frame.display_group.eq(group).to_numpy()]
            rings=contour['kde_contour_rings'](group_pixels,image.shape[:2])
            contour['draw_rings'](ax,rings,colors[group])
            rings_audit.append(dict(hemisphere=hemi,group=group,n_sites=len(group_pixels),n_contours=len(rings)))
    fig.canvas.draw()
    for ax,shift in zip(axes,(.025,-.025)):
        p=ax.get_position();ax.set_position([p.x0+shift,p.y0,p.width,p.height])
    handles=[Line2D([0],[0],color=colors[g],lw=1,label=f'{labels[g]} (n={int(points.display_group.eq(g).sum())})') for g in groups]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.095),fontsize=6,frameon=False,ncol=len(groups),handletextpad=.3,columnspacing=.7)
    fig.text(.5,.018,'Contours show spatial density',ha='center',va='bottom',fontsize=5.9)
    assert set(r['electrode'] for r in audit)==set(points.electrode)
    return fig,audit,rings_audit
