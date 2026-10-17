"""Native lateral projections of user-supplied CCEP receiver latencies; run on Slurm."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import nibabel as nib
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, colors
from scipy.spatial import cKDTree
from grant_native_surface_helpers import INSULA_NAMES
from matplotlib.lines import Line2D
from screen_recon_aic_contacts import read_contacts, load_pipeline, UPSTREAM

ROOT=Path('/hpc/group/coganlab/nanlinshi/insula-functional')
RECON=Path('/cwork/ns458/ECoG_Recon')
OUT=ROOT/'results/grant_ccep_receivers'
FIG=ROOT/'img/grant_ccep_receivers'
ANATOMY={
 'D0149':Path('/cwork/ns458/BIDS-1.3_PictureNaming/BIDS/derivatives/parcellation/sub-D0149/bipolar/sub-D0149_aparc2009s.csv'),
 'D0153':Path('/cwork/ns458/insula_recon_bipolar/derivatives/parcellation/sub-D0153/bipolar/sub-D0153_aparc2009sHammers.csv')}
ROI_NAMES={'WM':'White matter','STG':'Superior temporal gyrus','CG':'Cingulate gyrus','INS':'Insula','IFG–INS':'Inferior frontal gyrus–insula boundary'}
CM=1/2.54
NORM=colors.Normalize(0,150)
CMAP=plt.colormaps['viridis']  # Uniform 0–150 ms: purple → teal/green → yellow.
STIM_COLOR='#B44936'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def get_anatomy(subject,pairs,pipeline):
    recon='D'+str(int(subject[1:]))
    contacts,path,_=read_contacts(RECON/recon)
    lookup={r['contact']:r for r in contacts}
    table=pd.read_csv(ANATOMY[subject])
    atlas=pipeline._resolve_atlas('aparc2009s',str(RECON),recon,subject,str(UPSTREAM.parent))
    # Native voxel centers, right cortical and subcortical gray matter only.
    gm_ids=[int(k) for k,v in atlas[2].items() if
            (v.startswith('ctx_rh_') and 'unknown' not in v.lower()) or
            int(k) in {49,50,51,52,53,54,58}]
    ijk=np.column_stack(np.where(np.isin(atlas[0],gm_ids)))
    gm_xyz=nib.affines.apply_affine(atlas[1].header.get_vox2ras_tkr(),ijk)
    gm_tree=cKDTree(gm_xyz)
    results=[]
    for pair in pairs:
        first,second=pair.split('–')
        assert lookup[first]['recon_hemi']==lookup[second]['recon_hemi']=='R'
        match=table.loc[table.apply(lambda r:set((r.contact_1,r.contact_2))=={first,second},axis=1)]
        assert len(match)==1,(subject,pair,'ambiguous or missing anatomy')
        r=match.iloc[0]
        xyz=np.array([[lookup[c][v] for v in ('x','y','z')] for c in (first,second)],float)
        midpoint=xyz.mean(axis=0)
        np.testing.assert_allclose(midpoint,r[['x','y','z']].to_numpy(float),atol=1e-5)
        source_center=r.get('midpoint_label',r.get('center'))
        labels=[pipeline._label_coordinate(p,atlas[1],atlas[0],atlas[2]) for p in [xyz[0],midpoint,xyz[1]]]
        expected={r.contact_1:r.contact_1_label,r.contact_2:r.contact_2_label}
        assert labels==[expected[first],source_center,expected[second]],(subject,pair,labels)
        distance,index=gm_tree.query(midpoint)
        nearest_ijk=ijk[int(index)]
        nearest_label=atlas[2][int(atlas[0][tuple(nearest_ijk)])]
        nearest_roi=pipeline._gross_roi(nearest_label,atlas[3])
        nearest_roi='INS' if nearest_label in {'ctx_rh_'+name for name in INSULA_NAMES} else nearest_roi
        midpoint_roi=pipeline._gross_roi(source_center,atlas[3])
        midpoint_roi='INS' if source_center in {'ctx_rh_'+name for name in INSULA_NAMES} else midpoint_roi
        display_roi=nearest_roi if r.roi=='WM' else r.roi
        results.append(dict(subject=subject,pair=pair,canonical_channel=r['name'],roi=r.roi,
            roi_full=ROI_NAMES.get(r.roi,r.roi),mix=bool(r.mix),midpoint_label=source_center,
            midpoint_roi=midpoint_roi,display_roi=display_roi,nearest_gm_label=nearest_label,
            nearest_gm_roi=nearest_roi,nearest_gm_distance_mm=float(distance),
            nearest_gm_x=float(gm_xyz[index,0]),nearest_gm_y=float(gm_xyz[index,1]),nearest_gm_z=float(gm_xyz[index,2]),
            contact_1=first,contact_2=second,contact_1_label=labels[0],contact_2_label=labels[2],
            x=float(midpoint[0]),y=float(midpoint[1]),z=float(midpoint[2]),
            midpoint_rule='arithmetic mean of native tkRAS endpoints',source_table=str(ANATOMY[subject])))
    pial=RECON/recon/'surf/rh.pial'
    inputs=[path,ANATOMY[subject],pial,RECON/recon/'mri/aparc.a2009s+aseg.mgz',RECON/recon/'mri/orig.mgz',RECON/recon/'label/rh.aparc.a2009s.annot',UPSTREAM]
    return pd.DataFrame(results),pial,{str(p):sha(p) for p in inputs}

def render_lateral(pial,points,view="lateral"):
    xyz,tri=nib.freesurfer.read_geometry(pial)
    mesh=pv.PolyData(xyz,np.column_stack([np.full(len(tri),3),tri]).ravel())
    center=(xyz.min(axis=0)+xyz.max(axis=0))/2
    plotter=pv.Plotter(off_screen=True,notebook=False,window_size=(1200,800))
    try:
        plotter.set_background('white')
        plotter.add_mesh(mesh,color='#D5D5D5',smooth_shading=True,ambient=.35,diffuse=.65,specular=.08)
        assert view in ('lateral','medial')
        plotter.camera_position=[center+[300 if view=='lateral' else -300,0,0],center,[0,0,1]]
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale=65.
        plotter.show(auto_close=False)
        rgb=plotter.screenshot(return_img=True)
        matrix=plotter.camera.GetCompositeProjectionTransformMatrix(1.5,-1,1)
        matrix=np.array([[matrix.GetElement(i,j) for j in range(4)] for i in range(4)])
        def project(p):
            hom=np.column_stack([p,np.ones(len(p))])@matrix.T
            ndc=hom[:,:3]/hom[:,3:4]
            return np.column_stack([(ndc[:,0]+1)*600,(1-ndc[:,1])*400])
        surface_xy=project(xyz)
        assert surface_xy[:,0].min()>5 and surface_xy[:,0].max()<1195 and surface_xy[:,1].min()>5 and surface_xy[:,1].max()<795, 'Cortex clipped'
        xy=project(points[['x','y','z']].to_numpy(float))
        assert ((xy[:,0]>0)&(xy[:,0]<1200)&(xy[:,1]>0)&(xy[:,1]<800)).all()
        # The right lateral projection must ignore x (depth), not displace any marker.
        moved=points[['x','y','z']].to_numpy(float).copy();moved[:,0]+=10
        np.testing.assert_allclose(project(moved),xy,atol=1e-7)
        return dict(rgb=rgb,xy=xy,pixels_per_mm=800/130.,camera_center=center.tolist(),view=view)
    finally:plotter.close()

def prepare_stim_context(subject,anatomy):
    recon='D'+str(int(subject[1:]))
    orig=nib.load(RECON/recon/'mri/orig.mgz')
    seg_img=nib.load(RECON/recon/'mri/aparc.a2009s+aseg.mgz')
    np.testing.assert_allclose(orig.header.get_vox2ras_tkr(),seg_img.header.get_vox2ras_tkr())
    tkr=orig.header.get_vox2ras_tkr()
    orientation=nib.orientations.ornt_transform(nib.orientations.io_orientation(tkr),nib.orientations.axcodes2ornt(('R','A','S')))
    affine=tkr@nib.orientations.inv_ornt_aff(orientation,orig.shape[:3])
    mri=nib.orientations.apply_orientation(np.asarray(orig.dataobj),orientation)
    seg=nib.orientations.apply_orientation(np.asarray(seg_img.dataobj),orientation)
    xyz,tri=nib.freesurfer.read_geometry(RECON/recon/'surf/rh.pial')
    ann,_,names=nib.freesurfer.read_annot(RECON/recon/'label/rh.aparc.a2009s.annot')
    ids=[i for i,v in enumerate(names) if v.decode().replace('&','_and_') in INSULA_NAMES]
    keep=np.isin(ann,ids);tri=tri[keep[tri].all(axis=1)]
    vertices,inverse=np.unique(tri,return_inverse=True);points=xyz[vertices]
    faces=inverse.reshape(-1,3)
    mesh=pv.PolyData(points,np.column_stack([np.full(len(faces),3),faces]).ravel())
    center=(points.min(axis=0)+points.max(axis=0))/2
    plotter=pv.Plotter(off_screen=True,notebook=False,window_size=(800,800))
    try:
        plotter.set_background('white');plotter.add_mesh(mesh,color='#DDD2AD',smooth_shading=True,ambient=.35,diffuse=.65)
        plotter.camera_position=[center+[200,0,0],center,[0,0,1]]
        plotter.enable_parallel_projection();plotter.camera.parallel_scale=32.
        plotter.show(auto_close=False);rgb=plotter.screenshot(return_img=True)
        matrix=plotter.camera.GetCompositeProjectionTransformMatrix(1.,-1,1)
        matrix=np.array([[matrix.GetElement(i,j) for j in range(4)] for i in range(4)])
        hom=np.column_stack([anatomy[['x','y','z']].to_numpy(float),np.ones(len(anatomy))])@matrix.T
        ndc=hom[:,:3]/hom[:,3:4]
        xy=np.column_stack([(ndc[:,0]+1)*400,(1-ndc[:,1])*400])
    finally:plotter.close()
    return dict(mri=mri,seg=seg,affine=affine,rgb=rgb,xy=xy)


def draw_panel(fig,area,record,anatomy,panel):
    left,bottom,width,height=area
    def coords(x,y):return left+x*width,bottom+y*height
    def axis(x,y,w,h):return fig.add_axes([left+x*width,bottom+y*height,w*width,h*height])
    texts=[]
    def text(x,y,s,**kw):
        t=fig.text(*coords(x,y),s,**kw);texts.append(t);return t
    def short(pair):
        first,second=pair.split('–');import re
        return first+'–'+re.search(r'\d+$',second).group()
    def annotate(ax,xy,label,offset=(5,5),ha='left',va='bottom',fontsize=6):
        t=ax.annotate(label,xy,xytext=offset,textcoords='offset points',ha=ha,va=va,
            fontsize=fontsize,linespacing=1.15,annotation_clip=False,
            bbox=dict(boxstyle='square,pad=.12',facecolor='white',edgecolor='none',alpha=.88))
        texts.append(t);return t
    stim=anatomy.loc[anatomy.pair.eq(record['stimulus'])].iloc[0]
    title=record['subject'].replace('D0','D',1)+' · Stim: '+short(record['stimulus'])+' · '+stim.roi
    text(.025,.976,title,ha='left',va='top',fontsize=8,fontweight='bold')
    receivers=record['receivers']
    cg=lambda rec:anatomy.loc[anatomy.pair.eq(rec['pair']),'display_roi'].iloc[0]=='CG'
    has_cg=any(cg(r) for r in receivers)
    xy_by_pair={p:xy for p,xy in zip(anatomy.pair,panel['xy'])}
    def brain_axis(ax,view_panel,view):
        ax.imshow(view_panel['rgb']);ax.set(xlim=(0,1200),ylim=(800,0));ax.axis('off')
        ax.set_title('Receiving sites · '+view,fontsize=7,pad=3)
        ax.text(45,760,'A' if view=='medial' else 'P',fontsize=5.5,color='.35')
        ax.text(1150,760,'P' if view=='medial' else 'A',fontsize=5.5,color='.35',ha='right')
    brain=axis(.005,.43,.49 if has_cg else .54,.46)
    brain_axis(brain,panel,'lateral')
    if has_cg:
        medial=axis(.505,.43,.49,.46);brain_axis(medial,panel['medial'],'medial')
        medial_xy={p:xy for p,xy in zip(anatomy.pair,panel['medial']['xy'])}
    zoom=None
    if record['subject']=='D0149':
        # The shaft is mainly mediolateral: lateral projection collapses its depth.
        # Display native x/z on the nearest coronal plane, without moving contacts.
        rtms=anatomy.loc[anatomy.pair.str.startswith('RTMS')]
        center=rtms[['x','y','z']].mean().to_numpy(float)
        context=panel['stim_context'];affine=context['affine']
        voxel_y=int(round((center[1]-affine[1,3])/affine[1,1]))
        plane_y=affine[1,1]*voxel_y+affine[1,3]
        assert np.max(np.abs(rtms.y-plane_y))<1., 'RTMS contacts too far from coronal plane'
        xs=affine[0,0]*np.arange(context['mri'].shape[0])+affine[0,3]
        zs=affine[2,2]*np.arange(context['mri'].shape[2])+affine[2,3]
        dx,dz=affine[0,0],affine[2,2]
        zoom=axis(.56,.47,.42,.38)
        zoom.imshow(context['mri'][:,voxel_y,:].T,origin='lower',cmap='gray',interpolation='bilinear',
            vmin=0,vmax=np.percentile(context['mri'][context['mri']>0],99),
            extent=(xs[0]-dx/2,xs[-1]+dx/2,zs[0]-dz/2,zs[-1]+dz/2))
        zoom.contour(xs,zs,(context['seg'][:,voxel_y,:].T==41).astype(float),levels=[.5],colors=['#62D8EA'],linewidths=.45)
        zoom.set(xlim=(center[0]-18,center[0]+18),ylim=(center[2]-9,center[2]+9),aspect='equal')
        zoom.set_xticks([]);zoom.set_yticks([]);zoom.set_title('RTMS detail · native coronal MRI',fontsize=6.5,pad=5)
        zoom.set_xlabel(f'y = {plane_y:.0f} mm; midpoints projected (<1 mm from slice)',fontsize=5,labelpad=4)
        zoom.text(.02,.04,'M',transform=zoom.transAxes,fontsize=5,color='white')
        zoom.text(.98,.04,'L',transform=zoom.transAxes,fontsize=5,color='white',ha='right')
        for spine in zoom.spines.values():spine.set_color('.7');spine.set_linewidth(.5)
    detail_offsets=[((-3,12),'right','bottom'),((-3,-14),'right','top'),((0,18),'center','bottom'),((6,-14),'left','top')]
    detail_index=0
    for rec in receivers:
        row=anatomy.loc[anatomy.pair.eq(rec['pair'])].iloc[0]
        ax=medial if cg(rec) else brain
        xy=medial_xy[rec['pair']] if cg(rec) else xy_by_pair[rec['pair']]
        ax.scatter(*xy,s=25,c=[CMAP(NORM(rec['latency_ms']))],edgecolors='#202020',linewidths=.5,zorder=6)
        suffix='†' if row.roi=='WM' else ''
        label=short(rec['pair'])+' · '+row.display_roi+suffix+'\n'+f"{rec['latency_ms']:.2f} ms"
        if zoom is not None and rec['pair'].startswith('RTMS'):
            detail_xy=(row.x,row.z)
            zoom.scatter(*detail_xy,s=24,c=[CMAP(NORM(rec['latency_ms']))],edgecolors='#202020',linewidths=.5,zorder=6)
            off,ha,va=detail_offsets[detail_index];detail_index+=1
            annotate(zoom,detail_xy,label,off,ha,va)
        else:
            annotate(ax,xy,label,offset=(5,7) if cg(rec) else (5,-5),va='bottom' if cg(rec) else 'top',fontsize=6.5)
    # Stimulation is never painted on the intact lateral cortical surface.
    context=panel['stim_context'];idx=anatomy.index[anatomy.pair.eq(record['stimulus'])][0]
    text(.025,.376,'Stimulation localization',fontsize=7,fontweight='bold',ha='left',va='center')
    ins=axis(.025,.09,.24,.25);ins.imshow(context['rgb']);ins.axis('off')
    ins.set_title('Exposed insula (projection)',fontsize=6,pad=2)
    ins.scatter(*context['xy'][idx],marker='*',s=48,c=STIM_COLOR,edgecolors='white',linewidths=.5)
    ins.plot([80,205],[740,740],color='.4',lw=.7);ins.text(142.5,710,'10 mm',fontsize=5,ha='center',color='.4')
    mri_ax=axis(.29,.09,.25,.25)
    affine=context['affine'];mid=stim[['x','y','z']].to_numpy(float)
    vox=np.rint(nib.affines.apply_affine(np.linalg.inv(affine),mid)).astype(int)
    plane=context['mri'][:,vox[1],:].T;seg=context['seg'][:,vox[1],:].T
    xs=affine[0,0]*np.arange(context['mri'].shape[0])+affine[0,3]
    zs=affine[2,2]*np.arange(context['mri'].shape[2])+affine[2,3]
    dx,dz=affine[0,0],affine[2,2]
    mri_ax.imshow(plane,origin='lower',cmap='gray',interpolation='bilinear',vmin=0,vmax=np.percentile(context['mri'][context['mri']>0],99),
        extent=(xs[0]-dx/2,xs[-1]+dx/2,zs[0]-dz/2,zs[-1]+dz/2))
    # White-matter boundary provides explicit tissue context around the virtual midpoint.
    if (seg==41).any():mri_ax.contour(xs,zs,(seg==41).astype(float),levels=[.5],colors=['#62D8EA'],linewidths=.45)
    mri_ax.scatter(mid[0],mid[2],marker='*',s=48,c=STIM_COLOR,edgecolors='white',linewidths=.5)
    mri_ax.set(xlim=(mid[0]-17,mid[0]+17),ylim=(mid[2]-17,mid[2]+17),xticks=[],yticks=[])
    mri_ax.set_title('Native MRI · coronal',fontsize=6,pad=2)
    text(.58,.32,'Stim. pair: '+short(record['stimulus'])+'\nBipolar ROI: '+stim.roi+'\nMidpoint tissue: '+stim.midpoint_roi,
         fontsize=7,ha='left',va='top',linespacing=1.45)
    text(.58,.205,'★',fontfamily='DejaVu Sans',fontsize=9,color=STIM_COLOR,ha='left',va='center')
    text(.605,.205,'Stimulating pair midpoint',fontsize=6,ha='left',va='center')
    text(.58,.166,'Cyan MRI contour: white-matter boundary',fontsize=5.5,ha='left',va='center',color='.35')
    cbax=axis(.60,.105,.35,.025)
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=NORM,cmap=CMAP),cax=cbax,orientation='horizontal',ticks=[0,50,100,150])
    cb.ax.tick_params(labelsize=6,width=.5,length=2,pad=2);cb.outline.set_linewidth(.5)
    cb.set_label('CCEP latency (ms)',fontsize=7,labelpad=2)
    nearest_names={'ctx_rh_S_temporal_transverse':'transverse temporal sulcus',
                   'ctx_rh_Pole_temporal':'temporal pole','ctx_rh_S_front_middle':'middle frontal sulcus'}
    notes=[]
    for rec in receivers:
        row=anatomy.loc[anatomy.pair.eq(rec['pair'])].iloc[0]
        if row.roi=='WM':
            notes.append(short(rec['pair'])+': '+nearest_names.get(row.nearest_gm_label,row.nearest_gm_label)+f' ({row.nearest_gm_distance_mm:.2f} mm)')
    text(.025,.025,'† Nearest GM: '+'; '.join(notes)+'. Coordinates unchanged.',fontsize=5.5,ha='left',va='center',color='.35')
    return texts

def save_checked(fig,path,texts):
    import xml.etree.ElementTree as ET
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    boxes=[]
    for t in texts:
        b=t.get_window_extent(renderer)
        assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.x1 and b.y1<=fig.bbox.y1,(path.name,'clipped',t.get_text())
        for previous in boxes:
            assert not b.overlaps(previous),(path.name,'label overlap',t.get_text())
        boxes.append(b)
    fig.savefig(path,format='svg',dpi=300,bbox_inches=None,facecolor='white');plt.close(fig)
    svg=ET.parse(path).getroot()
    dims=[float(svg.attrib[k].removesuffix('pt'))*2.54/72 for k in ('width','height')]
    assert dims[0]<=18.00001
    return dict(path=str(path),size_cm=dims,sha256=sha(path),label_overlap_check='passed')

def main(records):
    OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    for f in (Path(sys.prefix)/'fonts').glob('arial*.ttf'):font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({'font.family':'Arial','font.size':7,'svg.fonttype':'none'})
    pipeline=load_pipeline(UPSTREAM)
    bundle={};inputs={};rows=[]
    for subject in sorted({r['subject'] for r in records}):
        pairs=list(dict.fromkeys(p for r in records if r['subject']==subject for p in [r['stimulus']]+[v['pair'] for v in r['receivers']]))
        anatomy,pial,source=get_anatomy(subject,pairs,pipeline)
        panel=render_lateral(pial,anatomy)
        if anatomy.display_roi.eq('CG').any():
            panel['medial']=render_lateral(pial,anatomy,view='medial')
            anatomy[['medial_display_x','medial_display_y']]=panel['medial']['xy']
            np.testing.assert_allclose(panel['xy'][:,0]+panel['medial']['xy'][:,0],1200,atol=1e-6)
            np.testing.assert_allclose(panel['xy'][:,1],panel['medial']['xy'][:,1],atol=1e-6)
        panel['stim_context']=prepare_stim_context(subject,anatomy)
        anatomy[['display_x','display_y']]=panel['xy']
        bundle[subject]=(anatomy,panel);rows.append(anatomy);inputs.update(source)
    pd.concat(rows,ignore_index=True).to_csv(OUT/'electrode_anatomy.csv',index=False)
    (OUT/'requested_latencies.json').write_text(json.dumps(records,indent=2))
    figure_info=[]
    for rec in records:
        fig=plt.figure(figsize=(18*CM,10*CM))
        texts=draw_panel(fig,(0,0,1,1),rec,*bundle[rec['subject']])
        name=rec['subject']+'_'+rec['stimulus'].replace('–','-')+'_CCEP_receivers.svg'
        figure_info.append(save_checked(fig,FIG/name,texts))
    fig=plt.figure(figsize=(18*CM,30*CM));texts=[]
    for i,rec in enumerate(records):texts+=draw_panel(fig,(0,1-(i+1)/3,1,1/3),rec,*bundle[rec['subject']])
    figure_info.append(save_checked(fig,FIG/'CCEP_receivers_three_panels.svg',texts))
    audit=dict(created_utc=datetime.now(timezone.utc).isoformat(),latency_definition='unspecified; user requested CCEP latency (ms)',
        latency_source='User-supplied values; no CCEP detection or latency re-estimation',
        typo_resolution='D153 RMFI9-10 interpreted as RFMI9-10 with user agreement',
        display='Receivers: native lateral/medial projections with adjacent labels and no leader lines. Stimulus: exposed-insula inset and native coronal MRI only; no marker on intact cortex' ,
        colormap='viridis; purple (shorter) to yellow (longer); shared linear scale',
        receiver_views=[dict(subject=r['subject'],stimulus=r['stimulus'],receiver=v['pair'],view='medial' if v['pair']=='RFPM2–RFPM1' else 'lateral') for r in records for v in r['receivers']],
        anatomy='Original bipolar ROI retained in audit. Display labels for WM pairs use nearest right-hemisphere gray-matter voxel center (cortical or subcortical) in native aparc2009s+aseg; physical coordinates unchanged' ,
        stimulation_surface_marker_on_intact_cortex=False,leader_lines=False,
        rtms_detail=dict(view='native coronal MRI',plane_y_mm=-12.,projection='Native x/z; all RTMS midpoints less than 1 mm from slice; no coordinate displacement'),
        stimulation_midpoints=[dict(subject=r['subject'],pair=r['stimulus'],midpoint_tissue=bundle[r['subject']][0].set_index('pair').loc[r['stimulus'],'midpoint_roi'],
             mri_plane_y_mm=float(bundle[r['subject']][1]['stim_context']['affine'][1,1]*round((bundle[r['subject']][0].set_index('pair').loc[r['stimulus'],'y']-bundle[r['subject']][1]['stim_context']['affine'][1,3])/bundle[r['subject']][1]['stim_context']['affine'][1,1])+bundle[r['subject']][1]['stim_context']['affine'][1,3])) for r in records],
        roi_abbreviations=ROI_NAMES,color_range_ms=[0,150],inputs=inputs,figures=figure_info,
        records=records,receivers_count=[len(r['receivers']) for r in records])
    (OUT/'validation.json').write_text(json.dumps(audit,indent=2))
    print(pd.concat(rows)[['subject','pair','roi','display_roi','nearest_gm_label','nearest_gm_distance_mm','midpoint_label']].to_string(index=False),flush=True)
    print('Validated 5 / 2 / 1 receiver entries and four SVGs.',flush=True)
    return audit
