"""Wide-context native MRI and unshifted 3D localization of each pair.

CT is retained only for the existing localization audit, never displayed.
"""
from functools import lru_cache
import hashlib
import json
import re
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from src.main_analysis import ROI_COLORS,TASK_DISPLAY_NAMES

ROOT=Path(__file__).resolve().parents[2]
RECON=Path('/cwork/ns458/ECoG_Recon')
OUT=ROOT/'results/supplementary/operculum/anatomy'
INSULA_NAMES={'G_insular_short','G_Ins_lg_and_S_cent_ins','S_circular_insula_ant','S_circular_insula_inf','S_circular_insula_sup'}
COLORS={'AIC':ROI_COLORS['Insula'],'FO':ROI_COLORS['IFG']}
# Physical-contact dimensions and MRI projection tolerance copied from the
# user-confirmed Figure 1 fig1-seeg-plot cell. Coordinates are never moved.
MRI_PROJECTION_TOL_MM=4.0
CONTACT_HEIGHT_MM=2.0


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=1)
def coordinate_rows():
    frame=pd.read_csv(ROOT/'results/audit/sound_only_cohort_2026-10-17/coordinate_variants.csv')
    return frame.loc[frame.description.eq('Repeat')&frame.modality.eq('sound')]


@lru_cache(maxsize=6)
def load_subject(subject,hemisphere):
    from nibabel.processing import resample_from_to
    # D0057 exports match the 2022 D57_2 LEPTO reconstruction, not the
    # 2021 images in D57. This changes image provenance, never point values.
    recon='D57_2' if subject=='D0057' else 'D'+str(int(subject[1:]))
    folder=RECON/recon
    native=nib.load(folder/'mri/orig.mgz')
    tkr=native.header.get_vox2ras_tkr();inverse=np.linalg.inv(tkr)
    ct_path=folder/'elec_recon'/('ctINt1.nii.gz' if subject=='D0032' else 'postInPre.nii.gz')
    ct=nib.load(ct_path)
    source_ct_shape=ct.shape;source_ct_affine=ct.affine.copy()
    if ct.shape!=native.shape or not np.allclose(ct.affine,native.affine,atol=1e-3):
        ct=resample_from_to(ct,(native.shape,native.affine),order=1)
    ct_data=np.asarray(ct.dataobj,dtype=np.float32)
    mri=np.asarray(native.dataobj,dtype=np.float32)
    atlas=nib.load(folder/'mri/aparc.a2009s+aseg.mgz')
    np.testing.assert_allclose(atlas.affine,native.affine,atol=1e-4)
    aseg=nib.load(folder/'mri/aseg.mgz')
    lut={}
    path=Path('/hpc/group/coganlab/nanlinshi/seeg-preprocessing/common/FreeSurferColorLUT.txt')
    for line in path.read_text().splitlines():
        parts=line.split()
        if len(parts)>1 and parts[0].isdigit():lut[int(parts[0])]=parts[1]
    hemi='lh' if hemisphere=='L' else 'rh'
    vertices,faces=nib.freesurfer.read_geometry(folder/'surf'/f'{hemi}.pial')
    labels,_,names=nib.freesurfer.read_annot(folder/'label'/f'{hemi}.aparc.a2009s.annot')
    decoded=[name.decode().replace('&','_and_') for name in names]
    insula_ids=[i for i,n in enumerate(decoded) if n in INSULA_NAMES]
    fo_ids=[i for i,n in enumerate(decoded) if n=='G_front_inf-Opercular']
    assert len(insula_ids)==5 and len(fo_ids)==1
    name_file=folder/'elec_recon'/f'{recon}.electrodeNames'
    position_file=folder/'elec_recon'/f'{recon}.LEPTO'
    contact_names=[line.split()[0] for line in name_file.read_text().splitlines()[2:] if line.strip()]
    contact_values=np.loadtxt(position_file,skiprows=2)
    assert len(contact_names)==len(contact_values)
    recon_contacts=dict(zip(contact_names,contact_values))
    files=[folder/'mri/orig.mgz',folder/'mri/aparc.a2009s+aseg.mgz',ct_path,
           folder/'surf'/f'{hemi}.pial',folder/'label'/f'{hemi}.aparc.a2009s.annot',
           folder/'elec_recon/T1.nii.gz',folder/'elec_recon/ct2t1.mat',name_file,position_file]
    result=dict(subject=subject,folder=folder,native=native,tkr=tkr,inverse=inverse,
        ct=ct_data,mri=mri,aparc=np.asarray(atlas.dataobj),aseg=np.asarray(aseg.dataobj),lut=lut,
        vertices=vertices,faces=faces,surface_labels=labels,insula_ids=insula_ids,fo_ids=fo_ids,
        ct_path=ct_path,mri_limits=np.percentile(mri[mri>0],[1,99.5]),
        ct_threshold=1000. if ct_data.max()>1500 else float(np.percentile(ct_data,99.5)),
        ct_upper=float(ct_data.max()),hemi=hemi,recon_contacts=recon_contacts,recon=recon)
    OUT.mkdir(parents=True,exist_ok=True)
    report=dict(subject=subject,reconstruction=recon,registered_CT=str(ct_path),source_CT_shape=source_ct_shape,
        source_CT_affine=source_ct_affine.tolist(),native_MRI_affine=native.affine.tolist(),
        resampling='Existing registered CT sampled in native MRI scanner-RAS geometry; no new registration',
        point_space='Original analysis native tkRAS millimeters; no cortical snapping or electrode displacement',
        files={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files if p.is_file()})
    (OUT/f'{subject}_image_sources.json').write_text(json.dumps(report,indent=2)+'\n')
    return result


def pair_points(row,subject):
    frame=coordinate_rows();points=[]
    for region,column in [('AIC','insula_site'),('FO','operculum_site')]:
        channel=row[column]
        one=frame.loc[frame.task.eq(row['task'])&frame.channel.eq(channel)]
        cols=['contact_1','contact_2','x1_native','y1_native','z1_native','x2_native','y2_native','z2_native','x_native','y_native','z_native']
        one=one[cols].drop_duplicates();assert len(one)==1
        r=one.iloc[0]
        a=np.array([r[f'{v}1_native'] for v in 'xyz'],float)
        b=np.array([r[f'{v}2_native'] for v in 'xyz'],float)
        midpoint=np.array([r[f'{v}_native'] for v in 'xyz'],float)
        np.testing.assert_allclose((a+b)/2,midpoint,atol=1e-5)
        for role,name,xyz in [('contact_1',r.contact_1,a),('midpoint',channel.split('_',1)[1],midpoint),('contact_2',r.contact_2,b)]:
            if role!='midpoint':
                np.testing.assert_allclose(xyz,subject['recon_contacts'][name],atol=1e-3,
                    err_msg='Analysis point does not match this reconstruction LEPTO contact')
            ijk=nib.affines.apply_affine(subject['inverse'],xyz)
            nearest=np.rint(ijk).astype(int)
            label=int(subject['aparc'][tuple(nearest)])
            lo=np.maximum(nearest-2,0);hi=np.minimum(nearest+3,subject['ct'].shape)
            ct_peak=float(subject['ct'][tuple(slice(x,y) for x,y in zip(lo,hi))].max())
            points.append(dict(task=row['task'],subject=row['subject'],region=region,channel=channel,
                role=role,contact=name,x=xyz[0],y=xyz[1],z=xyz[2],
                native_aparc=subject['lut'].get(label,str(label)),ct_peak_within_2vox=ct_peak))
    result=pd.DataFrame(points)
    assert len(result)==6
    return result


def localization_review_notes(row,points):
    notes=[]
    fo=points.loc[points.region.eq('FO')]
    mid=fo.loc[fo.role.eq('midpoint'),'native_aparc'].iloc[0]
    if 'G_front_inf-Opercular' not in mid:
        notes.append('FO candidate midpoint is '+mid.replace('ctx_lh_','').replace('ctx_rh_','')+' in matching MRI')
    endpoints=fo.loc[fo.role.ne('midpoint')]
    bad=endpoints.loc[~endpoints.native_aparc.str.contains('G_front_inf-Opercular|Cerebral-White-Matter',regex=True)]
    if len(bad):notes.append('FO endpoint tissue/region pending: '+', '.join(bad.contact))
    if row['subject']=='D0079' and 'LAI1' in set(points.contact):
        notes.append('D0079 LAI1 is ~6 mm from CT intensity ≥1000; localization not confirmed')
    return notes


def full_shank_contacts(subject,points):
    """Use every reconstructed real contact on the two selected pair shafts."""
    selected=points.loc[points.role.ne('midpoint')].set_index('contact')
    prefixes={re.fullmatch(r'(.+?)(\d+)',name).group(1) for name in selected.index}
    records=[]
    for prefix in sorted(prefixes):
        contacts=[]
        for name,xyz in subject['recon_contacts'].items():
            match=re.fullmatch(re.escape(prefix)+r'(\d+)',name)
            if match:contacts.append((int(match.group(1)),name,np.asarray(xyz,float)))
        contacts.sort()
        assert len(contacts)>=2,(subject['subject'],prefix)
        for number,name,xyz in contacts:
            ijk=np.rint(nib.affines.apply_affine(subject['inverse'],xyz)).astype(int)
            assert ((ijk>=0)&(ijk<np.array(subject['aparc'].shape))).all()
            label=subject['lut'].get(int(subject['aparc'][tuple(ijk)]),'Unknown')
            native_region='Other'
            if any(label.endswith('_'+insula) for insula in INSULA_NAMES):native_region='Insula'
            if label.endswith('_G_front_inf-Opercular'):native_region='Frontal operculum'
            records.append(dict(subject=subject['subject'],reconstruction=subject['recon'],
                shaft=prefix,contact_number=number,contact=name,x=xyz[0],y=xyz[1],z=xyz[2],
                native_aparc=label,native_region=native_region,
                selected_pair=selected.loc[name,'region'] if name in selected.index else ''))
    result=pd.DataFrame(records)
    assert result.contact.is_unique
    assert set(selected.index).issubset(set(result.contact))
    return result


def contact_color(region):
    return COLORS['AIC'] if region=='Insula' else COLORS['FO'] if region=='Frontal operculum' else '#777777'


def sample_plane(subject,key,xyz,axis1,axis2,half=18.,spacing=.20,order=1):
    scale=np.arange(-half,half+spacing*.5,spacing)
    xx,yy=np.meshgrid(scale,scale,indexing='xy')
    world=xyz+xx[...,None]*axis1+yy[...,None]*axis2
    ijk=nib.affines.apply_affine(subject['inverse'],world).reshape(-1,3)
    plane=map_coordinates(subject[key],ijk.T,order=order,mode='constant',cval=0).reshape(xx.shape)
    return scale,plane


def draw_mri(ax,subject,points,region,plane_number,shanks,region_text=True):
    frame=points.loc[points.region.eq(region)]
    xyz=frame[['x','y','z']].to_numpy()
    # Figure 1 uses a standard native sagittal plane, not a plane inferred
    # from a single bipolar pair. Add its orthogonal coronal companion.
    axis1=np.array([0.,1.,0.]) if plane_number==1 else np.array([1.,0.,0.])
    axis2=np.array([0.,0.,1.])
    normal=np.cross(axis1,axis2)
    # Move the *view center within the same plane*, never the contacts.
    # The field includes the surrounding hemisphere, Sylvian region and
    # midline rather than centering a tiny crop on the electrode tip.
    center=xyz[1]-np.dot(xyz[1],axis1)*axis1-np.dot(xyz[1],axis2)*axis2
    scale,t1=sample_plane(subject,'mri',center,axis1,axis2,half=90.,spacing=.5)
    _,atlas=sample_plane(subject,'aparc',center,axis1,axis2,half=90.,spacing=.5,order=0)
    extent=[scale[0],scale[-1],scale[0],scale[-1]]
    ax.imshow(t1,cmap='gray',origin='lower',extent=extent,
              vmin=subject['mri_limits'][0],vmax=subject['mri_limits'][1],interpolation='none')
    for label_region in ['AIC','FO']:
        prefix=f'ctx_{subject["hemi"]}_'
        if label_region=='AIC':
            codes=[code for code,name in subject['lut'].items()
                   if name.startswith(prefix) and any(n in name for n in INSULA_NAMES)]
        else:
            codes=[code for code,name in subject['lut'].items()
                   if name.startswith(prefix) and 'G_front_inf-Opercular' in name]
        mask=np.isin(atlas,codes)
        if mask.any() and not mask.all():
            from matplotlib.colors import ListedColormap
            ax.imshow(np.ma.masked_where(~mask,np.ones(mask.shape)),
                cmap=ListedColormap([COLORS[label_region]]),origin='lower',extent=extent,
                alpha=.18,interpolation='none',vmin=0,vmax=1)
            ax.contour(scale,scale,mask,levels=[.5],colors=[COLORS[label_region]],linewidths=.65,alpha=.85)
            # region_text defaults on so the automatic-control anatomy figures
            # keep their Insula / Frontal operculum slice annotations.
            if region_text:
                yy,xx=np.nonzero(mask)
                anchor=(float(np.median(scale[xx])),float(np.median(scale[yy])))
                label='Insula' if label_region=='AIC' else 'Frontal operculum'
                offset=(-34,18) if label_region=='AIC' else (14,-22)
                ax.annotate(label,anchor,xytext=offset,textcoords='offset points',
                    fontsize=5.5,color=COLORS[label_region],
                    arrowprops=dict(arrowstyle='-',color=COLORS[label_region],lw=.6),
                    bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=.8))
    # Exactly the Figure 1 display convention: project contacts within
    # +/-4 mm, explicitly identified as projection rather than an MRI slab.
    for _,shaft in shanks.groupby('shaft',sort=False):
        path=shaft[['x','y','z']].to_numpy()
        depth=(path-center)@normal
        xy=np.column_stack([(path-center)@axis1,(path-center)@axis2])
        for i in range(len(path)-1):
            d0,d1=depth[i:i+2];lo,hi=0.,1.
            if abs(d1-d0)<1e-10:
                if abs(d0)>MRI_PROJECTION_TOL_MM:continue
            else:
                bounds=sorted([(-MRI_PROJECTION_TOL_MM-d0)/(d1-d0),(MRI_PROJECTION_TOL_MM-d0)/(d1-d0)])
                lo=max(lo,bounds[0]);hi=min(hi,bounds[1])
                if lo>hi:continue
            visible=xy[i]+np.array([lo,hi])[:,None]*(xy[i+1]-xy[i])
            ax.plot(visible[:,0],visible[:,1],color='#B8B8B8',lw=.55,alpha=.9,zorder=5)
        tangent=path[-1]-path[0]
        direction=np.array([np.dot(tangent,axis1),np.dot(tangent,axis2)])
        direction/=max(np.linalg.norm(direction),1e-12)
        for i,r in enumerate(shaft.itertuples()):
            if abs(depth[i])<=MRI_PROJECTION_TOL_MM:
                ends=xy[i]+np.array([-.7,.7])[:,None]*direction
                ax.plot(ends[:,0],ends[:,1],color=contact_color(r.native_region),
                        lw=.8,solid_capstyle='round',zorder=6)
    projected=np.column_stack([(xyz-center)@axis1,(xyz-center)@axis2])
    ax.plot(projected[[0,2],0],projected[[0,2],1],c=COLORS[region],lw=.7)
    for i,r in enumerate(frame.itertuples()):
        x,y=projected[i]
        ax.scatter(x,y,s=9 if i==1 else 6,marker='D' if i==1 else 'o',
                   facecolors='none',edgecolors=COLORS[region],linewidths=.6,zorder=8)
        offset=[(-5,8),(0,-13),(5,8)][i]
        ax.annotate('BP' if i==1 else r.contact,(x,y),xytext=offset,
                    ha=['right','center','left'][i],
                    textcoords='offset points',color='white',fontsize=5.5,
                    bbox=dict(facecolor='black',alpha=.55,edgecolor='none',pad=.6))
    ax.plot([-78,-58],[-76,-76],c='white',lw=1)
    ax.text(-68,-83,'20 mm',color='white',ha='center',fontsize=5)
    for axis,positive,negative in [(axis1,(.98,.5),(.02,.5)),(axis2,(.5,.98),(.5,.02))]:
        dominant=int(np.argmax(np.abs(axis)))
        pos=['R','A','S'][dominant];neg=['L','P','I'][dominant]
        if axis[dominant]<0:pos,neg=neg,pos
        for name,where in [(pos,positive),(neg,negative)]:
            ax.text(*where,name,transform=ax.transAxes,color='white',fontsize=6,
                ha='center',va='center',bbox=dict(facecolor='black',alpha=.5,edgecolor='none',pad=.5))
    label=region
    if region=='FO' and not frame.loc[frame.role.eq('midpoint'),'native_aparc'].str.contains('G_front_inf-Opercular').all():
        label='FO candidate (review)'
    orientation='sagittal' if plane_number==1 else 'coronal'
    ax.set_title(f'{label} · native {orientation} MRI\n{frame.channel.iloc[0]} · contacts projected within ±4 mm',fontsize=6.5)
    ax.set_xlim(-90,90);ax.set_ylim(-90,90);ax.set_aspect('equal');ax.axis('off')


def render_surface(subject,points,shanks,zoom=False):
    import pyvista as pv
    faces=subject['faces'];vertices=subject['vertices'];labels=subject['surface_labels']
    plotter=pv.Plotter(off_screen=True,notebook=False,window_size=(1000,850))
    try:
        plotter.set_background('white')
        mesh=pv.PolyData(vertices,np.column_stack([np.full(len(faces),3),faces]).ravel())
        plotter.add_mesh(mesh,color='#C8C8C8',opacity=.12,smooth_shading=True,ambient=.45)
        for region,ids in [('AIC',subject['insula_ids']),('FO',subject['fo_ids'])]:
            selected=np.isin(labels,ids)
            triangles=faces[selected[faces].all(axis=1)]
            roi=pv.PolyData(vertices,np.column_stack([np.full(len(triangles),3),triangles]).ravel())
            plotter.add_mesh(roi,color=COLORS[region],opacity=.40 if region=='FO' else .45,
                             smooth_shading=True,ambient=.5,diffuse=.5)
        for _,shaft in shanks.groupby('shaft',sort=False):
            coords=shaft[['x','y','z']].to_numpy()
            axis=coords[-1]-coords[0];axis/=np.linalg.norm(axis)
            tip=coords[0]-1.5*axis;end=coords[-1]+8.*axis
            path=np.vstack([tip,coords,end])
            shaft_radius=.5 if zoom else .45
            ring_radius=.8 if zoom else .68
            for a,b in zip(path[:-1],path[1:]):
                delta=b-a;height=np.linalg.norm(delta)
                if height>1e-6:
                    plotter.add_mesh(pv.Cylinder(center=(a+b)/2,direction=delta,
                        height=height,radius=shaft_radius,resolution=24),
                        color='#B8B8B8',smooth_shading=True)
            plotter.add_mesh(pv.Sphere(radius=shaft_radius,center=tip,
                theta_resolution=16,phi_resolution=16),color='#B8B8B8',smooth_shading=True)
            for i,r in enumerate(shaft.itertuples()):
                local=coords[min(i+1,len(coords)-1)]-coords[max(0,i-1)]
                plotter.add_mesh(pv.Cylinder(center=[r.x,r.y,r.z],direction=local,
                    height=CONTACT_HEIGHT_MM,radius=ring_radius,resolution=24),
                    color=contact_color(r.native_region) if r.native_region!='Other' else '#595959',
                    smooth_shading=True)
        xyz=points[['x','y','z']].to_numpy();outward=-1 if subject['hemi']=='lh' else 1
        all_xyz=shanks[['x','y','z']].to_numpy()
        center=(all_xyz.min(0)+all_xyz.max(0))/2 if zoom else (vertices.min(0)+vertices.max(0))/2
        direction=np.array([outward*220.,0.,0.])
        if zoom:
            # View across the electrode shaft so real contacts and their
            # midpoint do not collapse into the same projected location.
            shaft=xyz[2]-xyz[0];shaft/=np.linalg.norm(shaft)
            lateral=np.array([float(outward),0.,0.])
            cross_view=lateral-np.dot(lateral,shaft)*shaft
            if np.linalg.norm(cross_view)<.15:
                anterior=np.array([0.,1.,.25])
                cross_view=anterior-np.dot(anterior,shaft)*shaft
            cross_view/=np.linalg.norm(cross_view)
            direction=220*(cross_view+.20*lateral)
            if shanks.shaft.nunique()>1:
                # Keep both complete shafts visible: an AIC-only cross view
                # can look straight down the opercular shaft and hide it.
                axes=[]
                for _,group in shanks.groupby('shaft',sort=False):
                    values=group[['x','y','z']].to_numpy()
                    axis=values[-1]-values[0];axes.append(axis/np.linalg.norm(axis))
                candidates=[]
                for anterior in [-1.5,-1.,-.5,.5,1.,1.5]:
                    for superior in [-.75,0.,.75]:
                        vector=np.array([outward,anterior,superior],float)
                        vector/=np.linalg.norm(vector)
                        visibility=min(np.sqrt(max(0.,1-np.dot(vector,axis)**2)) for axis in axes)
                        candidates.append((visibility+.08*np.dot(vector,lateral)+.01*vector[1],vector))
                direction=220*max(candidates,key=lambda item:item[0])[1]
        plotter.camera_position=[center+direction,center,[0,0,1]]
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale=max(22.,np.linalg.norm(np.ptp(all_xyz,axis=0))/2+12.) if zoom else 90.
        plotter.show(auto_close=False)
        image=plotter.screenshot(return_img=True)
        transform=plotter.camera.GetCompositeProjectionTransformMatrix(1000/850,-1,1)
        matrix=np.array([[transform.GetElement(i,j) for j in range(4)] for i in range(4)])
        complete=np.column_stack([all_xyz,np.ones(len(all_xyz))])@matrix.T
        complete=complete[:,:3]/complete[:,3,None]
        assert (np.abs(complete[:,:2])<1).all(),'Complete shank clipped by 3D view'
        homogeneous=np.column_stack([xyz,np.ones(len(xyz))])@matrix.T
        ndc=homogeneous[:,:3]/homogeneous[:,3,None]
        pixels=np.column_stack([(ndc[:,0]+1)*500,(1-ndc[:,1])*425])
        assert ((pixels[:,0]>=0)&(pixels[:,0]<=1000)&(pixels[:,1]>=0)&(pixels[:,1]<=850)).all()
        return image,pixels
    finally:plotter.close()


def make_pair_figure(row):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    subject=load_subject(row['subject'],row['hemisphere'])
    points=pair_points(row,subject)
    shanks=full_shank_contacts(subject,points)
    pair_id=f'{row["task"]}__{row["insula_site"]}__{row["operculum_site"]}'
    points.to_csv(OUT/f'{pair_id}__points.csv',index=False)
    shanks.assign(pair_id=pair_id).to_csv(OUT/f'{pair_id}__shanks.csv',index=False)
    review_notes=localization_review_notes(row,points)
    fig,axes=plt.subplots(3,2,figsize=(18/2.54,18/2.54))
    for i,plane in enumerate([1,2]):
        for j,region in enumerate(['AIC','FO']):draw_mri(axes[i,j],subject,points,region,plane,shanks)
    for j,zoom in enumerate([False,True]):
        ax=axes[2,j];image,pixels=render_surface(subject,points,shanks,zoom)
        ax.imshow(image);ax.axis('off')
        for region in ['AIC','FO']:
            inds=np.flatnonzero(points.region.eq(region))
            xy=pixels[inds]
            ax.plot(xy[[0,2],0],xy[[0,2],1],c=COLORS[region],lw=.7)
            ax.scatter(xy[[0,2],0],xy[[0,2],1],s=12,marker='o',c=COLORS[region],edgecolors='white',linewidths=.25)
            ax.scatter(xy[1,0],xy[1,1],s=25,marker='D',facecolors='white',edgecolors=COLORS[region],linewidths=1)
            name='Candidate' if region=='FO' and any(n.startswith('FO candidate') for n in review_notes) else region
            ax.annotate(name,xy[1],xytext=(6,9 if region=='AIC' else -12),textcoords='offset points',
                        color=COLORS[region],fontsize=6,bbox=dict(facecolor='white',alpha=.6,edgecolor='none',pad=.5))
            if zoom:
                for pos,index in [(0,inds[0]),(2,inds[2])]:
                    ax.annotate(points.iloc[index].contact,pixels[index],
                        xytext=(-5,8) if pos==0 else (5,8),
                        ha='right' if pos==0 else 'left',textcoords='offset points',
                        color=COLORS[region],fontsize=5,
                        bbox=dict(facecolor='white',alpha=.7,edgecolor='none',pad=.3))
        ax.set_title('Complete native shanks · '+('oblique view' if zoom else 'lateral overview'),fontsize=6.5)
    labels=[]
    for region in ['AIC','FO']:
        group=points.loc[points.region.eq(region)]
        simple=[r.native_aparc.replace('ctx_lh_','').replace('ctx_rh_','').replace('Left-Cerebral-','').replace('Right-Cerebral-','') for r in group.itertuples()]
        labels.append(f'{region}: '+ ' | '.join(f'{r.contact}: {label}' for r,label in zip(group.itertuples(),simple)))
    fig.suptitle(f'{TASK_DISPLAY_NAMES[row["task"]]} | {row["insula_site"]} — {row["operculum_site"]}\nNative MRI + individual 3D cortex · {row["native_distance_mm"]:.1f} mm between bipolar midpoints',fontsize=8,y=.985)
    fig.legend(handles=[Line2D([],[],marker='o',ls='',mfc='none',mec='black',label='Real monopolar contact'),
        Line2D([],[],marker='D',ls='',mfc='none',mec='black',label='Virtual bipolar midpoint'),
        Line2D([],[],color=COLORS['AIC'],label='Native insula contacts / boundary'),
        Line2D([],[],color=COLORS['FO'],label='Native operculum contacts / boundary'),
        Line2D([],[],color='#777777',lw=2,label='Physical shank / other contacts')],
        loc='lower center',bbox_to_anchor=(.5,.075),ncol=2,fontsize=6,frameon=False)
    note='Physical shanks in 3D; native sagittal/coronal MRI with ±4 mm contact projection (Figure 1 convention).'
    if review_notes:note='REVIEW: '+'\n'.join(review_notes)
    fig.text(.5,.055,note,ha='center',fontsize=5.5,color='#B22222' if review_notes else 'black')
    fig.text(.02,.030,labels[0],fontsize=4.8);fig.text(.02,.012,labels[1],fontsize=4.8)
    fig.subplots_adjust(left=.025,right=.975,top=.89,bottom=.16,hspace=.40,wspace=.10)
    return fig,points
