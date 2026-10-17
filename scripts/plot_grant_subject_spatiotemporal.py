"""Native electrode localization beside audited single-subject HGA traces.

Run on Slurm with Xvfb. Uses the established SHI-100 surface/camera and
CCEP native-MRI slice conventions; never snaps contacts or relabels inputs.
"""
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
from matplotlib import font_manager, patheffects
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from screen_recon_aic_contacts import read_contacts, load_pipeline, UPSTREAM
from grant_native_surface_helpers import prepare_surface, render_panel, INSULA_NAMES

ROOT = Path('/hpc/group/coganlab/nanlinshi/insula-functional')
RECON = Path('/cwork/ns458/ECoG_Recon')
MAPER = Path('/cwork/ns458/maper_run')
OUT = ROOT/'results/grant_subject_spatiotemporal'
FIG = ROOT/'img/grant_subject_spatiotemporal'
WINDOWS = {'Stimulus':(-.5,.6),'Delay':(0,1),'Response':(-.5,1.5)}
CM = 1/2.54
FIRST, SECOND, CENTER = '#00b8ce', '#e08000', '#792a8a'
GOLD = '#B18C2B'
INSULA_LABELS = {'ctx_rh_'+name for name in INSULA_NAMES}

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def intervals(mask,times):
    delta=np.diff(np.r_[False,np.asarray(mask,bool),False].astype(int))
    return [(times[a],times[b-1]+1/128) for a,b in zip(np.flatnonzero(delta==1),np.flatnonzero(delta==-1))]

def load_subject(subject,pipeline):
    recon='D'+str(int(subject[1:]))
    records,coordpath,_=read_contacts(RECON/recon)
    lookup={r['contact']:np.array([r['x'],r['y'],r['z']]) for r in records}
    if subject=='D0153':
        table=ROOT/'results/grant_d0153_highgamma/insula_electrodes.csv'
        traces=ROOT/'results/grant_d0153_highgamma/timecourse_mean_sem.csv'
        anatomy=pd.read_csv(table)
        assert set(anatomy.name)=={'D0153_RFMI8-9','D0153_RFMI9-10','D0153_RIA1-2','D0153_RTAS1-2'}
        data=pd.read_csv(traces)
        conditions=('Decision','Repeat')
    else:
        table=ROOT/'results/grant_d0149_highgamma/electrode_anatomy.csv'
        traces=ROOT/'results/grant_d0149_highgamma/timecourse_mean_sem.csv'
        anatomy=pd.read_csv(table)
        data=pd.read_csv(traces).query("modality == 'sound'").copy()
        conditions=('Passive','Repeat')
    assert set(data.channel)==set(anatomy.name)
    aparc=pipeline._resolve_atlas('aparc2009s',str(RECON),recon,subject,str(UPSTREAM.parent))
    hammers=pipeline._resolve_atlas('hammers',str(RECON),recon,subject,str(UPSTREAM.parent),maper_run=str(MAPER))
    crosswalk=dict(aparc[3]);crosswalk.update({s:'INS' for s in INSULA_LABELS})
    crosswalk.update({k:v for k,v in hammers[3].items() if k.startswith('insula ')})
    points=[]
    for row in anatomy.itertuples():
        xyz1,xyz2=lookup[row.contact_1],lookup[row.contact_2]
        midpoint=(xyz1+xyz2)/2
        np.testing.assert_allclose(midpoint,[row.x,row.y,row.z],atol=1e-6)
        merged=[]
        for role,name,xyz in [('contact_1',row.contact_1,xyz1),('midpoint','BP center',midpoint),('contact_2',row.contact_2,xyz2)]:
            a=pipeline._label_coordinate(xyz,aparc[1],aparc[0],aparc[2])
            h=pipeline._label_coordinate(xyz,hammers[1],hammers[0],hammers[2])
            # Match the subject's existing source atlas; do not invent a new consensus.
            label=h if subject=='D0153' and a in INSULA_LABELS and h.startswith('insula ') and h.endswith(' R') else a
            expected=getattr(row,role+'_label') if role!='midpoint' else getattr(row,'midpoint_label',getattr(row,'center',None))
            assert label==expected,(subject,row.name,role,label,expected)
            merged.append(label)
            points.append(dict(subject=subject,channel=row.name,role=role,contact=name,
                x=xyz[0],y=xyz[1],z=xyz[2],aparc_label=a,hammers_gm_label=h,
                consensus_point_label=label,channel_roi=row.roi,mixed=bool(row.mix)))
        assert pipeline.consensus_roi(merged,crosswalk)==(row.roi,bool(row.mix)),row.name
    points=pd.DataFrame(points)
    surface=prepare_surface(recon,'rh',hammers[0],hammers[1])
    panel=render_panel(recon,'rh',surface,points)
    points[['display_x','display_y']]=panel['xy']
    paths=[coordpath,table,traces,UPSTREAM,RECON/recon/'mri/orig.mgz',
           RECON/recon/'mri/aparc.a2009s+aseg.mgz',surface['pial_path'],surface['annot_path']]
    paths.extend(MAPER/subject/'output'/f'f30-seg95-{subject}{suffix}.nii.gz' for suffix in ('','-tc3crisp'))
    return dict(subject=subject,recon=recon,anatomy=anatomy,data=data,conditions=conditions,
                points=points,panel=panel,aparc=aparc,hammers=hammers,
                inputs=[dict(path=str(p),sha256=sha(p)) for p in paths])

def draw_surface(ax,subject,channel,zoom=1.,boundary_note=True):
    if subject.get('cortical_localization',False):
        panel=subject['panel'];row=subject['points'].set_index('canonical_channel').loc[channel]
        # Preserve the native screenshot pixels in SVG rather than downsampling
        # the small anatomical inset at the figure's default 100 dpi.
        ax.imshow(panel['rgb'],interpolation='none');ax.scatter(*panel['xy'][0],s=28,marker='+',c=CENTER,linewidths=1,zorder=7)
        ax.set_title(channel.split('_',1)[1]+' · WM\nnear '+row.display_roi,fontsize=7,pad=5)
        ax.text(70,760,'P',fontsize=5.5,color='.4');ax.text(1130,760,'A',fontsize=5.5,color='.4',ha='right')
        ax.set(xlim=(0,1200),ylim=(800,0),aspect='equal');ax.axis('off')
        return
    panel=subject['panel'];points=subject['points'].query('channel == @channel')
    assert len(points)==3
    # Enlarge the anatomy within the existing panel, applying exactly the same
    # display transform to the surface, boundaries, midpoint, and scale bar.
    center=np.array([475.,460.])
    def enlarged(xy):return center+(np.asarray(xy)-center)*zoom
    h,w=panel['rgb'].shape[:2]
    edges=enlarged([[-.5,-.5],[w-.5,h-.5]])
    ax.imshow(panel['rgb'],extent=(edges[0,0],edges[1,0],edges[1,1],edges[0,1]))
    ax.add_collection(LineCollection([enlarged(s) for s in panel['minor']],colors='.45',linewidths=.35,linestyles='dashed'))
    ax.add_collection(LineCollection([enlarged(s) for s in panel['major']],colors=GOLD,linewidths=.6))
    xy=enlarged(points[['display_x','display_y']].to_numpy())
    ax.scatter(*xy[1],s=28,marker='+',c=CENTER,linewidths=1,zorder=7)
    for x,y,text in [(70,450,'P'),(810,450,'A'),(450,205,'S'),(450,720,'I')]:
        ax.text(x,y,text,color='.45',fontsize=6,ha='center',va='center')
    scale=10*900/86*zoom
    ax.plot([75,75+scale],[695,695],c='.4',lw=.8)
    ax.text(75+scale/2,730,'10 mm',ha='center',fontsize=5.5,color='.4')
    roi=points.channel_roi.iloc[0]
    ax.set_title(channel.split('_',1)[1]+' · '+roi+('*' if points.mixed.iloc[0] and boundary_note else ''),fontsize=7.5,pad=5)
    ax.set(xlim=(0,950),ylim=(755,165),aspect='equal')
    ax.axis('off')

def save_checked(fig,path,layout):
    import xml.etree.ElementTree as ET
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    texts=list(fig.texts)
    for ax in fig.axes:
        if not ax.axison:
            texts.append(ax.title)
            continue
        texts.extend([ax.title,ax.xaxis.label,ax.yaxis.label])
        # Matplotlib creates some locator ticks outside the view interval but
        # does not draw them. Check only labels that appear in the export.
        for axis in (ax.xaxis,ax.yaxis):
            low,high=sorted(axis.get_view_interval())
            texts.extend(t.label1 for t in axis.get_major_ticks() if low<=t.get_loc()<=high)
    for legend in fig.legends:texts.extend(legend.get_texts())
    boxes=[]
    for t in texts:
        if not t.get_visible() or not t.get_text():continue
        box=t.get_window_extent(renderer)
        assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.x1+1 and box.y1<=fig.bbox.y1+1,(path.name,'clipped text',t.get_text())
        boxes.append((t.get_text(),box))
    for i,(name,box) in enumerate(boxes):
        for other,b in boxes[i+1:]:
            # A real area intersection, not touching edges.
            if min(box.x1,b.x1)-max(box.x0,b.x0)>.5 and min(box.y1,b.y1)-max(box.y0,b.y0)>.5:
                raise AssertionError((path.name,'overlapping text',name,other))
    fig.savefig(path,format='svg',facecolor='white',bbox_inches=None)
    root=ET.parse(path).getroot()
    width=float(root.attrib['width'].removesuffix('pt'))*2.54/72
    height=float(root.attrib['height'].removesuffix('pt'))*2.54/72
    assert width<=18.00001,(path,width)
    layout.update(path=str(path),width_cm=width,height_cm=height,text_overlap_check='passed',text_clipping_check='passed')
    plt.close(fig)
    return layout


def make_main(subject,channels=None,repeat_only=False,y_limits=None,single_condition=None):
    channels=subject['anatomy'].name.tolist() if channels is None else channels
    data=subject['data'].loc[subject['data'].channel.isin(channels)]
    conditions=(single_condition,) if single_condition is not None else ('Repeat',) if repeat_only else subject['conditions']
    compact=repeat_only or single_condition is not None
    assert all(c in subject['conditions'] for c in conditions)
    data=data.loc[data.condition.isin(conditions)]
    colors={'Decision':'#2369BD','Passive':'#2369BD','Repeat':'#A9373B'}
    shown=pd.concat([data.loc[data.phase.eq(p)&data.time.between(*w)] for p,w in WINDOWS.items()])
    lo=min(0.,float((shown['mean']-shown['sem']).min()))
    hi=max(0.,float((shown['mean']+shown['sem']).max()))
    span=hi-lo
    ylim=tuple(y_limits) if y_limits is not None else (lo-.05*span,hi+.07*span)
    rows=len(channels);height=3.0+3.0*rows
    fig=plt.figure(figsize=(18*CM,height*CM))
    outer=fig.add_gridspec(rows,2,width_ratios=[1.15,4.1],left=.025,right=.985,
        top=1-.9/height,bottom=2.05/height,wspace=.25,hspace=.70)
    for row,channel in enumerate(channels):
        draw_surface(fig.add_subplot(outer[row,0]),subject,channel,
                     zoom=1.3 if compact else 1.,boundary_note=not compact)
        times=outer[row,1].subgridspec(2,3,height_ratios=[1,.09],width_ratios=[1.1,1,2],hspace=.04,wspace=.18)
        for col,(phase,window) in enumerate(WINDOWS.items()):
            ax=fig.add_subplot(times[0,col])
            sigax=fig.add_subplot(times[1,col],sharex=ax)
            for ci,condition in enumerate(conditions):
                d=data.loc[data.channel.eq(channel)&data.phase.eq(phase)&data.condition.eq(condition)]
                keep=d.loc[d.time.between(*window)]
                ax.plot(keep.time,keep['mean'],c=colors[condition],lw=1)
                ax.fill_between(keep.time.to_numpy(),(keep['mean']-keep['sem']).to_numpy(),
                    (keep['mean']+keep['sem']).to_numpy(),color=colors[condition],alpha=.16,lw=0)
                bar_y=.5 if compact else .75-.5*ci
                segments=[[(max(a,window[0]),bar_y),(min(b,window[1]),bar_y)]
                          for a,b in intervals(d.significant,d.time.to_numpy()) if a<window[1] and b>window[0]]
                if segments:sigax.add_collection(LineCollection(segments,colors=colors[condition],linewidths=1.7))
            ax.axhline(0,c='.55',ls='--',lw=.5);ax.axvline(0,c='.55',ls='--',lw=.5)
            ax.set(xlim=window,ylim=ylim,title=phase+('†' if subject['subject']=='D0149' and phase=='Response' and not repeat_only else ''))
            ax.tick_params(axis='x',bottom=False,labelbottom=False)
            ax.spines[['top','right','bottom']].set_visible(False)
            if col==0:ax.set_ylabel('HGA (z)',labelpad=3)
            else:ax.spines['left'].set_visible(False);ax.tick_params(axis='y',left=False,labelleft=False)
            sigax.set(ylim=(0,1),yticks=[],xlabel='Time (s)')
            sigax.set_xticks([-.5,0,.5] if phase=='Stimulus' else [0,.5,1] if phase=='Delay' else [-.5,0,.5,1,1.5])
            sigax.xaxis.labelpad=2
            sigax.spines[['top','left','right']].set_visible(False)
    single=len(channels)==1
    contact=channels[0].split('_',1)[1]
    roi=subject['anatomy'].set_index('name').loc[channels[0],'roi']
    title=('D0149 · Picture Naming · Sound · RTAS5–6' if subject['subject']=='D0149'
           else 'D0153 · Lexical Delay · '+contact+' · '+roi+(' boundary' if roi=='IFG–INS' else '') if single
           else 'D0153 · Lexical Delay · Insula and IFG–INS boundary')
    fig.suptitle(title+(' · '+conditions[0] if compact else ''),fontsize=8,y=1-.10/height)
    handles=[Line2D([0],[0],c=colors[c],lw=1.3,label=c) for c in conditions]
    handles += [Line2D([0],[0],marker='+',c=CENTER,ls='none',label='Virtual bipolar midpoint')]
    extra=('†Passive Response: estimated alignment; Repeat: speech onset.' if subject['subject']=='D0149'
           else '*RFMI8–9: IFG–INS boundary channel.' if not single else '')
    if not compact:
        fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.61/height),fontsize=6)
        fig.text(.5,.10/height,'Native midpoint, no snapping; pale gold / blue-gray: AIC / PIC. Mean ± SEM; bars: significance vs baseline.'
            +('\n'+extra if extra else ''),ha='center',va='bottom',fontsize=5.5,linespacing=1.3)
    stem=subject['subject']+('_PictureNaming_sound_RTAS5-6' if subject['subject']=='D0149'
          else '_'+contact+'_LexicalDelay' if single else '_LexicalDelay')
    if compact:stem+='_'+conditions[0]
    path=FIG/(stem+'_spatiotemporal.svg')
    layout=save_checked(fig,path,dict(channels=channels,conditions=list(conditions),ylim=list(ylim),
        brain_display_zoom=1.3 if compact else 1.,bottom_legend=not compact,
        envelope_min=lo,envelope_max=hi,significance_location='separate strip below waveform; no extra y-range'))
    return path,layout

def make_mri_qc(subjects):
    # Same three pairs and midpoint slice convention as the existing CCEP QC.
    targets=[('D0149','D0149_RTAS5-6'),('D0153','D0153_RFMI8-9'),('D0153','D0153_RFMI9-10')]
    fig,axes=plt.subplots(3,3,figsize=(18*CM,16*CM))
    offsets=[]
    for row,(sid,channel) in enumerate(targets):
        s=subjects[sid];aparc=s['aparc']
        # aparc[1] supplies segmentation geometry, not anatomical MRI intensities.
        orig=nib.load(RECON/s['recon']/'mri/orig.mgz')
        assert orig.shape[:3]==aparc[0].shape
        np.testing.assert_allclose(orig.header.get_vox2ras_tkr(),aparc[1].header.get_vox2ras_tkr())
        tkr=orig.header.get_vox2ras_tkr()
        orientation=nib.orientations.ornt_transform(nib.orientations.io_orientation(tkr),nib.orientations.axcodes2ornt(('R','A','S')))
        affine=tkr@nib.orientations.inv_ornt_aff(orientation,orig.shape[:3])
        mri=nib.orientations.apply_orientation(np.asarray(orig.dataobj),orientation)
        seg=nib.orientations.apply_orientation(aparc[0],orientation)
        ins_ids=[int(k) for k,v in aparc[2].items() if v in INSULA_LABELS]
        ifg_ids=[int(k) for k,v in aparc[2].items() if v=='ctx_rh_G_front_inf-Opercular']
        assert len(ins_ids)==5 and len(ifg_ids)==1
        pts=s['points'].query('channel == @channel');xyz=pts[['x','y','z']].to_numpy();mid=xyz[1]
        vox=np.rint(nib.affines.apply_affine(np.linalg.inv(affine),mid)).astype(int)
        vmax=float(np.percentile(mri[mri>0],99))
        for ax,(fixed,h,v,view) in zip(axes[row],[(0,1,2,'Sagittal'),(1,0,2,'Coronal'),(2,0,1,'Axial')]):
            plane=np.take(mri,vox[fixed],axis=fixed).T;labels=np.take(seg,vox[fixed],axis=fixed).T
            xs=affine[h,h]*np.arange(mri.shape[h])+affine[h,3];ys=affine[v,v]*np.arange(mri.shape[v])+affine[v,3]
            dx,dy=affine[h,h],affine[v,v]
            ax.imshow(plane,origin='lower',cmap='gray',vmin=0,vmax=vmax,extent=(xs[0]-dx/2,xs[-1]+dx/2,ys[0]-dy/2,ys[-1]+dy/2))
            for ids,color in [(ins_ids,'#f6cd59'),(ifg_ids,'#77d9ac')]:
                mask=np.isin(labels,ids)
                if mask.any():ax.contour(xs,ys,mask.astype(float),levels=[.5],colors=[color],linewidths=.8)
            slice_pos=affine[fixed,fixed]*vox[fixed]+affine[fixed,3]
            for idx,color,marker,offset,ha in [(1,CENTER,'+',(0,9),'center')]:
                ax.plot(xyz[idx,h],xyz[idx,v],marker=marker,mfc='none',mec=color,ms=5,mew=1.1,ls='none')
                text=ax.annotate(pts.iloc[idx].contact,(xyz[idx,h],xyz[idx,v]),xytext=offset,textcoords='offset points',color=color,fontsize=6,ha=ha)
                text.set_path_effects([patheffects.withStroke(linewidth=1.5,foreground='black')])
            ax.set(xlim=(mid[h]-18,mid[h]+18),ylim=(mid[v]-18,mid[v]+18),xticks=[],yticks=[])
            ax.set_title(s['recon']+' '+channel.split('_',1)[1]+' · '+view,fontsize=7)
            delta=xyz[[0,2],fixed]-slice_pos
            ax.set_xlabel(f'Midpoint plane: {slice_pos:.1f} mm',fontsize=5.5)
            offsets.append(dict(subject=sid,channel=channel,view=view,contact1_offset_mm=float(delta[0]),contact2_offset_mm=float(delta[1])))
    fig.suptitle('Native MRI check · virtual bipolar midpoints',fontsize=9,y=.98)
    fig.legend(handles=[Line2D([0],[0],c='#c89d2c',label='Destrieux insula'),Line2D([0],[0],c='#409778',label='IFG opercular'),
        Line2D([0],[0],marker='+',c=CENTER,ls='none',label='Bipolar midpoint')],
        loc='lower center',bbox_to_anchor=(.5,.055),ncol=3,frameon=False,fontsize=6)
    fig.text(.5,.02,'Only virtual bipolar midpoints are shown. Slices pass through the nearest voxel to each midpoint.\n'
        'Surface projection alone does not establish tissue membership.',ha='center',fontsize=6)
    fig.subplots_adjust(left=.03,right=.97,top=.91,bottom=.17,hspace=.65,wspace=.12)
    path=FIG/'D0149_D0153_native_MRI_pair_QC.svg'
    layout=save_checked(fig,path,dict(spatial_markers='virtual midpoint only'))
    pd.DataFrame(offsets).to_csv(OUT/'mri_slice_offsets.csv',index=False)
    return path,layout

def main():
    OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    for path in (Path(sys.prefix)/'fonts').glob('arial*.ttf'):font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({'font.family':'Arial','font.size':7,'svg.fonttype':'none','axes.linewidth':.75,
        'xtick.major.size':2,'ytick.major.size':2,'xtick.major.width':.75,'ytick.major.width':.75})
    pv.OFF_SCREEN=True
    pipeline=load_pipeline(UPSTREAM)
    subjects={sid:load_subject(sid,pipeline) for sid in ('D0153','D0149')}
    outputs=[make_main(subjects['D0153'],['D0153_RFMI9-10']),make_main(subjects['D0149']),
             make_main(subjects['D0153']),make_mri_qc(subjects),
             make_main(subjects['D0153'],['D0153_RFMI9-10'],repeat_only=True,y_limits=(-.2,.45)),
             make_main(subjects['D0149'],['D0149_RTAS5-6'],repeat_only=True),
             make_main(subjects['D0153'],['D0153_RFMI8-9'],repeat_only=True,y_limits=(-.2,.45))]
    paths=[p for p,layout in outputs]
    layouts=[layout for p,layout in outputs]
    (OUT/'layout_validation.json').write_text(json.dumps(layouts,indent=2))
    points=pd.concat([s['points'] for s in subjects.values()],ignore_index=True)
    points.to_csv(OUT/'native_points_and_labels.csv',index=False)
    assert len(points)==15
    mixed=subjects['D0153']['anatomy'].query("name == 'D0153_RFMI8-9'")
    assert mixed.roi.iloc[0]=='IFG–INS' and bool(mixed.mix.iloc[0])
    import xml.etree.ElementTree as ET
    for path in paths:
        assert path.stat().st_size>1000 and ET.parse(path).getroot().tag.endswith('svg')
    inputs=[record for s in subjects.values() for record in s['inputs']]
    assert all(sha(record['path'])==record['sha256'] for record in inputs)
    report=dict(status='passed',completed_utc=datetime.now(timezone.utc).isoformat(),
        subjects=list(subjects),d0153_channels=subjects['D0153']['anatomy'].name.tolist(),
        d0149_display_modality='sound',d0149_conditions=list(subjects['D0149']['conditions']),
        boundary_channel='D0153_RFMI8-9: IFG–INS, mixed; explicitly included by user',
        spatial_method='Native right lateral orthographic view, exact SHI-100 camera and surface helper; original tkRAS contacts and computed midpoints, no snapping.',
        surface_caveat='Hammers nearest GM fill up to 8 mm is for surface display only, never for contact relabeling.',
        mri_method='Existing CCEP midpoint orthogonal-slice convention; Destrieux insula and IFG boundaries added for ROI interpretation.',
        source_helpers=['notebooks/shi100_insula_contacts.ipynb','scripts/qc_ccep_insula_locations.py'],
        displayed_spatial_markers='virtual bipolar midpoint only',layouts=layouts,inputs=inputs,figures=[dict(path=str(p),sha256=sha(p)) for p in paths])
    (OUT/'validation.json').write_text(json.dumps(report,indent=2))
    print('Validated spatial/temporal panels:',*[str(p) for p in paths],sep='\n',flush=True)
    return paths,points

if __name__=='__main__':
    main()
