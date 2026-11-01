"""Condition-colored Figure 3 Venns and insula maps, without component display."""
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET

import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles
import numpy as np
import pandas as pd

from src.supplementary.decision_repeat_maps import _figure1_local_insula_renderer
from src.supplementary.selected_brain_contours import _figure2_contour_helpers

ROOT=Path('/cwork/ns458/insula-paper')
OUT=ROOT/'results/supplementary/fig3_condition_response_maps_2026-10-23'
IMG=ROOT/'img/supplementary/fig3_condition_response_maps'
PREVIOUS=ROOT/'results/supplementary/fig3_response_characterization_2026-10-23'
COLORS=dict(first='#2166AC',second='#D95F02',third='#762A83')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);IMG.mkdir(parents=True,exist_ok=True)
    # The 0--.6-s waveform export is cropped. Read full native masks instead.
    picture=pd.read_csv(PREVIOUS/'sound_image_electrode_membership.csv')
    source=ROOT/'results/supplementary/sound_image_test/event_repair_2026-10-21/native_input_manifest.json'
    manifest=json.loads(source.read_text());checks=[]
    flags={}
    for subject,group in picture.groupby('subject'):
        channels=set(group.channel)
        for level in ('sound','image'):
            name=f'sub-{subject}_task-PictureNaming_proc-Stimulus_recording-{level}_desc-Repeat_highgamma.h5'
            repaired=[r for r in manifest['replacement_inputs'] if Path(r['stats']['path']).name==name]
            records=repaired or [r for r in manifest['input_files'] if Path(r['stats']['path']).name==name]
            records=[r for r in records if channels<=set(r['channels'])]
            assert records,(subject,level)
            record=max(records,key=lambda r:len(r['channels']))
            assert sha(record['stats']['path'])==record['statistics_sha256']
            with np.load(record['cache'],allow_pickle=False) as f:times=f['times'][:]
            keep=(times>=0)&(times<=1);assert int(keep.sum())==129
            with h5py.File(record['stats']['path'],'r') as f:
                names=[x.decode() if isinstance(x,bytes) else str(x) for x in f['ch_names'][:]]
                for r in group.itertuples():flags[(r.electrode,level)]=bool(f['mask'][names.index(r.channel)].astype(bool)[keep].any())
            checks.append(dict(subject=subject,condition=level,window_start=0,window_end=1,
                statistics_path=record['stats']['path'],statistics_sha256=record['statistics_sha256'],repaired_source=bool(repaired)))
    picture['sound']=picture.electrode.map(lambda e:flags[e,'sound'])
    picture['image']=picture.electrode.map(lambda e:flags[e,'image'])
    picture['display_group']=np.select([picture.sound & ~picture.image,~picture.sound & picture.image,picture.sound & picture.image],
        ['Sound only','Image only','Both'],default='Not shown')
    picture=picture[['subject','channel','electrode','x','y','z','sound','image','display_group']]
    assert len(picture)==81 and picture.subject.nunique()==24
    lexical=pd.read_csv(PREVIOUS/'decision_repeat_electrode_membership.csv')
    assert len(lexical)==123 and lexical.subject.nunique()==34
    tested=ROOT/'results/supplementary/decision_repeat_test/delay_electrode_selection'
    plan=json.loads((tested/'selection_manifest.json').read_text())
    for name,digest in plan['outputs'].items():assert sha(tested/name)==digest
    selected=pd.read_csv(tested/'selected_electrodes.csv');assert len(selected)==29
    assert set(selected.electrode)<=set(lexical.electrode)
    input_audit=pd.read_csv(tested/'tested_electrodes.csv') if (tested/'tested_electrodes.csv').exists() else None
    if input_audit is not None:assert set(input_audit.electrode)==set(lexical.electrode)
    lexical['decision']=lexical.response_a.astype(bool);lexical['repeat']=lexical.response_b.astype(bool)
    lexical['decision_gt_repeat']=lexical.electrode.isin(selected.electrode)
    lexical=lexical[['subject','channel','electrode','x','y','z','decision','repeat','decision_gt_repeat']]
    picture.to_csv(OUT/'picture_stimulus_0_1_electrodes.csv',index=False)
    lexical.to_csv(OUT/'lexical_delay_0_1_electrodes.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'picture_native_mask_sources.csv',index=False)
    sound=set(picture.loc[picture.sound,'electrode']);image=set(picture.loc[picture.image,'electrode'])
    decision=set(lexical.loc[lexical.decision,'electrode']);repeat=set(lexical.loc[lexical.repeat,'electrode']);difference=set(selected.electrode)
    rows=[]
    for comparison,data,sets in [('picture_stimulus',picture,{'Sound':sound,'Image':image,'Sound only':sound-image,'Image only':image-sound,'Both':sound&image}),
        ('lexical_delay',lexical,{'Decision':decision,'Repeat':repeat,'Decision > Repeat':difference})]:
        for name,sites in sets.items():
            sub=data.loc[data.electrode.isin(sites)]
            rows.append(dict(comparison=comparison,set=name,n_electrodes=len(sites),n_participants=sub.subject.nunique(),participants=';'.join(sorted(sub.subject.unique()))))
    counts=pd.DataFrame(rows);counts.to_csv(OUT/'set_counts.csv',index=False)
    regions=[]
    for a in (False,True):
        for b in (False,True):
            for c in (False,True):
                sub=lexical.loc[lexical.decision.eq(a)&lexical.repeat.eq(b)&lexical.decision_gt_repeat.eq(c)]
                regions.append(dict(region=f'{int(a)}{int(b)}{int(c)}',decision=a,repeat=b,decision_gt_repeat=c,n_electrodes=len(sub),n_participants=sub.subject.nunique()))
    pd.DataFrame(regions).to_csv(OUT/'lexical_venn_regions.csv',index=False)
    write_json(OUT/'specification.json',dict(picture_window=[0,1],picture_n_sites=81,picture_n_participants=24,
        lexical_window=[0,1],lexical_n_sites=123,lexical_n_participants=34,
        lexical_denominator='Intersection of existing baseline observations and the existing per-electrode condition-difference tested cohort; the 12 untested pre-gate sites are not labeled negative.',
        picture_colors={'Sound only':COLORS['first'],'Image only':COLORS['second'],'Both':COLORS['third']},
        lexical_colors={'Decision':COLORS['first'],'Repeat':COLORS['second'],'Decision > Repeat':COLORS['third']},
        no_component_display=True,new_statistics=False,nmf_changed=False,baseline_rule='Any existing native baseline mask hit in the stated window',
        difference_source=str(tested/'selected_electrodes.csv'),difference_sha256=sha(tested/'selected_electrodes.csv')))
    print(counts.to_string(index=False),flush=True)
    return picture,lexical,counts


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


def draw_picture(picture):
    sound=set(picture.loc[picture.sound,'electrode']);image=set(picture.loc[picture.image,'electrode'])
    cm=1/2.54
    fig,ax=plt.subplots(figsize=(7.2*cm,5.4*cm));fig.subplots_adjust(left=.08,right=.92,bottom=.20,top=.80)
    counts=(len(sound-image),len(image-sound),len(sound&image))
    diagram=venn2(subsets=counts,set_labels=(f'Sound\n(n={len(sound)})',f'Image\n(n={len(image)})'),ax=ax)
    for ident,color in [('10',COLORS['first']),('01',COLORS['second']),('11',COLORS['third'])]:
        patch=diagram.get_patch_by_id(ident)
        if patch is not None:patch.set_facecolor(color);patch.set_alpha(.40)
    for circle,color in zip(venn2_circles(subsets=counts,ax=ax,lw=.8),(COLORS['first'],COLORS['second'])):circle.set_edgecolor(color)
    common=diagram.get_patch_by_id('11')
    if common is not None:ax.add_patch(PathPatch(common.get_path(),transform=common.get_transform(),facecolor='none',edgecolor='white',lw=3,zorder=2.5))
    for label in diagram.subset_labels:
        if label is not None:label.set_fontsize(8)
    for label in diagram.set_labels:
        if label is not None:label.set_fontsize(7)
    fig.suptitle('Picture Naming · Stimulus 0–1 s',fontsize=8,y=.98)
    fig.text(.5,.025,f'Tested: {len(picture)} electrodes / {picture.subject.nunique()} participants\nNeither detected: {len(picture)-len(sound|image)}',ha='center',fontsize=6.5)
    venn=IMG/'fig3_picture_stimulus_0_1_venn.svg';fig.savefig(venn,bbox_inches='tight',pad_inches=.03);plt.close(fig)
    points=picture.loc[picture.display_group.ne('Not shown')].copy()
    groups=('Sound only','Image only','Both');colors=dict(zip(groups,(COLORS['first'],COLORS['second'],COLORS['third'])))
    fig,audit,rings=brain_pair(points,groups,colors,{g:g for g in groups},title='Picture Naming · Stimulus 0–1 s')
    brain=IMG/'fig3_picture_stimulus_0_1_brain.svg';fig.savefig(brain,bbox_inches='tight',pad_inches=.03);plt.close(fig)
    pd.DataFrame(audit).to_csv(OUT/'picture_brain_display_coordinates.csv',index=False)
    pd.DataFrame(rings).to_csv(OUT/'picture_brain_contours.csv',index=False)
    return venn,brain


def draw_lexical(lexical,*,reuse_brains=False):
    from matplotlib_venn.layout.venn3 import DefaultLayoutAlgorithm
    sets=[set(lexical.loc[lexical[c],'electrode']) for c in ('decision','repeat','decision_gt_repeat')]
    names=('Decision','Repeat','Decision > Repeat');palette=(COLORS['first'],COLORS['second'],COLORS['third'])
    cm=1/2.54
    fig,ax=plt.subplots(figsize=(7.2*cm,6.4*cm));fig.subplots_adjust(left=.05,right=.95,bottom=.22,top=.82)
    # Unequal-circle area fitting loses the real 1-site Repeat∩difference region.
    # Fixed geometry preserves all seven intersections; labels keep exact counts.
    layout=DefaultLayoutAlgorithm(fixed_subset_sizes=(1,1,1,1,1,1,1))
    diagram=venn3(subsets=sets,set_labels=[f'{n}\n(n={len(s)})' for n,s in zip(names,sets)],set_colors=palette,alpha=.20,ax=ax,layout_algorithm=layout)
    for circle,color in zip(venn3_circles(subsets=sets,ax=ax,lw=.8,layout_algorithm=layout),palette):circle.set_edgecolor(color)
    assert all(diagram.get_label_by_id(k) is not None for k in ('100','010','110','001','101','011','111'))
    assert sum(int(t.get_text()) for t in diagram.subset_labels)==len(set.union(*sets))
    for label in diagram.subset_labels:
        if label is not None:label.set_fontsize(7.5)
    for label,color in zip(diagram.set_labels,palette):
        if label is not None:label.set_fontsize(7);label.set_color(color)
    fig.suptitle('Lexical · Delay 0–1 s',fontsize=8,y=.99)
    fig.text(.5,.02,f'Tested: {len(lexical)} electrodes / {lexical.subject.nunique()} participants\nDecision/Repeat: relative to baseline\nSchematic areas; exact counts shown',ha='center',fontsize=6.3)
    venn=IMG/'fig3_lexical_delay_0_1_venn.svg';fig.savefig(venn,bbox_inches='tight',pad_inches=.03);plt.close(fig)
    paths=[];audits=[]
    for column,name,color in zip(('decision','repeat','decision_gt_repeat'),names,palette):
        path=IMG/f'fig3_lexical_delay_{column}_brain.svg'
        if reuse_brains:
            assert path.is_file();paths.append(path);continue
        points=lexical.loc[lexical[column]].copy();points['display_group']=name
        fig,audit,rings=brain_pair(points,[name],{name:color},{name:name},title=f'{name} · Delay 0–1 s')
        fig.savefig(path,bbox_inches='tight',pad_inches=.03);plt.close(fig)
        paths.append(path);audits.extend(audit)
    if not reuse_brains:pd.DataFrame(audits).to_csv(OUT/'lexical_brain_display_coordinates.csv',index=False)
    return venn,paths


def assemble(venn,brains,name):
    """Embed SVG panels without rasterizing labels or colliding clipping IDs."""
    ns='http://www.w3.org/2000/svg';ET.register_namespace('',ns)
    width,height=450.,max(185.,len(brains)*145.+10.)
    root=ET.Element(f'{{{ns}}}svg',{'width':f'{width}pt','height':f'{height}pt','viewBox':f'0 0 {width} {height}'})
    ET.SubElement(root,f'{{{ns}}}rect',{'width':str(width),'height':str(height),'fill':'white'})
    panels=[(venn,0.,(height-175.)/2,185.,175.)]+[(p,187.,(height-145.*len(brains))/2+i*145.,263.,145.) for i,p in enumerate(brains)]
    for i,(path,x,y,w,h) in enumerate(panels):
        source=ET.fromstring(Path(path).read_text())
        ids={e.attrib['id']:f'p{i}_{e.attrib["id"]}' for e in source.iter() if 'id' in e.attrib}
        for element in source.iter():
            if 'id' in element.attrib:element.attrib['id']=ids[element.attrib['id']]
            for key,value in list(element.attrib.items()):
                value=re.sub(r'url\(#([^)]*)\)',lambda m:f'url(#{ids.get(m[1],m[1])})',value)
                if key.endswith('href') and value.startswith('#'):value='#'+ids.get(value[1:],value[1:])
                element.attrib[key]=value
        source.set('x',str(x));source.set('y',str(y));source.set('width',str(w));source.set('height',str(h))
        source.set('preserveAspectRatio','xMidYMid meet');root.append(source)
    path=IMG/name;ET.ElementTree(root).write(path,encoding='unicode',xml_declaration=True)
    return path
