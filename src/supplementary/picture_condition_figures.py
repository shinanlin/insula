"""The two author-selected PN Stimulus baseline-response displays."""
from pathlib import Path
import json
import re
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib_venn import venn2,venn2_circles
import numpy as np
import pandas as pd

from scripts.refresh_picture_condition_sets import ROOT,OUT,IMG
from src.supplementary import lexical_condition_figures as reference

COLORS={'Sound only':'#2166AC','Image only':'#D95F02','Both':'#762A83'}
SAMPLING_COLORS={'Sampled, n.s.':'#BDBDBD','Significant, Sound or Image':'#762A83'}
CM=1/2.54


def save(fig,name):
    path=IMG/name
    fig.savefig(path,bbox_inches='tight',pad_inches=.03)
    plt.close(fig)
    return path


def load():
    plt.rcParams.update({'font.family':'Arial','font.size':7,'svg.fonttype':'none'})
    table=pd.read_csv(OUT/'electrodes.csv')
    assert len(table)==128 and table.subject.nunique()==27
    assert table.electrode.is_unique and not table.native_qc_bad.any()
    responsive=table.sound_responsive|table.image_responsive
    np.testing.assert_array_equal(responsive,table.baseline_class.ne('Neither'))
    assert int(responsive.sum())==49
    counts=pd.read_csv(OUT/'set_counts.csv')
    counts=counts.loc[counts['set'].isin(['Sound only','Both','Image only','Neither'])].copy()
    assert counts.set_index('set').n_electrodes.to_dict()=={'Sound only':25,'Both':13,'Image only':11,'Neither':79}
    return table,counts


def venn(table):
    sound=set(table.loc[table.sound_responsive,'electrode'])
    image=set(table.loc[table.image_responsive,'electrode'])
    counts=(len(sound-image),len(image-sound),len(sound&image))
    assert counts==(25,11,13)
    fig,ax=plt.subplots(figsize=(7.2*CM,5.4*CM))
    fig.subplots_adjust(left=.08,right=.92,bottom=.20,top=.80)
    v=venn2(subsets=counts,set_labels=(f'Sound\n(n={len(sound)})',f'Image\n(n={len(image)})'),ax=ax)
    for key,color in zip(('10','01','11'),(COLORS['Sound only'],COLORS['Image only'],COLORS['Both'])):
        patch=v.get_patch_by_id(key)
        if patch is not None:patch.set_facecolor(color);patch.set_alpha(.40)
    for circle,color in zip(venn2_circles(subsets=counts,ax=ax,lw=.8),(COLORS['Sound only'],COLORS['Image only'])):
        circle.set_edgecolor(color)
    for label in v.subset_labels:
        if label is not None:label.set_fontsize(8)
    for label,color in zip(v.set_labels,(COLORS['Sound only'],COLORS['Image only'])):
        if label is not None:label.set_fontsize(7);label.set_color(color)
    fig.suptitle('Picture Naming · Stimulus 0–1 s',fontsize=8,y=.98)
    fig.text(.5,.025,'128 electrodes / 27 participants\nNeither detected: 79',ha='center',fontsize=6.5)
    return save(fig,'fig3_picture_stimulus_0_1_venn.svg')


def brains(table):
    namespace=reference.brain_renderer()
    paths={};display_proofs={}
    responsive=table.sound_responsive|table.image_responsive
    for name in ['sampling_response','baseline_classes']:
        if name=='sampling_response':
            points=table.copy()
            points['responsive']=responsive
            # Draw background sampling sites first, leaving responses visible.
            points=points.sort_values('responsive',kind='stable')
            points['display_group']=np.where(points.responsive,'Significant, Sound or Image','Sampled, n.s.')
            colors=SAMPLING_COLORS
            title='Picture Naming · Stimulus 0–1 s\n128 sampled sites / 27 participants'
            assert len(points)==128
        else:
            points=table.loc[responsive].copy()
            points['display_group']=points.baseline_class
            colors=COLORS
            title='Picture Naming · Stimulus 0–1 s'
            assert len(points)==49
        fig,audit=namespace['brain_pair'](points,list(colors),colors,title)
        paths[name]=save(fig,f'fig3_picture_stimulus_{name}_brain.svg')
        columns=['electrode','subject','hemisphere','display_group','pixel_x','pixel_y','surface_residual_mm','display_lift_mm']
        frame=pd.DataFrame(audit,columns=columns)
        assert set(frame.electrode)==set(points.electrode)
        frame.to_csv(OUT/f'{name}_display_coordinates.csv',index=False)
        display_proofs[name]=dict(n_electrodes=len(points),n_participants=points.subject.nunique(),
            n_left=int(points.x.lt(0).sum()),n_right=int(points.x.gt(0).sum()),
            groups=points.display_group.value_counts().to_dict(),image=str(paths[name]),
            selected_identities=points.electrode.tolist())
    proof=dict(renderer=namespace['renderer_proof'],original_brain_pair_reused_verbatim=True,
        marker_size=24.,display_lift_mm=4.,original_camera_and_hemisphere_spacing=True,
        analytic_coordinates_changed=False,window=[0,1],panels=display_proofs,
        sampling_gray_meaning='Eligible sampled sites with neither native baseline test detecting a response',
        significant_union='Sound OR Image versus native baseline; 49 sites',
        display_scope='Two baseline-response steps requested by the author')
    (OUT/'brain_display_provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
    return paths


def compose(panels,name,width,height):
    ns='http://www.w3.org/2000/svg';xl='http://www.w3.org/1999/xlink'
    ET.register_namespace('',ns);ET.register_namespace('xlink',xl)
    root=ET.Element(f'{{{ns}}}svg',{'width':f'{width}pt','height':f'{height}pt','viewBox':f'0 0 {width} {height}'})
    ET.SubElement(root,f'{{{ns}}}rect',{'width':str(width),'height':str(height),'fill':'white'})
    for i,(path,x,y,w,h) in enumerate(panels):
        source=ET.fromstring(Path(path).read_text())
        identifiers={e.attrib['id']:f'pn{i}_{e.attrib["id"]}' for e in source.iter() if 'id' in e.attrib}
        for e in source.iter():
            if 'id' in e.attrib:e.set('id',identifiers[e.attrib['id']])
            for k,v in list(e.attrib.items()):
                v=re.sub(r'url\(#([^)]*)\)',lambda m:f'url(#{identifiers.get(m.group(1),m.group(1))})',v)
                if k.endswith('href') and v.startswith('#'):v='#'+identifiers.get(v[1:],v[1:])
                e.set(k,v)
            href=e.attrib.get(f'{{{xl}}}href')
            if href and href.startswith('data:'):
                header,payload=href.split(',',1);href=header+','+''.join(payload.split())
                e.set(f'{{{xl}}}href',href);e.set('href',href)
        source.set('x',str(x));source.set('y',str(y));source.set('width',str(w));source.set('height',str(h))
        source.set('preserveAspectRatio','xMidYMid meet');root.append(source)
    path=IMG/name
    ET.ElementTree(root).write(path,encoding='unicode',xml_declaration=True)
    return path


def assemble(venn_path,paths):
    condition_pair=compose([(venn_path,0.,0.,265.,230.),
        (paths['baseline_classes'],280.,18.,370.,210.)],
        'fig3_picture_stimulus_0_1_condition_sets.svg',650.,230.)
    sheet=compose([(paths['sampling_response'],140.,0.,370.,230.),
        (venn_path,0.,250.,265.,230.),(paths['baseline_classes'],280.,268.,370.,210.)],
        'fig3_picture_stimulus_0_1_summary.svg',650.,480.)
    return condition_pair,sheet
