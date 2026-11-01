"""Author-approved three-region condition-difference schematics and maps.

The center is *no detected difference*, not a set intersection. PN's one
bidirectional site is a fourth, separately displayed category. No tests rerun.
"""
from pathlib import Path
import json
import os
import sys
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from matplotlib_venn import venn2,venn2_circles
from matplotlib_venn.layout.venn2 import DefaultLayoutAlgorithm
from matplotlib_venn._math import circle_intersection_area
import numpy as np
import pandas as pd

from src.figure3 import CONDITION_SPECS as SPECS
from src.main_analysis import FUNCTIONAL_COMPONENTS, assigned_functional_sites

ROOT = Path('/cwork/ns458/insula-paper')
OUT = ROOT/'results/fig3/regions'
IMG = ROOT/'img/fig3'
NOTEBOOK = ROOT/'vizpub/fig3.ipynb'
COMPONENTS = FUNCTIONAL_COMPONENTS
COLORS = dict(sensory='#2369BD', sustain='#C4A35A', motor='#A9373B')
LABELS = dict(sensory='Sensory', sustain='Sustained', motor='Motor')


def load_groups(key):
    """Verify the saved responsive-pool partition without changing any source."""
    from scripts.refresh_lexical_condition_sets import sha, write
    spec = SPECS[key]
    folder = ROOT/'results/supplementary'/spec['source']
    from src.supplementary.assigned_condition_cohorts import load_cohort
    table, all_anatomical = load_cohort(key)
    a,b = spec['conditions']
    first,second = f'{a}_gt_{b}',f'{b}_gt_{a}'
    pool = table.loc[table[f'{a}_responsive'] | table[f'{b}_responsive']].copy()
    assert len(pool) == {'picture':44,'lexical':71}[key] and pool.electrode.is_unique
    assert pool.eligible.all() and pool.paired_statistics_complete.all() and not pool.native_qc_bad.any()
    old = ROOT/'results/supplementary/fig3_directional_venn_draft_2026-10-28'
    saved = pd.read_csv(old/key/'responsive_pool_electrodes.csv')
    assert set(assigned_functional_sites(saved).electrode) == set(pool.electrode)
    # Verify every original source protected during the previous direct-test audit.
    guard = json.loads((old/'protected_existing_files.json').read_text())
    for name in ('electrodes.csv','timeperm.csv.gz'):
        assert sha(folder/name) == guard[str(folder/name)]
    masks = dict(first_only=pool[first] & ~pool[second],
        no_detected_difference=~pool[first] & ~pool[second],
        second_only=pool[second] & ~pool[first],
        bidirectional=pool[first] & pool[second])
    expected = {'picture':(18,4,1,21),'lexical':(27,1,0,43)}[key]
    assert tuple(int(masks[g].sum()) for g in ('first_only','second_only','bidirectional','no_detected_difference')) == expected
    assert np.vstack([s.to_numpy() for s in masks.values()]).sum(axis=0).tolist() == [1]*len(pool)
    groups = {g:pool.loc[m].copy() for g,m in masks.items()}
    titles = dict(first_only=spec['names'][0]+' only',
        no_detected_difference='No detected difference',second_only=spec['names'][1]+' only',
        bidirectional='Bidirectional'+(' · D0080 LPI2–3' if key=='picture' else ''))
    target = OUT/key
    target.mkdir(parents=True,exist_ok=True)
    rows = []
    for name,points in groups.items():
        reference = pd.read_csv(old/key/f'{name}_electrodes.csv')
        assert set(assigned_functional_sites(reference).electrode) == set(points.electrode)
        points.to_csv(target/f'{name}_electrodes.csv',index=False)
        counts = points.functional_cluster.fillna('Unassigned').value_counts()
        rows.append(dict(group=name,label=titles[name],n_electrodes=len(points),n_participants=points.subject.nunique(),
            **{c:int(counts.get(c,0)) for c in COMPONENTS}))
    if key=='picture':
        assert groups['bidirectional'].channel.tolist() == ['D0080_LPI2-3']
    else: assert groups['bidirectional'].empty
    counts = pd.DataFrame(rows)
    counts.to_csv(target/'group_counts.csv',index=False)
    pool.to_csv(target/'responsive_pool_electrodes.csv',index=False)
    write(target/'selection_provenance.json',dict(passed=True,anatomical_inventory_n=len(all_anatomical),parent_n=len(table),
        parent_subjects=table.subject.nunique(),pool_n=len(pool),pool_subjects=pool.subject.nunique(),
        groups=rows,exclusive_partition=True,previous_partition_matched=True,
        statistical_parameters=dict(p_thresh=.05,p_cluster=.05,n_perm=2000,tails=2,seed=42),
        source_hashes={str(folder/n):sha(folder/n) for n in ('electrodes.csv','timeperm.csv.gz')},
        inference='Reused original per-electrode two-sided time_perm_cluster and saved directional masks; no new tests',
        center_meaning='Neither significant direction detected; not equivalence or both-baseline responsiveness',
        diagram='Three-category classification; areas proportional to counts; center is no detected difference, not a literal response-set intersection'))
    return pool,groups,counts,titles


def _group_editable_regions(path,region_groups):
    """Group region fills and text without changing the normal Venn appearance."""
    namespaces={'':'http://www.w3.org/2000/svg',
        'xlink':'http://www.w3.org/1999/xlink',
        'dc':'http://purl.org/dc/elements/1.1/',
        'cc':'http://creativecommons.org/ns#',
        'rdf':'http://www.w3.org/1999/02/22-rdf-syntax-ns#'}
    for prefix,uri in namespaces.items():ET.register_namespace(prefix,uri)
    tree=ET.parse(path);root=tree.getroot();svg='{'+namespaces['']+'}'
    parents={child:parent for parent in root.iter() for child in parent}
    elements={element.get('id'):element for element in root.iter() if element.get('id')}
    figure=elements['figure_1']
    for group_id,spec in region_groups.items():
        group=ET.SubElement(figure,svg+'g',id=group_id)
        ET.SubElement(group,svg+'title').text=spec['label']
        for member_id in spec['members']:
            element=elements[member_id]
            parents[element].remove(element)
            group.append(element)
            for child in element.iter():child.attrib.pop('clip-path',None)
            if member_id.endswith('_shape'):
                shape=element.find(svg+'path')
                # matplotlib-venn returns to the first vertex but omits the SVG
                # close command. Close explicitly for Illustrator path editing.
                if not shape.get('d').strip().lower().endswith('z'):
                    shape.set('d',shape.get('d').rstrip()+' z')
    # Keep both ordinary circle outlines above the fills, as in the original
    # matplotlib-venn drawing. The author will split them locally if desired.
    outlines=ET.SubElement(figure,svg+'g',id='circle_outlines')
    for number in (1,2):
        element=elements[f'venn_circle_{number}_outline']
        parents[element].remove(element)
        outlines.append(element)
    # Matplotlib's axes group is empty after all diagram artists are regrouped.
    for element in list(figure):
        if element.get('id')=='axes_1' and not len(element):figure.remove(element)
    ET.indent(tree,space=' ')
    tree.write(path,encoding='utf-8',xml_declaration=True)


def diagram(key,pool,groups):
    """Count-proportional, normally overlapping circles with editable SVG fills."""
    spec = SPECS[key]
    fig,ax = plt.subplots(figsize=(10/2.54,7.3/2.54))
    fig.subplots_adjust(left=.025,right=.975,bottom=.24,top=.79)
    subsets=tuple(len(groups[g]) for g in ('first_only','second_only','no_detected_difference'))
    layout=DefaultLayoutAlgorithm(normalize_to=sum(subsets)/78.)
    v=venn2(subsets=subsets,set_labels=None,ax=ax,layout_algorithm=layout)
    circles=venn2_circles(subsets=subsets,ax=ax,layout_algorithm=layout,linewidth=1)
    names=dict(zip(('10','11','01'),(SPECS[key]['conditions'][0]+'_gt_'+SPECS[key]['conditions'][1],
        'no_detected_difference',SPECS[key]['conditions'][1]+'_gt_'+SPECS[key]['conditions'][0])))
    region_groups={}
    for region,label in (('10',spec['names'][0]+' only'),
        ('11','No detected difference'),('01',spec['names'][1]+' only')):
        group_id='region_'+names[region]
        region_groups[group_id]=dict(label=label,offset_mm=0,
            members=[group_id+'_shape',group_id+'_count',group_id+'_label'])
    for region,color in zip(('10','01','11'),('#2166AC','#D95F02','#E8E8E8')):
        patch=v.get_patch_by_id(region)
        group_id='region_'+names[region]
        patch.set_alpha(None);patch.set_facecolor(to_rgba(color,1 if region=='11' else .22))
        patch.set_edgecolor('none')
        patch.set_clip_on(False);patch.set_gid(group_id+'_shape')
        label=v.get_label_by_id(region);label.set_fontsize(12);label.set_gid(group_id+'_count')
    for number,(circle,color) in enumerate(zip(circles,('#2166AC','#D95F02')),start=1):
        circle.set_edgecolor(color);circle.set_clip_on(False)
        circle.set_gid(f'venn_circle_{number}_outline')
    if subsets[1]==1:
        label=v.get_label_by_id('01');location=label.get_position();label.set_visible(False)
        group_id='region_'+names['01']
        annotation=ax.annotate('1',xy=location,xytext=(.77,-.29),ha='center',va='center',fontsize=12,
            arrowprops=dict(arrowstyle='-',color='#555555',linewidth=.65))
        annotation.set_gid(group_id+'_count');annotation.arrow_patch.set_gid(group_id+'_leader')
        region_groups[group_id]['members'].insert(1,group_id+'_leader')
    ax.set(xlim=(-.90,.90),ylim=(-.60,.60),aspect='equal');ax.axis('off')
    fig.suptitle(spec['title'],fontsize=9,y=.98)
    fig.text(.5,.88,f'{len(pool)} responsive electrodes / {pool.subject.nunique()} participants',ha='center',fontsize=7)
    for region,x,label,color in (('10',.205,spec['names'][0]+' only','#2166AC'),
        ('11',.5,'No detected\ndifference','#444444'),('01',.795,spec['names'][1]+' only','#D95F02')):
        text=fig.text(x,.815,label,ha='center',va='center',fontsize=6.7,color=color)
        text.set_gid('region_'+names[region]+'_label')
    if len(groups['bidirectional']):
        fig.add_artist(Rectangle((.08,.12),.84,.105,transform=fig.transFigure,
            fill=False,linewidth=.65,edgecolor='#555555'))
        fig.text(.5,.173,'Bidirectional: D0080 LPI2–3 · n=1 (shown separately)',ha='center',va='center',fontsize=6.7)
    else:
        fig.text(.5,.17,'All three categories are mutually exclusive',ha='center',fontsize=6.5)
    fig.text(.5,.06,'Areas proportional to electrode counts · same scale across panels\n'
        'Center: no detected difference, not statistical equivalence',ha='center',fontsize=6.2)
    fig.canvas.draw()
    first_area,second_area=[np.pi*c.get_radius()**2 for c in circles]
    distance=np.linalg.norm(np.array(circles[0].center)-np.array(circles[1].center))
    shared_area=circle_intersection_area(circles[0].get_radius(),circles[1].get_radius(),distance)
    area_counts=np.array([first_area-shared_area,second_area-shared_area,shared_area])*78
    np.testing.assert_allclose(area_counts,subsets,atol=1e-3,rtol=0)
    scale=float(np.linalg.norm(ax.transData.transform([1,0])-ax.transData.transform([0,0])))
    from scripts.refresh_lexical_condition_sets import write
    write(OUT/key/'diagram_area_provenance.json',dict(passed=True,
        region_order=['first_only','second_only','no_detected_difference'],
        target_counts=subsets,recovered_counts_from_circle_geometry=area_counts.tolist(),
        circle_totals=[subsets[0]+subsets[2],subsets[1]+subsets[2]],
        radii_data_units=[c.get_radius() for c in circles],
        common_area_per_electrode_data_units=1/78,
        pixels_per_data_unit=scale,center_means_no_detected_difference=True,
        editable_svg_groups=region_groups,independent_closed_region_borders=False,
        conventional_overlapping_circle_outlines=True,no_display_offsets=True,
        original_region_shapes_and_areas_preserved=True))
    path = IMG/f'fig3_{key}_condition_difference_schematic.svg'
    # Identical uncropped canvases keep the common physical area scale visible
    # when the two independent SVGs are placed at the same displayed width.
    fig.savefig(path,format='svg',bbox_inches=None);plt.close(fig)
    _group_editable_regions(path,region_groups)
    return path


def maps(key,groups,titles):
    from src.figure3_brain import brain_pair
    from src.supplementary.selected_brain_contours import _figure2_contour_helpers
    from scripts.refresh_lexical_condition_sets import sha,write
    _,contour_proof = _figure2_contour_helpers(ROOT)
    assert contour_proof['minimum_sites_per_component_hemisphere']==3
    reference_path = ROOT/'results/supplementary'/SPECS[key]['source']/(
        'baseline_classes_display_coordinates.csv' if key=='picture' else 'lexical_brain_display_coordinates.csv')
    reference = pd.read_csv(reference_path)
    assert reference.electrode.is_unique
    paths = {}
    for name,points in groups.items():
        if points.empty: continue
        points = points.copy()
        assert points.functional_cluster.isin(COMPONENTS).all()
        assert np.isfinite(points[['x','y','z']]).all().all() and points.x.ne(0).all()
        points['display_group']=points.functional_cluster
        fig,coordinates,rings = brain_pair(points,list(COMPONENTS),COLORS,LABELS,
            title=f"{titles[name]} · n={len(points)}\n{SPECS[key]['title']}")
        path = IMG/f'fig3_{key}_{name}_functional_brain.svg'
        fig.savefig(path,format='svg',bbox_inches='tight',pad_inches=.03);plt.close(fig)
        projection = pd.DataFrame(coordinates)
        matched = projection.merge(reference,on='electrode',suffixes=('_new','_ref'),validate='one_to_one')
        assert len(matched)==len(points)
        for col in ('pixel_x','pixel_y','surface_residual_mm','display_lift_mm'):
            np.testing.assert_allclose(matched[col+'_new'],matched[col+'_ref'],rtol=0,atol=1e-8)
        contours = pd.DataFrame(rings)
        assert contours.loc[contours.n_sites.lt(3),'n_contours'].eq(0).all()
        projection.to_csv(OUT/key/f'{name}_display_coordinates.csv',index=False)
        contours.to_csv(OUT/key/f'{name}_contours.csv',index=False)
        write(OUT/key/f'{name}_map_provenance.json',dict(passed=True,
            n_sites=len(points),n_subjects=points.subject.nunique(),
            hemispheres=dict(left=int(points.x.lt(0).sum()),right=int(points.x.gt(0).sum())),
            exact_existing_projection=True,point_size_pixels=24,display_lift_mm=4,
            palette=COLORS,contour_method=contour_proof,
            original_coordinates_and_functional_labels_unchanged=True,
            svg=str(path),svg_sha256=sha(path)))
        paths[name]=str(path)
        print(f'RENDERED {key} {name}: {len(points)} sites',flush=True)
    return paths


def run():
    assert os.environ.get('SLURM_JOB_ID'),'Run via Slurm'
    from scripts.refresh_lexical_condition_sets import protected,sha,write
    OUT.mkdir(parents=True,exist_ok=True);IMG.mkdir(parents=True,exist_ok=True)
    paths=[]
    for folder in ('results/nmf','img/fig1','img/fig2',
        'results/supplementary/fig3_picture_stimulus_condition_sets',
        'results/supplementary/fig3_lexical_delay_condition_sets/full_cohort_2026-10-27',
        'results/supplementary/fig3_directional_venn_draft_2026-10-28',
        'img/supplementary/sound_image_test','img/supplementary/decision_repeat_test',
        'img/supplementary/fig3_directional_venn_draft'):
        paths += [p for p in (ROOT/folder).rglob('*') if p.is_file()]
    paths += list((ROOT/'vizpub').glob('fig[1-6].ipynb'))
    paths += [p for p in (ROOT/'vizpub/supplementary').glob('*.ipynb') if p!=NOTEBOOK]
    frozen={str(p):sha(p) for p in sorted(set(paths))}
    write(OUT/'protected_existing_files.json',frozen)
    for font in Path(sys.prefix,'fonts').glob('arial*.ttf'):font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Arial','font.size':7,'svg.fonttype':'none'})
    results={}
    for key in SPECS:
        pool,groups,counts,titles=load_groups(key)
        venn=diagram(key,pool,groups)
        brain_paths=maps(key,groups,titles)
        results[key]=dict(diagram=str(venn),brains=brain_paths,counts=counts.to_dict('records'),
            pool_n=len(pool),pool_subjects=pool.subject.nunique())
    assert all(sha(p)==h for p,h in frozen.items())
    write(OUT/'completion.json',dict(passed=True,job_id=os.environ['SLURM_JOB_ID'],
        existing_files_unchanged=len(frozen),fresh_main_and_source_guard=True,
        comparisons=results,no_new_tests=True,no_NMF_changes=True))
    return results
