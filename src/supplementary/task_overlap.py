"""Four-task sampling sets and exact, mutually exclusive Venn counts."""
from itertools import product
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.main_analysis import TASKS, CONDITION, MODALITY, TASK_DISPLAY_NAMES

ABBREVIATIONS = dict(zip(TASKS,['PS','LD','PN','SR']))
# This task palette is separate from anatomical ROI / functional palettes.
TASK_COLORS = dict(zip(TASKS,['#0072B2','#E69F00','#009E73','#CC79A7']))
SCOPES = {'subjects':'Participants',
          'all_sites':'All recorded bipolar sites',
          'insula_sites':'Sampled insula bipolar sites'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_overlap_sets(root, output):
    root,output = Path(root),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    coord_path = root/'results/audit/sound_only_cohort_2026-10-17/coordinate_variants.csv'
    insula_path = root/'docs/evidence/sound_only_cohort_2026-10-17/sampled_sites_by_subject_task.csv'
    fig1_path = root/'results/fig1/figure_manifest.json'
    reference = json.loads(fig1_path.read_text())
    coords = pd.read_csv(coord_path)
    assert coords.description.eq(CONDITION).all() and coords.modality.eq(MODALITY).all()
    assert set(coords.task)==set(TASKS)
    assert coords.channel.nunique()==reference['raw_recorded_site_count']
    coords = coords.loc[~coords.subject.isin(reference['excluded_subjects'])].copy()
    all_sites = coords[['task','subject','channel']].drop_duplicates().rename(columns={'channel':'member_id'})
    assert all_sites.member_id.nunique()==reference['retained_coordinate_site_count']
    identity_mismatch = all_sites.loc[[not r.member_id.startswith(r.subject+'_') for r in all_sites.itertuples()]]
    identity_mismatch.to_csv(output/'venn_subject_prefix_mismatches.csv',index=False)
    # D0091LHETMA10-11 / 11-12 lack the usual underscore in the source.
    # Preserve their Figure 1 identifiers; use the explicit subject column.
    assert all(r.member_id.startswith(r.subject) for r in all_sites.itertuples())
    subject_lookup=all_sites[['member_id','subject']].drop_duplicates()
    assert not subject_lookup.member_id.duplicated().any()
    subject_map=subject_lookup.set_index('member_id').subject.to_dict()
    insula = pd.read_csv(insula_path)
    insula_sites = insula[['task','subject','site_id']].rename(columns={'site_id':'member_id'}).drop_duplicates()
    participants = all_sites[['task','subject']].drop_duplicates().assign(member_id=lambda d:d.subject)
    tables = {'subjects':participants,'all_sites':all_sites,'insula_sites':insula_sites}
    for task in TASKS:
        assert set(insula_sites.loc[insula_sites.task.eq(task),'member_id']).issubset(
            set(all_sites.loc[all_sites.task.eq(task),'member_id']))
    assert insula_sites.member_id.nunique()==reference['sampled_sites']==348

    # Match Figure 1's recorded subject/site identifiers. Flag native-coordinate
    # ambiguity separately; this diagram does not assert physical identity for
    # names whose reconstruction differs across task records.
    geometry_columns = [f'{axis}{endpoint}_native' for endpoint in [1,2] for axis in 'xyz']
    geometry = coords[['task','subject','channel',*geometry_columns]].drop_duplicates()
    extrema = geometry.groupby('channel')[geometry_columns].agg(['min','max'])
    conflict = np.zeros(len(extrema),dtype=bool)
    for col in geometry_columns:
        conflict |= (extrema[(col,'max')]-extrema[(col,'min')]).gt(1e-3).to_numpy()
    conflict_ids = set(extrema.index[conflict])
    conflict_rows = geometry.loc[geometry.channel.isin(conflict_ids)].sort_values(['channel','task'])
    conflict_rows.to_csv(output/'venn_native_coordinate_conflicts.csv',index=False)

    sets = {}; members = []; regions = []; totals = []
    for scope,table in tables.items():
        sets[scope] = {task:set(table.loc[table.task.eq(task),'member_id']) for task in TASKS}
        union = set.union(*sets[scope].values())
        membership = pd.DataFrame({'member_id':sorted(union)})
        membership['subject'] = membership.member_id if scope=='subjects' else membership.member_id.map(subject_map)
        assert membership.subject.notna().all()
        membership['scope'] = scope
        for task in TASKS:
            membership[ABBREVIATIONS[task]] = membership.member_id.isin(sets[scope][task])
        membership['region'] = membership[[ABBREVIATIONS[t] for t in TASKS]].astype(int).astype(str).agg(''.join,axis=1)
        membership['tasks'] = membership.apply(lambda r:'|'.join(t for t in TASKS if r[ABBREVIATIONS[t]]),axis=1)
        membership['native_coordinate_conflict'] = membership.member_id.isin(conflict_ids) if scope!='subjects' else False
        membership.to_csv(output/f'venn_{scope}_membership.csv',index=False)
        members.append(membership)
        counts = membership.region.value_counts()
        for bits in product([0,1],repeat=4):
            if not any(bits): continue
            region = ''.join(map(str,bits))
            chosen = [t for t,b in zip(TASKS,bits) if b]
            regions.append(dict(scope=scope,region=region,tasks='|'.join(chosen),
                exclusive_count=int(counts.get(region,0)),
                inclusive_count=len(set.intersection(*(sets[scope][t] for t in chosen)))))
        for task in TASKS:
            totals.append(dict(scope=scope,task=task,n_members=len(sets[scope][task]),union=len(union)))
    region_table = pd.DataFrame(regions)
    total_table = pd.DataFrame(totals)
    all_members = pd.concat(members,ignore_index=True)
    for scope in SCOPES:
        reg = region_table.loc[region_table.scope.eq(scope)]
        assert reg.exclusive_count.sum()==len(set.union(*sets[scope].values()))
        for i,task in enumerate(TASKS):
            assert reg.loc[reg.region.str[i].eq('1'),'exclusive_count'].sum()==len(sets[scope][task])
    region_table.to_csv(output/'venn_exact_intersections.csv',index=False)
    total_table.to_csv(output/'venn_task_totals.csv',index=False)
    all_members.to_csv(output/'venn_all_membership.csv',index=False)
    report = dict(condition=CONDITION,modality=MODALITY,tasks=list(TASKS),
        task_abbreviations=ABBREVIATIONS,task_colors=TASK_COLORS,
        unions={s:len(set.union(*sets[s].values())) for s in SCOPES},
        task_counts={s:{t:len(sets[s][t]) for t in TASKS} for s in SCOPES},
        excluded_subjects=reference['excluded_subjects'],
        subject_scope='All participants represented in Figure 1 whole-brain Repeat+sound coordinate records, not only participants with accepted insula sampling',
        all_sites_scope='All Figure 1 recorded bipolar subject/site names after D0121 exclusion; before anatomical ROI/insula-response filtering',
        insula_scope='Frozen accepted 348-site insula sampling list, without significance filtering',
        site_identity='Same subject + bipolar name, matching Figure 1 recorded-site total; never match bare channel names across subjects',
        native_coordinate_conflict_names=len(conflict_ids),
        recorded_names_without_standard_subject_delimiter=int(identity_mismatch.member_id.nunique()),
        participant_identity='Explicit subject column; original Figure 1 channel identifiers retained, including two D0091 names without an underscore',
        native_conflict_note='Whole-brain recorded-name overlaps are not a new adjudication of physical contact identity. Coordinate conflicts are exported separately; accepted insula identities retain their frozen decisions.',
        region_definition='Bits PS,LD,PN,SR. Each region contains members in exactly those sets, excluding the other tasks.',
        area_proportional=False,significance_filter_applied=False,
        sources={str(p):sha(p) for p in [coord_path,insula_path,fig1_path]})
    (output/'venn_input_and_counts.json').write_text(json.dumps(report,indent=2)+'\n')
    return sets,region_table,total_table,report


def four_set_geometry():
    """Fixed four-ellipse Venn: all 15 exclusive regions are represented."""
    from shapely import affinity
    from shapely.geometry import Point
    from shapely.ops import polylabel
    specs=[(.35,.40,140),(.45,.50,140),(.55,.50,40),(.65,.40,40)]
    ellipses=[]
    for x,y,angle in specs:
        shape=affinity.scale(Point(0,0).buffer(1,resolution=256),.36,.20)
        shape=affinity.translate(affinity.rotate(shape,angle,origin=(0,0)),x,y)
        ellipses.append(shape)
    anchors={}
    for bits in product([0,1],repeat=4):
        if not any(bits): continue
        included=[e for e,b in zip(ellipses,bits) if b]
        shape=included[0]
        for e in included[1:]:shape=shape.intersection(e)
        for e,b in zip(ellipses,bits):
            if not b:shape=shape.difference(e)
        assert not shape.is_empty and shape.area>1e-6, f'Missing Venn region: {bits}'
        polygon=max(shape.geoms,key=lambda p:p.area) if shape.geom_type=='MultiPolygon' else shape
        point=polylabel(polygon,tolerance=.0002)
        anchors[''.join(map(str,bits))]=(point.x,point.y)
    return specs,anchors


def draw_four_task_venn(ax, task_sets, region_counts, title):
    """Draw exact counts; ellipse areas deliberately do not encode set sizes."""
    from matplotlib.patches import Ellipse
    specs,anchors=four_set_geometry()
    for task,(x,y,angle) in zip(TASKS,specs):
        color=TASK_COLORS[task]
        ax.add_patch(Ellipse((x,y),.72,.40,angle=angle,facecolor=color,alpha=.16,edgecolor='none'))
        ax.add_patch(Ellipse((x,y),.72,.40,angle=angle,facecolor='none',edgecolor=color,lw=.9))
    counts=region_counts.set_index('region').exclusive_count
    for region,(x,y) in anchors.items():
        ax.text(x,y,str(int(counts.loc[region])),ha='center',va='center',fontsize=7,color='black')
    for task,(x,y,align) in zip(TASKS,[(.01,.10,'left'),(.17,.86,'center'),(.83,.86,'center'),(.99,.10,'right')]):
        ax.text(x,y,f'{TASK_DISPLAY_NAMES[task]}\nn={len(task_sets[task]):,}',
                ha=align,va='center',fontsize=7,color=TASK_COLORS[task])
    ax.set_title(f'{title}\nUnion: {len(set.union(*task_sets.values())):,}',fontsize=8,pad=6)
    ax.set_xlim(-.02,1.02);ax.set_ylim(.05,.90);ax.set_aspect('equal');ax.axis('off')


def _exclusive_regions(scales,resolution):
    from shapely import affinity
    from shapely.geometry import Point
    specs=[(.35,.40,140),(.45,.50,140),(.55,.50,40),(.65,.40,40)]
    ellipses=[]
    for (x,y,angle),(sx,sy) in zip(specs,scales):
        shape=affinity.scale(Point(0,0).buffer(1,resolution=resolution),.36*sx,.20*sy)
        ellipses.append(affinity.translate(affinity.rotate(shape,angle,origin=(0,0)),x,y))
    regions={}
    for bits in product([0,1],repeat=4):
        if not any(bits): continue
        included=[e for e,b in zip(ellipses,bits) if b]
        shape=included[0]
        for e in included[1:]: shape=shape.intersection(e)
        for e,b in zip(ellipses,bits):
            if not b: shape=shape.difference(e)
        regions[''.join(map(str,bits))]=shape
    return specs,regions


def _min_region_area(regions):
    return min(0.0 if shape.is_empty else shape.area for shape in regions.values())


def _axis_scales(area,uniform,floor):
    major=np.maximum(uniform,floor)
    minor=area/major
    return [(float(sx),float(sy)) for sx,sy in zip(major,minor)]


def _region_anchor(shape):
    from shapely.ops import polylabel
    if shape.geom_type=='GeometryCollection':
        shape=max((g for g in shape.geoms if g.area>0),key=lambda g:g.area)
    if shape.geom_type=='MultiPolygon':
        shape=max(shape.geoms,key=lambda g:g.area)
    point=polylabel(shape,tolerance=.0002)
    return (point.x,point.y)


def scaled_four_set_geometry(sizes):
    """Scale ellipse area to each task total without dropping an exclusive region."""
    sizes=np.asarray(list(sizes),dtype=float)
    area=sizes/sizes.max()
    uniform=np.sqrt(area)
    lo,hi=float(uniform.min()),1.0

    def smallest(floor,resolution):
        _,regions=_exclusive_regions(_axis_scales(area,uniform,floor),resolution)
        return _min_region_area(regions)

    if smallest(lo,96)>=1e-4:
        floor=lo
    else:
        for _step in range(40):
            mid=(lo+hi)/2
            if smallest(mid,96)>=1e-4: hi=mid
            else: lo=mid
        floor=hi
    scales=_axis_scales(area,uniform,floor)
    specs,regions=_exclusive_regions(scales,256)
    if not (_min_region_area(regions)>1e-6):
        floor=1.0
        scales=_axis_scales(area,uniform,floor)
        specs,regions=_exclusive_regions(scales,256)
    assert _min_region_area(regions)>1e-6 and len(regions)==15
    anchors={key:_region_anchor(shape) for key,shape in regions.items()}
    return specs,scales,anchors


def draw_scaled_four_task_venn(ax, task_sets, region_counts, title):
    """Draw exact counts; ellipse area is proportional to each task total."""
    from matplotlib.patches import Ellipse
    sizes=[len(task_sets[task]) for task in TASKS]
    specs,scales,anchors=scaled_four_set_geometry(sizes)
    for task,(x,y,angle),(sx,sy) in zip(TASKS,specs,scales):
        color=TASK_COLORS[task]
        ax.add_patch(Ellipse((x,y),.72*sx,.40*sy,angle=angle,facecolor=color,alpha=.16,edgecolor='none'))
        ax.add_patch(Ellipse((x,y),.72*sx,.40*sy,angle=angle,facecolor='none',edgecolor=color,lw=.9))
    counts=region_counts.set_index('region').exclusive_count
    for region,(x,y) in anchors.items():
        ax.text(x,y,str(int(counts.loc[region])),ha='center',va='center',fontsize=7,color='black')
    for task,(x,y,align) in zip(TASKS,[(.01,.10,'left'),(.17,.86,'center'),(.83,.86,'center'),(.99,.10,'right')]):
        ax.text(x,y,f'{TASK_DISPLAY_NAMES[task]}\nn={len(task_sets[task]):,}',
                ha=align,va='center',fontsize=7,color=TASK_COLORS[task])
    ax.set_title(f'{title}\nUnion: {len(set.union(*task_sets.values())):,}',fontsize=8,pad=6)
    ax.set_xlim(-.02,1.02);ax.set_ylim(.05,.90);ax.set_aspect('equal');ax.axis('off')
