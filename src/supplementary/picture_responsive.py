"""Describe existing Picture Stimulus responses; no new inference."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.supplementary.sound_image import draw_stimulus_venn


def prepare_picture_responses(root, frame, out, *, require_sound=False):
    root,out = Path(root),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    candidates = frame[['subject','channel','electrode','functional_cluster']].drop_duplicates()
    assert len(candidates)==81 and candidates.subject.nunique()==24
    assert frame.task.eq('PictureNaming').all() and frame.description.eq('Repeat').all()
    mapped = frame.loc[frame.phase.eq('stimulus') & frame.time.between(0,.6)].copy()
    assert mapped.mask_observed.all()
    assert mapped.groupby(['electrode','level']).size().eq(77).all()
    assert not mapped.duplicated(['electrode','level','time']).any()
    masks = mapped.groupby(['electrode','level'])['mask'].any().unstack('level')
    assert not masks.isna().any().any()
    members = candidates.merge(masks.reset_index(),on='electrode',validate='one_to_one')
    members = members.rename(columns={'sound':'sound_baseline_responsive','image':'image_baseline_responsive'})
    members['picture_stimulus_responsive'] = members.image_baseline_responsive
    members['common_stimulus_responsive'] = members.sound_baseline_responsive & members.image_baseline_responsive
    selected_column = 'common_stimulus_responsive' if require_sound else 'picture_stimulus_responsive'
    prefix = 'common_stimulus' if require_sound else 'picture_stimulus'
    members['selection_window_start'],members['selection_window_end'] = 0.,.6
    members['selection_basis'] = ('Existing Sound and Image baseline masks each hit Stimulus [0,.6] s'
                                  if require_sound else 'Existing Picture baseline mask=True in Stimulus [0,.6] s')
    selected = members.loc[members[selected_column]].copy()
    assert 0 < len(selected) <= len(candidates)
    assert set(selected.electrode) <= set(candidates.electrode)
    sound = set(members.loc[members.sound_baseline_responsive,'electrode'])
    picture = set(members.loc[members.picture_stimulus_responsive,'electrode'])
    common = sound & picture
    all_sites = set(members.electrode)
    regions = {'Sound only':sound-picture,'Both':sound&picture,'Image only':picture-sound,
               'Neither':all_sites-(sound|picture)}
    members['venn_region'] = members.electrode.map({s:r for r,sites in regions.items() for s in sites})
    selected = members.loc[members[selected_column]].copy()
    members.to_csv(out/'baseline_overlap_membership.csv',index=False)
    selected.to_csv(out/f'{prefix}_responsive_electrodes.csv',index=False)
    counts = pd.DataFrame([dict(region=r,n_sites=len(sites),n_subjects=members.loc[members.electrode.isin(sites),'subject'].nunique())
                           for r,sites in regions.items()])
    counts.to_csv(out/'baseline_overlap_region_counts.csv',index=False)
    component_counts = pd.DataFrame([dict(functional_cluster=c,
        n_candidates=int(candidates.functional_cluster.eq(c).sum()),
        n_selected=int(selected.functional_cluster.eq(c).sum()),
        n_picture_responsive=int((members.picture_stimulus_responsive & members.functional_cluster.eq(c)).sum()),
        n_common_responsive=int((members.common_stimulus_responsive & members.functional_cluster.eq(c)).sum()),
        n_subjects=int(selected.loc[selected.functional_cluster.eq(c),'subject'].nunique()))
        for c in ['sensory','sustain','motor']])
    component_counts.to_csv(out/'component_counts.csv',index=False)
    selected.groupby('subject').size().rename('n_selected').reset_index().to_csv(out/'subject_counts.csv',index=False)
    intervals=[]
    for site,group in mapped.loc[mapped.level.eq('image') & mapped.electrode.isin(set(selected.electrode))].groupby('electrode'):
        group=group.sort_values('time');t=group.time.to_numpy();mask=group['mask'].to_numpy(bool)
        edges=np.r_[t[0]-(t[1]-t[0])/2,(t[:-1]+t[1:])/2,t[-1]+(t[-1]-t[-2])/2]
        changes=np.diff(np.r_[False,mask,False].astype(int))
        info=selected.loc[selected.electrode.eq(site)].iloc[0]
        for i,(a,b) in enumerate(zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)),1):
            intervals.append(dict(subject=info.subject,channel=info.channel,electrode=site,
                functional_cluster=info.functional_cluster,interval=i,start=max(0.,float(edges[a])),
                end=min(.6,float(edges[b])),first_significant_time=float(t[a]),last_significant_time=float(t[b-1])))
    pd.DataFrame(intervals).to_csv(out/'existing_picture_baseline_intervals.csv',index=False)
    assignments = pd.read_csv(root/'results/nmf/channel_assignments.csv')
    points=candidates[['subject','channel','electrode']].merge(assignments,on=['subject','channel'],validate='one_to_one')
    assert len(points)==81 and np.isfinite(points[['x','y','z']]).all().all()
    points=points.rename(columns={'significant':'original_assignment_significant'})
    points['picture_stimulus_responsive']=points.electrode.isin(picture)
    points['common_stimulus_responsive']=points.electrode.isin(common)
    points['significant']=points.electrode.isin(set(selected.electrode))
    points['pattern']=np.where(points.significant,1.,np.nan)
    points['map_phase'],points['map_window_start'],points['map_window_end']='stimulus',0.,.6
    points.to_csv(out/f'{prefix}_responsive_brain_membership.csv',index=False)
    hemi=pd.DataFrame([dict(hemisphere=name,n_candidates=int(side.sum()),
        n_selected=int(points.loc[side,'significant'].sum()))
        for name,side in [('Left',points.x.lt(0)),('Right',points.x.gt(0))]])
    hemi.to_csv(out/'brain_hemisphere_counts.csv',index=False)
    report=dict(n_sites=81,n_subjects=24,n_picture_responsive=len(picture),
        n_picture_responsive_subjects=members.loc[members.picture_stimulus_responsive,'subject'].nunique(),
        n_common_responsive=len(common),n_common_subjects=members.loc[members.common_stimulus_responsive,'subject'].nunique(),
        n_selected=len(selected),n_selected_subjects=selected.subject.nunique(),
        sound_baseline_responsive=len(sound),image_baseline_responsive=len(picture),
        region_counts={r:len(sites) for r,sites in regions.items()},
        component_counts={c:int(selected.functional_cluster.eq(c).sum()) for c in ['sensory','sustain','motor']},hemisphere_counts=hemi.to_dict(orient='records'),
        selection_window=[0,.6],selection=('Sound and Image baseline mask.any separately within Stimulus [0,.6] s; intersection'
            if require_sound else 'Picture baseline mask.any within Stimulus [0,.6] s'),
        same_phase_sound_response_required=require_sound,new_statistics=False,condition_difference_test_used=False,
        brain_definition='The same selected sites as the lower Picture heatmaps',
        heatmap_condition='image',NMF_changed=False)
    (out/'membership_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return members,selected,counts,points,report


def prepare_common_responses(root, frame, out):
    """Both conditions respond in the window; their significant bins need not overlap."""
    return prepare_picture_responses(root, frame, out, require_sound=True)


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.nmf.waveform_analysis import FUNCTION_COLORS
from src.supplementary.decision_repeat_maps import _figure1_local_insula_renderer, CLUSTERS, LABELS


def draw_picture_responsive_brain(points, crop, *, display_output=None):
    """Selected points with original Figure 2 contours; omit gray background sites."""
    from src.supplementary.selected_brain_contours import draw_selected_brain_contours
    return draw_selected_brain_contours(points, 'Picture-responsive · Stimulus 0–0.6 s', display_output=display_output)


def draw_common_responsive_brain(points, crop, *, display_output=None):
    """The shared Sound/Image Stimulus set with the approved contour display."""
    from src.supplementary.selected_brain_contours import draw_selected_brain_contours
    return draw_selected_brain_contours(points, 'Sound ∩ Image · Stimulus 0–0.6 s', display_output=display_output)
