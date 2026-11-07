"""Approved Sound–Image group test and baseline-overlap descriptions."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.supplementary.decision_repeat_maps import sha

PARAMS = dict(p_thresh=.05,p_cluster=.05,tails=1,n_perm=5000,axis=0,
              permutation_type='samples',seed=42)


def intervals(times, mask, window):
    changes = np.diff(np.r_[False,mask,False].astype(int))
    edges = np.r_[times[0]-(times[1]-times[0])/2,
                  (times[:-1]+times[1:])/2,times[-1]+(times[-1]-times[-2])/2]
    return [dict(start=float(max(edges[a],window[0])),end=float(min(edges[b],window[1])),
                 first_significant_time=float(times[a]),last_significant_time=float(times[b-1]),n_bins=int(b-a))
            for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1))]


def pooled_timeperm(frame, trace_table, xlims, out, *, recompute=False):
    """Same electrode estimand and subject-block swaps as Decision/Repeat."""
    from ieeg.calc import stats
    from ieeg.calc.fast import mean_diff
    out = Path(out)
    source = out.parent / 'context_comparisons/waveforms.pkl'
    stats_path = inspect.getsourcefile(stats.time_perm_cluster)
    original_stats_sha = sha(stats_path)
    manifest_path = out / 'condition_timeperm_manifest.json'
    table_path = out / 'sound_gt_image_timeperm.csv'
    interval_path = out / 'sound_gt_image_significant_intervals.csv'
    if manifest_path.exists() and not recompute:
        manifest = json.loads(manifest_path.read_text())
        assert manifest['parameters'] == PARAMS and manifest['source_cache_sha256'] == sha(source)
        for name,expected in manifest['outputs'].items(): assert sha(out/name)==expected
        table,spans = pd.read_csv(table_path),pd.read_csv(interval_path)
    else:
        sites = frame[['electrode','subject']].drop_duplicates().sort_values('electrode')
        subjects = sorted(sites.subject.unique())
        assert len(sites)==81 and len(subjects)==24
        payload = dict(electrode=sites.electrode.to_numpy(dtype=str),subject=np.asarray(subjects,dtype=str))
        tables,spans = [],[]
        for phase,window in xlims.items():
            sub = frame.loc[frame.phase.eq(phase) & frame.time.between(*window)]
            weighted = {}
            for level in ['sound','image']:
                pivot = sub.loc[sub.level.eq(level)].pivot(index='electrode',columns='time',values='value')
                pivot = pivot.reindex(sites.electrode).sort_index(axis=1)
                assert np.isfinite(pivot.to_numpy()).all()
                times = pivot.columns.to_numpy(float)
                weighted[level] = np.stack([pivot.loc[sites.loc[sites.subject.eq(s),'electrode']].sum(axis=0).to_numpy()
                    for s in subjects]) * len(subjects)/len(sites)
                payload[f'{phase}_{level}'] = weighted[level]
                payload[f'{phase}_time'] = times
            observed = mean_diff(weighted['sound'],weighted['image'],axis=0)
            expected = trace_table.loc[trace_table.phase.eq(phase)].pivot(index='time',columns='level',values='mean').sort_index()
            np.testing.assert_array_equal(times,expected.index.to_numpy())
            np.testing.assert_allclose(observed,expected.sound-expected.image,rtol=0,atol=1e-14)
            print('Pooled Sound > Image:',phase,len(times),'samples; 5000 subject-block swaps',flush=True)
            mask,p_raw = stats.time_perm_cluster(weighted['sound'],weighted['image'],stat_func=mean_diff,n_jobs=1,**PARAMS)
            mask,p_raw = np.asarray(mask,dtype=bool),np.asarray(p_raw,dtype=float)
            assert mask.shape == p_raw.shape == observed.shape and np.isfinite(p_raw).all()
            assert not np.any(mask & (observed<=0))
            tables.append(pd.DataFrame(dict(phase=phase,time=times,mask=mask,p_value=p_raw,
                mean_difference=observed,n_sites=81,n_subjects=24)))
            spans.extend(dict(phase=phase,**span) for span in intervals(times,mask,window))
        table = pd.concat(tables,ignore_index=True)
        spans = pd.DataFrame(spans,columns=['phase','start','end','first_significant_time','last_significant_time','n_bins'])
        table.to_csv(table_path,index=False);spans.to_csv(interval_path,index=False)
        np.savez_compressed(out/'pooled_paired_inputs.npz',**payload)
        assert sha(stats_path)==original_stats_sha
        manifest = dict(parameters=PARAMS,statistic='ieeg.calc.fast.mean_diff',n_sites=81,n_subjects=24,
            source_cache_sha256=sha(source),approval_sha256=sha(out/'approval.json'),
            installed_stats_source=stats_path,installed_stats_sha256=original_stats_sha,installed_library_unmodified=True,
            estimand='Electrode-equal mean; each subject contribution is S/N times its electrode sum',
            exchangeability='Paired condition exchange for all sites of one subject together',
            correction_scope='Original time_perm_cluster separately within each displayed phase',
            outputs={name:sha(out/name) for name in [table_path.name,interval_path.name,'pooled_paired_inputs.npz']})
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    result = {}
    for phase in xlims:
        saved = table.loc[table.phase.eq(phase)].sort_values('time')
        expected = trace_table.loc[trace_table.phase.eq(phase)].pivot(index='time',columns='level',values='mean').sort_index()
        np.testing.assert_allclose(saved.time,expected.index,rtol=0,atol=1e-15)
        np.testing.assert_allclose(saved.mean_difference,expected.sound-expected.image,rtol=0,atol=1e-14)
        result[phase] = dict(time=saved.time.to_numpy(),mask=saved['mask'].to_numpy(dtype=bool),
            intervals=list(spans.loc[spans.phase.eq(phase),['start','end']].itertuples(index=False,name=None)))
    return result,spans


def prepare_stimulus_overlap(root, frame, selected, out):
    root,out = Path(root),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    frozen = frame[['subject','channel','electrode','functional_cluster']].drop_duplicates()
    assert len(frozen)==81 and frozen.subject.nunique()==24
    mapped = frame.loc[frame.phase.eq('stimulus') & frame.time.between(0,.6)]
    assert mapped.mask_observed.all() and not mapped.duplicated(['electrode','level','time']).any()
    assert mapped.groupby(['electrode','level']).size().eq(77).all()
    masks = mapped.groupby(['electrode','level'])['mask'].any().unstack('level')
    assert not masks.isna().any().any()
    members = frozen.merge(masks.reset_index(),on='electrode',validate='one_to_one')
    members = members.rename(columns={'sound':'sound_baseline_responsive','image':'image_baseline_responsive'})
    sound = set(members.loc[members.sound_baseline_responsive,'electrode'])
    image = set(members.loc[members.image_baseline_responsive,'electrode'])
    sites = set(members.electrode)
    regions = {'Sound only':sound-image,'Both':sound&image,'Image only':image-sound,'Neither':sites-(sound|image)}
    members['venn_region'] = members.electrode.map({s:r for r,group in regions.items() for s in group})
    members['sound_gt_image_selected'] = members.electrode.isin(selected.electrode)
    members['phase'],members['window_start'],members['window_end'] = 'stimulus',0.,.6
    members.to_csv(out/'baseline_overlap_membership.csv',index=False)
    counts = pd.DataFrame([dict(region=r,n_sites=len(group),n_subjects=members.loc[members.electrode.isin(group),'subject'].nunique())
                           for r,group in regions.items()])
    counts.to_csv(out/'baseline_overlap_region_counts.csv',index=False)
    members.groupby(['functional_cluster','venn_region']).size().rename('n_sites').reset_index().to_csv(out/'baseline_overlap_component_counts.csv',index=False)
    assignments = pd.read_csv(root/'results/nmf/channel_assignments.csv')
    points = frozen[['subject','channel','electrode']].merge(assignments,on=['subject','channel'],validate='one_to_one')
    assert len(points)==81 and np.isfinite(points[['x','y','z']]).all().all() and (points.x!=0).all()
    assert points.functional_cluster.eq(points.electrode.map(frozen.set_index('electrode').functional_cluster)).all()
    points = points.rename(columns={'significant':'original_assignment_significant'})
    points['significant'] = points.electrode.isin(selected.electrode)
    points['pattern'] = np.where(points.significant,1.,np.nan)
    points['map_phase'],points['map_window_start'],points['map_window_end'] = 'stimulus',0.,.6
    points.to_csv(out/'sound_gt_image_brain_membership.csv',index=False)
    hemi = pd.DataFrame([dict(hemisphere=name,n_tested=int(side.sum()),n_selected=int(points.loc[side,'significant'].sum()))
                         for name,side in [('Left',points.x.lt(0)),('Right',points.x.gt(0))]])
    hemi.to_csv(out/'brain_hemisphere_counts.csv',index=False)
    cross = members.groupby(['venn_region','sound_gt_image_selected']).size().rename('n_sites').reset_index()
    cross.to_csv(out/'baseline_and_difference_cross_tab.csv',index=False)
    report = dict(window=[0,.6],n_sites=81,n_subjects=24,sound_baseline_responsive=len(sound),
        image_baseline_responsive=len(image),region_counts={r:len(group) for r,group in regions.items()},
        n_selected=len(selected),n_selected_subjects=int(selected.subject.nunique()),
        component_counts=selected.functional_cluster.value_counts().to_dict(),hemisphere_counts=hemi.to_dict(orient='records'),
        baseline_definition='Any existing native baseline time-permutation mask=True in Stimulus [0,.6] s',
        baseline_masks_recomputed=False,NMF_changed=False,
        brain_definition='Saved positive-direction per-electrode Sound > Image temporal significance in Stimulus [0,.6] s')
    (out/'input_and_membership_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return members,counts,points,report


def draw_stimulus_venn(members,report):
    from matplotlib_venn import venn2,venn2_circles
    from matplotlib.patches import PathPatch
    sound = set(members.loc[members.sound_baseline_responsive,'electrode'])
    image = set(members.loc[members.image_baseline_responsive,'electrode'])
    counts = (len(sound-image),len(image-sound),len(sound&image))
    cm = 1/2.54
    fig,ax = plt.subplots(figsize=(7.2*cm,5.4*cm))
    fig.subplots_adjust(left=.08,right=.92,bottom=.23,top=.78)
    diagram = venn2(subsets=counts,set_labels=(f'Sound\n(n={len(sound)})',f'Image\n(n={len(image)})'),
        set_colors=('#6A5ACD','#000000'),alpha=.30,ax=ax)
    for artist,color in zip(venn2_circles(subsets=counts,ax=ax,linewidth=.8),('#6A5ACD','#000000')):
        artist.set_edgecolor(color)
    # Separate the three regions visually while retaining their exact geometry.
    common_patch = diagram.get_patch_by_id('11')
    if common_patch is not None:
        ax.add_patch(PathPatch(common_patch.get_path(), transform=common_patch.get_transform(),
            facecolor='none', edgecolor='white', linewidth=3., joinstyle='round',
            capstyle='round', zorder=2.5))
    for label in diagram.subset_labels:
        if label is not None: label.set_fontsize(8)
    for label,color in zip(diagram.set_labels,('#6A5ACD','#000000')):
        if label is not None: label.set_fontsize(7);label.set_color(color)
    fig.suptitle('Stimulus 0–0.6 s\nBaseline-responsive sites',fontsize=7.5,y=.97)
    fig.text(.5,.025,f"Neither: n={report['region_counts']['Neither']} · Total: n=81",ha='center',va='bottom',fontsize=6.5)
    return fig


from matplotlib.lines import Line2D
from src.nmf.waveform_analysis import FUNCTION_COLORS
from src.supplementary.decision_repeat_maps import _figure1_local_insula_renderer, CLUSTERS, LABELS


def draw_sound_difference_brain(points, crop, *, display_output=None):
    """Use the same local insula renderer and surface snap as Figure 1."""
    root = Path(__file__).resolve().parents[2]
    renderer,proof = _figure1_local_insula_renderer(root)
    display_output = Path(display_output) if display_output is not None else None
    audit_rows = []
    cm = 1/2.54
    fig,axes = plt.subplots(1,2,figsize=(8.6*cm,4.7*cm))
    fig.subplots_adjust(left=.03,right=.98,bottom=.26,top=.80,wspace=.03)
    fig.suptitle('Sound > Image · Stimulus 0–0.6 s',fontsize=7.5,y=.98)
    for ax,hemi,title,side in zip(axes,('lh','rh'),('Left insula','Right insula'),(points.x.lt(0),points.x.gt(0))):
        frame = points.loc[side].copy()
        coordinates = frame[['x','y','z']].to_numpy(float)
        colors = np.asarray([FUNCTION_COLORS[row.functional_cluster] if row.significant
                             else renderer['NSIG_COLOR'] for row in frame.itertuples()],dtype=object)
        # Author-requested visibility increase for this map only; retain
        # the exact Figure 1 surface, snap, camera and renderer helpers.
        sizes = np.where(frame.significant,24.,10.)
        brain = None
        try:
            image,project,brain = renderer['_render_insula_hemi'](hemi,coordinates,colors,sizes)
            camera = renderer['_camera_xyz'](brain)
            surf,pial = renderer[f'{hemi}_surf'],renderer[f'{hemi}_pial_coords']
            display_xyz = renderer['_project_to_insula'](coordinates,surf,pial,camera)
            # Undo only the known Figure 1 camera lift to verify the base
            # point lies exactly on a vertex of the displayed insula mesh.
            toward_camera = camera-display_xyz
            toward_camera /= np.linalg.norm(toward_camera,axis=1,keepdims=True)
            snapped = display_xyz-renderer['ELECTRODE_OFFSET_MM']*toward_camera
            residual,vertex_local = renderer['cKDTree'](pial[surf['verts']]).query(snapped)
            assert np.max(residual) < 1e-8, 'Display projection is off the insula surface'
            np.testing.assert_allclose(np.linalg.norm(display_xyz-snapped,axis=1),4.,rtol=0,atol=1e-10)
            pixels = project(display_xyz)
            for i,row in enumerate(frame.itertuples()):
                audit_rows.append(dict(electrode=row.electrode,subject=row.subject,channel=row.channel,
                    hemisphere=hemi,selected=bool(row.significant),functional_cluster=row.functional_cluster,
                    x_original=float(coordinates[i,0]),y_original=float(coordinates[i,1]),z_original=float(coordinates[i,2]),
                    x_surface=float(snapped[i,0]),y_surface=float(snapped[i,1]),z_surface=float(snapped[i,2]),
                    x_display=float(display_xyz[i,0]),y_display=float(display_xyz[i,1]),z_display=float(display_xyz[i,2]),
                    insula_vertex=int(surf['verts'][vertex_local[i]]),surface_residual_mm=float(residual[i]),
                    display_lift_mm=4.,pixel_x=float(pixels[i,0]),pixel_y=float(pixels[i,1])))
        finally:
            if brain is not None:brain.close()
        ax.imshow(crop(image,pad=24));ax.axis('off')
        n = int(points.loc[side,'significant'].sum())
        ax.set_title(f'{title} · n={n}',fontsize=7,pad=2)
    handles = [Line2D([0],[0],marker='o',color='none',markerfacecolor=FUNCTION_COLORS[c],
        markeredgewidth=0,markersize=4,label=f"{LABELS[c]} ({int((points.significant & points.functional_cluster.eq(c)).sum())})")
        for c in CLUSTERS]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.095),fontsize=6,
        frameon=False,ncol=3,handletextpad=.25,columnspacing=.65)
    fig.text(.5,.018,f'Colored: {int(points.significant.sum())} selected sites · Gray: {int((~points.significant).sum())} other tested sites',
        ha='center',va='bottom',fontsize=5.9)
    assert len(audit_rows) == 81 and sum(row['selected'] for row in audit_rows) == int(points.significant.sum())
    if display_output is not None:
        display_output.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(audit_rows).to_csv(display_output/'display_projection_audit.csv',index=False)
        proof.update(n_sites=81,n_selected=int(points.significant.sum()),
                     all_base_points_on_insula_surface=True,
                     display_marker_sizes_px=dict(selected=24.,context=10.),
                     maximum_surface_residual_mm=max(row['surface_residual_mm'] for row in audit_rows))
        (display_output/'figure1_renderer_provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
    return fig
