"""Figure 3 complete response denominators from existing native baseline masks.

Descriptive counts and displays only. No inference or old analysis is rewritten.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib_venn import venn2, venn2_circles
import numpy as np
import pandas as pd

from src.nmf.waveform_analysis import FUNCTION_COLORS
from src.supplementary.decision_repeat_maps import CLUSTERS, LABELS

ROOT = Path('/cwork/ns458/insula-paper')
OUT = ROOT/'results/supplementary/fig3_response_characterization_2026-10-23'
IMG = ROOT/'img/supplementary/fig3_response_characterization'
SPECS = [
    dict(key='sound_image', title='Sound–Image', task='PictureNaming', phase='stimulus',
         window=(0., .6), levels=('sound', 'image'), names=('Sound', 'Image'), bins=77,
         cache='sound_image_test/event_repair_2026-10-21/waveforms.pkl',
         native='sound_image_test/event_repair_2026-10-21/native_input_manifest.json',
         members='sound_image_test/common_stimulus_response/baseline_overlap_membership.csv',
         frozen='sound_image_test/frozen_heatmap_row_membership.csv'),
    dict(key='decision_repeat', title='Decision–Repeat · original analysis cohort', task='LexicalDelay', phase='delay',
         window=(0., 1.), levels=('Decision', 'Repeat'), names=('Decision', 'Repeat'), bins=129,
         cache='context_comparisons/waveforms.pkl', native='context_comparisons/native_input_manifest.json',
         members='decision_repeat_test/delay_overlap_maps/baseline_overlap_membership.csv',
         frozen='decision_repeat_test/frozen_heatmap_row_membership.csv'),
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')


def decoded(ds):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in ds[:]]


def native_checks(spec, frame, membership):
    """Match cached phase masks to their exact original/repaired statistics."""
    base = ROOT/'results/supplementary'
    manifest = json.loads((base/spec['native']).read_text())
    rows = []
    for subject, group in membership.groupby('subject'):
        requested = set(group.channel)
        for level in spec['levels']:
            modality = level if spec['key'] == 'sound_image' else 'sound'
            condition = 'Repeat' if spec['key'] == 'sound_image' else level
            recording = f'_recording-{modality}' if spec['task'] == 'PictureNaming' else ''
            suffix = f"sub-{subject}_task-{spec['task']}_proc-{spec['phase'].capitalize()}{recording}_desc-{condition}_highgamma.h5"
            replacements = [r for r in manifest.get('replacement_inputs', []) if Path(r['stats']['path']).name == suffix]
            records = replacements or [r for r in manifest['input_files'] if Path(r['stats']['path']).name == suffix]
            records = [r for r in records if requested <= set(r['channels'])]
            assert records, (subject, level, 'Missing retained native source')
            record = max(records, key=lambda r: len(r['channels']))
            stats = Path(record['stats']['path'])
            assert sha(stats) == record['statistics_sha256'], str(stats)
            with np.load(record['cache'], allow_pickle=False) as f:
                times = f['times'][:]
            keep = (times >= spec['window'][0]) & (times <= spec['window'][1])
            assert int(keep.sum()) == spec['bins']
            with h5py.File(stats, 'r') as f:
                names = decoded(f['ch_names'])
                for channel in sorted(requested):
                    native = f['mask'][names.index(channel)].astype(bool)[keep]
                    saved = frame.loc[frame.subject.eq(subject) & frame.channel.eq(channel) & frame.level.eq(level)].sort_values('time')
                    np.testing.assert_array_equal(saved.time, times[keep])
                    np.testing.assert_array_equal(saved['mask'].to_numpy(bool), native)
            rows.append(dict(subject=subject, level=level, n_sites=len(requested), stats=str(stats),
                             statistics_sha256=record['statistics_sha256'], repaired_source=bool(replacements),
                             native_masks_identical=True))
    pd.DataFrame(rows).to_csv(OUT/f"{spec['key']}_native_mask_verification.csv", index=False)


def count_sets(membership, spec):
    rows = []
    for component in ('all', *CLUSTERS):
        group = membership if component == 'all' else membership.loc[membership.functional_cluster.eq(component)]
        sets = {'Candidates': np.ones(len(group), bool), spec['names'][0]+' responsive': group.response_a,
                spec['names'][1]+' responsive': group.response_b}
        sets.update({r: group.response_class.eq(r) for r in ('A only', 'Both', 'B only', 'Neither')})
        for name, mask in sets.items():
            subset = group.loc[np.asarray(mask)]
            rows.append(dict(comparison=spec['key'], component=component, category=name,
                n_sites=len(subset), n_subjects=subset.subject.nunique(),
                n_candidate_sites=len(group), n_candidate_subjects=group.subject.nunique(),
                subject_list=';'.join(sorted(subset.subject.unique()))))
    return pd.DataFrame(rows)


def denominator_chain(assignments, coverage, spec, membership):
    """Separate task availability/QC/original gate from current phase responses."""
    sampled = pd.read_csv(ROOT/'docs/evidence/sound_only_cohort_2026-10-17/sampled_sites_by_subject_task.csv')
    # Figure 1 names the same canonical bipolar identifier "site_id".
    sampled = sampled.rename(columns={'site_id':'channel'})
    sampled = sampled.loc[sampled.task.eq(spec['task'])]
    accepted = set(zip(sampled.subject, sampled.channel))
    task_subjects = set(coverage.loc[coverage.task.eq(spec['task']), 'subject'])
    good_by_subject, sig_by_subject = {}, {}
    for subject in sorted(task_subjects & set(assignments.subject)):
        path = ROOT/f"results/native_preprocessing_restore_2026-10-18/{spec['task']}/{subject}/output_validation.json"
        if not path.exists():
            continue
        products = json.loads(path.read_text())['products']
        good, hits = None, set()
        for p in products:
            if p['modality'] != 'sound':
                continue
            with h5py.File(p['statistics_path'], 'r') as f:
                names = set(decoded(f['ch_names']))
                good = names if good is None else good & names
                hits.update(decoded(f['sig_ch_names']))
        good_by_subject[subject] = good or set()
        sig_by_subject[subject] = hits
    included = set(membership.electrode)
    rows = []
    for r in assignments.itertuples():
        key = f'{r.subject}|{r.channel}'
        task_record = r.subject in task_subjects
        native_good = r.channel in good_by_subject.get(r.subject, set())
        response_gate = spec['key'] == 'sound_image' or r.channel in sig_by_subject.get(r.subject, set())
        reason = ('Included' if key in included else
                  'No subject record for this task' if not task_record else
                  'Not retained in the original native-good channel intersection' if not native_good else
                  'Did not pass the original any-phase sound-response gate' if not response_gate else
                  'Absent from retained paired comparison cache; inspect separately')
        rows.append(dict(comparison=spec['key'], subject=r.subject, channel=r.channel, electrode=key,
            functional_cluster=r.functional_cluster, has_task_record=task_record,
            accepted_task_sample=(r.subject, r.channel) in accepted, native_good=native_good,
            original_response_gate=response_gate, included=key in included, reason=reason))
    table = pd.DataFrame(rows)
    table.to_csv(OUT/f"{spec['key']}_cohort_flow_all_nmf_sites.csv", index=False)
    table.loc[~table.included].to_csv(OUT/f"{spec['key']}_excluded_from_comparison.csv", index=False)
    stages = {'All fixed NMF sites': np.ones(len(table), bool),
              'NMF sites with subject task record': table.has_task_record,
              'Original native-good intersection': table.has_task_record & table.native_good,
              'Original task response gate': table.has_task_record & table.native_good & table.original_response_gate,
              'Retained paired comparison cohort': table.included}
    result = []
    for component in ('all', *CLUSTERS):
        for name, mask in stages.items():
            subset = table.loc[np.asarray(mask) & (True if component == 'all' else table.functional_cluster.eq(component))]
            result.append(dict(comparison=spec['key'], component=component, stage=name,
                n_sites=len(subset), n_subjects=subset.subject.nunique()))
    result.append(dict(comparison=spec['key'], component='all', stage='Task anatomical sampling (before NMF)',
                       n_sites=len(sampled.drop_duplicates(['subject','channel'])), n_subjects=sampled.subject.nunique()))
    return pd.DataFrame(result)


def svg_labels(path):
    tree = ET.parse(path)
    return ' '.join(''.join(node.itertext()) for node in tree.iter() if node.tag.endswith('}text'))


def version_audit(memberships):
    checks = []
    bundle = OUT/'review_bundle'
    bundle.mkdir(exist_ok=True)
    for spec in SPECS:
        key = spec['key']
        topic = ROOT/f'results/supplementary/{key}_test'
        nb = json.loads((ROOT/f'vizpub/supplementary/{key}_test.ipynb').read_text())
        if key == 'sound_image':
            selected = memberships[key].loc[memberships[key].response_class.eq('Both')]
            rowpath = topic/'picture_heatmap_row_membership.csv'
            matrices = topic/'picture_heatmap_matrices.npz'
            prefix = 'picture'
            brain = ROOT/'img/supplementary/sound_image_test/fig3_sound_image_common_stimulus_brain.svg'
            expected = dict(sensory=2, sustain=6, motor=3)
            cellid = 'sound-plot'
        else:
            selected = pd.read_csv(topic/'delay_electrode_selection/selected_electrodes.csv')
            rowpath = topic/'decision_heatmap_row_membership.csv'
            matrices = topic/'decision_heatmap_matrices.npz'
            prefix = 'decision'
            brain = ROOT/'img/supplementary/decision_repeat_test/fig3_decision_gt_repeat_delay_brain.svg'
            expected = dict(sensory=3, sustain=14, motor=12)
            cellid = 'test-plot'
        selected_set = set(selected.electrode)
        rows = pd.read_csv(rowpath)
        with np.load(matrices, allow_pickle=False) as f:
            for phase in ('stimulus', 'delay', 'response'):
                for component, n in expected.items():
                    sub = rows.loc[rows.phase.eq(phase) & rows.functional_cluster.eq(component)]
                    assert len(sub) == n
                    assert set(sub.electrode) == set(selected.loc[selected.functional_cluster.eq(component), 'electrode'])
                    assert f[f'{phase}_{component}'].shape[0] == n
        heatmap = ROOT/f'img/supplementary/{key}_test/fig3_{key}_test.svg'
        text = svg_labels(heatmap)
        for component, n in expected.items():
            assert f'{LABELS[component]} (n={n})' in text, (heatmap, component, text)
        cell = next(c for c in nb['cells'] if c['id'] == cellid)
        embedded = [''.join(o['data']['image/svg+xml']) if isinstance(o['data']['image/svg+xml'], list)
                    else o['data']['image/svg+xml'] for o in cell['outputs'] if 'image/svg+xml' in o.get('data', {})]
        assert embedded
        for component, n in expected.items():
            assert f'{LABELS[component]} (n={n})' in embedded[-1]
        for path in (heatmap, brain):
            shutil.copy2(path, bundle/path.name)
        checks.append(dict(comparison=key, selected_sites=len(selected_set), component_counts=expected,
            current_heatmap_rows_and_matrices_agree=True, current_svg_and_notebook_labels_agree=True,
            heatmap=str(heatmap), brain=str(brain), heatmap_sha256=sha(heatmap), brain_sha256=sha(brain)))
    screenshot = '/hpc/home/ns458/.codex/attachments/c79037f5-4c68-44f0-a182-61331f4d89fa/codex-clipboard-85d0d010-5409-4af5-ada4-75b53be76a9d.png'
    report = dict(local_panels=checks, reviewed_assembled_screenshot=screenshot,
        screenshot_finding='Sound/Image heatmap shows 3/8/7 Picture responders (18), while brain shows 2/6/3 common responders (11): mixed versions.',
        current_editable_assembly_available=False,
        limitation='The supplied assembled screenshot and current local panels were checked; no newer editable manuscript assembly was supplied.',
        review_bundle=str(bundle))
    write_json(OUT/'version_audit.json', report)
    return report


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    assignments = pd.read_csv(ROOT/'results/nmf/channel_assignments.csv')
    assert len(assignments) == 231 and not assignments.duplicated(['subject','channel']).any()
    coverage = pd.read_csv(ROOT/'results/paper_trial_inputs/subject_task_coverage.csv')
    memberships, totals, flow = {}, [], []
    for spec in SPECS:
        base = ROOT/'results/supplementary'
        data = pd.read_pickle(base/spec['cache'])
        data = data.loc[data.contrast.eq(spec['key'])]
        candidates = data[['electrode','subject','channel','functional_cluster']].drop_duplicates()
        frozen = pd.read_csv(base/spec['frozen'])
        frozen = frozen[['electrode','subject','channel','functional_cluster']].drop_duplicates()
        assert set(candidates.electrode) == set(frozen.electrode)
        frame = data.loc[data.phase.eq(spec['phase']) & data.time.between(*spec['window'])].copy()
        assert frame.mask_observed.all() and set(frame.level) == set(spec['levels'])
        assert not frame.duplicated(['electrode','level','time']).any()
        assert frame.groupby(['electrode','level']).size().eq(spec['bins']).all()
        masks = frame.groupby(['electrode','level'])['mask'].any().unstack('level')
        assert len(masks) == len(candidates) and not masks.isna().any().any()
        membership = candidates.merge(masks.reset_index(), on='electrode', validate='one_to_one')
        membership = membership.rename(columns={spec['levels'][0]:'response_a', spec['levels'][1]:'response_b'})
        membership['response_class'] = np.select([membership.response_a & ~membership.response_b,
            membership.response_a & membership.response_b, ~membership.response_a & membership.response_b],
            ['A only','Both','B only'], default='Neither')
        saved = pd.read_csv(base/spec['members']).set_index('electrode')
        for r in membership.itertuples():
            assert saved.loc[r.electrode, 'venn_region'] == {'A only':spec['names'][0]+' only',
                'Both':'Both','B only':spec['names'][1]+' only','Neither':'Neither'}[r.response_class]
        native_checks(spec, frame, membership)
        coords = assignments[['subject','channel','x','y','z','functional_cluster']]
        membership = membership.merge(coords, on=['subject','channel','functional_cluster'], validate='one_to_one')
        assert np.isfinite(membership[['x','y','z']]).all().all()
        membership.to_csv(OUT/f"{spec['key']}_electrode_membership.csv", index=False)
        memberships[spec['key']] = membership
        totals.append(count_sets(membership, spec))
        flow.append(denominator_chain(assignments, coverage, spec, membership))
        print(spec['key'], len(membership), membership.subject.nunique(), membership.response_class.value_counts().to_dict(), flush=True)
    counts = pd.concat(totals, ignore_index=True)
    counts.to_csv(OUT/'component_response_counts.csv', index=False)
    chain = pd.concat(flow, ignore_index=True)
    chain.to_csv(OUT/'denominator_chain.csv', index=False)
    subjects = pd.concat([m.assign(comparison=k) for k,m in memberships.items()], ignore_index=True)
    subjects.groupby(['comparison','subject','functional_cluster','response_class']).size().rename('n_sites').reset_index().to_csv(OUT/'subject_component_response_counts.csv',index=False)
    with pd.ExcelWriter(OUT/'Figure3_denominators_and_participants.xlsx') as writer:
        counts.to_excel(writer, sheet_name='Component response counts', index=False)
        chain.to_excel(writer, sheet_name='Denominator chain', index=False)
        subjects.to_excel(writer, sheet_name='Every electrode', index=False)
        for k,m in memberships.items():
            m.loc[:, ['subject','channel','functional_cluster','response_class']].to_excel(writer, sheet_name=k, index=False)
    versions = version_audit(memberships)
    write_json(OUT/'cohort_summary.json', dict(n_nmf_sites=len(assignments), n_nmf_subjects=assignments.subject.nunique(),
        nmf_component_counts=assignments.functional_cluster.value_counts().to_dict(),
        comparisons={k:dict(n_sites=len(m), n_subjects=m.subject.nunique(), response_classes=m.response_class.value_counts().to_dict()) for k,m in memberships.items()},
        no_new_statistics=True, subject_counts_are_unique_within_each_cell_and_not_additive=True,
        response_definition='Any saved native baseline-mask hit in the specified phase window; never a condition-difference test.',
        anatomical_scope='Frozen sound-defined NMF cohort intersected with original task-specific comparison eligibility; not all anatomical insula samples.'))
    return counts, chain, versions


def draw_venns(spec):
    membership = pd.read_csv(OUT/f"{spec['key']}_electrode_membership.csv")
    cm = 1/2.54
    fig, axes = plt.subplots(1,4,figsize=(18*cm,5.5*cm))
    fig.subplots_adjust(left=.025,right=.99,bottom=.18,top=.70,wspace=.30)
    for ax, component in zip(axes, ('all', *CLUSTERS)):
        group = membership if component == 'all' else membership.loc[membership.functional_cluster.eq(component)]
        a, b = set(group.loc[group.response_a,'electrode']), set(group.loc[group.response_b,'electrode'])
        sets = {'10':a-b,'11':a&b,'01':b-a}
        counts = (len(sets['10']),len(sets['01']),len(sets['11']))
        diagram = venn2(subsets=counts,set_labels=spec['names'],set_colors=('#6A5ACD','#000000'),alpha=.30,ax=ax)
        for patch,color in zip(venn2_circles(subsets=counts,ax=ax,linewidth=.8),('#6A5ACD','#000000')):
            patch.set_edgecolor(color)
        for region, sites in sets.items():
            label = diagram.get_label_by_id(region)
            if label is not None:
                p = group.loc[group.electrode.isin(sites),'subject'].nunique()
                label.set_text(f'{len(sites)}\n({p} P)')
                label.set_fontsize(7)
        # Preserve the reviewed white separation around the common region.
        common = diagram.get_patch_by_id('11')
        if common is not None:
            ax.add_patch(PathPatch(common.get_path(), transform=common.get_transform(),
                facecolor='none', edgecolor='white', linewidth=3., joinstyle='round',
                capstyle='round', zorder=2.5))
        for label,color in zip(diagram.set_labels,('#6A5ACD','#000000')):
            if label is not None:
                label.set_fontsize(7);label.set_color(color)
        label = 'All components' if component == 'all' else LABELS[component]
        ax.set_title(f'{label}\n{len(group)} electrodes · {group.subject.nunique()} participants',fontsize=7,
            color='black' if component=='all' else FUNCTION_COLORS[component],pad=10)
        neither = group.loc[group.response_class.eq('Neither')]
        ax.text(.5,-.22,f'Neither detected: {len(neither)} ({neither.subject.nunique()} P)',
                transform=ax.transAxes,ha='center',fontsize=6.5)
    window = 'Stimulus 0–0.6 s' if spec['phase']=='stimulus' else 'Delay 0–1 s'
    fig.suptitle(f"{spec['title']} · {window}\nBaseline-responsive electrodes",fontsize=8,y=.99)
    fig.text(.5,.018,'P = unique participants in each region; participant counts overlap across regions/components.',
             ha='center',fontsize=6)
    path = IMG/f"fig3_{spec['key']}_full_denominator_venn.svg"
    fig.savefig(path,bbox_inches='tight',pad_inches=.03);plt.close(fig)
    return path


def draw_brain_class(spec, region):
    from src.supplementary.selected_brain_contours import draw_selected_brain_contours
    membership = pd.read_csv(OUT/f"{spec['key']}_electrode_membership.csv")
    names = {'A only':spec['names'][0]+' only','Both':'Both conditions','B only':spec['names'][1]+' only','Neither':'Neither detected'}
    points = membership.copy()
    points['significant'] = points.response_class.eq(region)
    selected = points.loc[points.significant]
    title = f'{names[region]} · {len(selected)} electrodes · {selected.subject.nunique()} participants'
    fig = draw_selected_brain_contours(points,title,display_output=OUT/f"{spec['key']}_brain_display"/region.replace(' ','_'))
    path = IMG/f"fig3_{spec['key']}_{region.lower().replace(' ','_')}_brain.svg"
    fig.savefig(path,bbox_inches='tight',pad_inches=.03);plt.close(fig)
    print('Rendered',spec['key'],region,len(selected),'sites',flush=True)
    return path


def draw_brains(spec):
    return [draw_brain_class(spec,r) for r in ('A only','Both','B only','Neither')]


def assemble(spec, venn, brains):
    """Place the real vector exports in one review sheet, retaining editable text."""
    ns = 'http://www.w3.org/2000/svg'
    ET.register_namespace('',ns)
    width = 510.24  # 18 cm in SVG points.
    items = []
    for i,path in enumerate([venn,*brains]):
        source = ET.parse(path).getroot()
        vb = list(map(float, source.attrib['viewBox'].split()))
        ids = {el.attrib['id']:f'p{i}_{el.attrib["id"]}' for el in source.iter() if 'id' in el.attrib}
        for el in source.iter():
            for key,val in list(el.attrib.items()):
                if key=='id':el.attrib[key]=ids[val]
                else:
                    for old,new in ids.items():
                        val=val.replace(f'url(#{old})',f'url(#{new})')
                        if val==f'#{old}':val=f'#{new}'
                    el.attrib[key]=val
        items.append((source,vb))
    top_scale=width/items[0][1][2]
    top_h=items[0][1][3]*top_scale
    card_w=width/2
    scales=[card_w/vb[2] for source,vb in items[1:]]
    row_h=max(vb[3]*s for (source,vb),s in zip(items[1:],scales))
    height=top_h+2*row_h+8
    root=ET.Element(f'{{{ns}}}svg',width=f'{width}pt',height=f'{height}pt',viewBox=f'0 0 {width} {height}')
    ET.SubElement(root,f'{{{ns}}}rect',x='0',y='0',width=str(width),height=str(height),fill='white')
    for i,((source,vb),scale) in enumerate(zip(items,[top_scale,*scales])):
        x=0 if i==0 else ((i-1)%2)*card_w
        y=0 if i==0 else top_h+((i-1)//2)*row_h+4
        group=ET.SubElement(root,f'{{{ns}}}g',transform=f'translate({x},{y}) scale({scale})')
        for child in source:group.append(deepcopy(child))
    path=IMG/f"fig3_{spec['key']}_full_response_characterization.svg"
    ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
    return path
