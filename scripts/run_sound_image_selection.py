"""Approved Sound > Image Stimulus test; original Decision/Repeat temporal method."""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from ieeg.calc import stats

ROOT = Path('/cwork/ns458/insula-paper')
OUT = ROOT / 'results/supplementary/sound_image_test/stimulus_electrode_selection'
REVIEW = ROOT / 'results/supplementary/decision_repeat_test/delay_electrode_selection/compatibility_review'
CLUSTERS = ('sensory', 'sustain', 'motor')
PARAMS = dict(p_thresh=.05, p_cluster=.05, n_perm=2000, tails=2,
              axis=0, permutation_type='independent', n_jobs=1, seed=42)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def assert_protected():
    for path, expected in json.loads((OUT / 'protected_main_inputs.json').read_text()).items():
        assert sha(path) == expected, f'Protected input changed: {path}'


def install_reviewed_runtime_compatibility():
    """Bind only the two approved keyword changes; never edit installed files."""
    approval = json.loads((REVIEW / 'approval.json').read_text())
    proposal = json.loads((REVIEW / 'proposal.json').read_text())
    assert approval['approved']
    assert sha(REVIEW / 'proposal.json') == approval['proposal_sha256']
    assert sha(REVIEW / 'proposed_function.py') == approval['proposed_function_sha256']
    original = inspect.getsource(stats._comp_by_sort)
    assert hashlib.sha256(original.encode()).hexdigest() == proposal['original_function_sha256']
    proposed = (REVIEW / 'proposed_function.py').read_text()
    assert proposed == original.replace('stable=False', "kind='quicksort'")
    original_ast, repaired_ast = ast.parse(original), ast.parse(proposed)
    changes = 0
    for node in ast.walk(repaired_ast):
        if isinstance(node, ast.keyword) and node.arg == 'kind':
            assert isinstance(node.value, ast.Constant) and node.value.value == 'quicksort'
            node.arg, node.value = 'stable', ast.Constant(value=False)
            changes += 1
    assert changes == 2
    assert ast.dump(original_ast) == ast.dump(repaired_ast)
    namespace = dict(np=np)
    exec(compile(proposed, str(REVIEW / 'proposed_function.py'), 'exec'), namespace)
    compatible = namespace['_comp_by_sort']
    # Explicit quicksort must equal this same NumPy's default unstable sort,
    # including tied values, NaNs, both axes and the original rank convention.
    reference_namespace = dict(np=np)
    exec(original.replace(', stable=False', ''), reference_namespace)
    example = np.array([[1., 0., np.nan], [1., 3., 2.], [0., 3., -1.]])
    for axis in (0, 1):
        np.testing.assert_array_equal(compatible(example, axis=axis),
            reference_namespace['_comp_by_sort'](example, axis=axis))
    stats._comp_by_sort = compatible
    return dict(approval_sha256=sha(REVIEW / 'approval.json'),
                original_function_sha256=proposal['original_function_sha256'],
                proposed_function_sha256=proposal['proposed_function_sha256'],
                ast_only_two_keyword_changes=True, default_sort_equivalence_verified=True)


def contiguous_intervals(times, mask):
    transitions = np.diff(np.r_[False, mask, False].astype(int))
    edges = np.r_[times[0] - (times[1]-times[0])/2,
                  (times[:-1]+times[1:])/2, times[-1]+(times[-1]-times[-2])/2]
    intervals = []
    for start, end in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)):
        intervals.append(dict(start=max(0., float(edges[start])), end=min(.6, float(edges[end])),
            first_significant_time=float(times[start]), last_significant_time=float(times[end-1]),
            n_bins=int(end-start)))
    return intervals


def fit(index):
    assert_protected()
    subjects = pd.read_csv(OUT / 'subjects.csv')
    assert len(subjects) == 24
    subject_row = subjects.iloc[index]
    subject = subject_row.subject
    assert sha(subject_row.input) == subject_row.sha256
    audit = json.loads((OUT / 'input_audit.json').read_text())
    stats_path = inspect.getsourcefile(stats.time_perm_cluster)
    assert sha(stats_path) == audit['installed_stats_sha256']
    compatibility = install_reviewed_runtime_compatibility()
    data = np.load(subject_row.input, allow_pickle=False)
    times, channels = data['time'], data['channel']
    assert times[0] == 0 and times[-1] == .59375 and len(times) == 77
    values, trial_rows = {}, []
    for level in ('sound', 'image'):
        names = data[f'{level}_event_names']
        # PN native event QC has already rejected bad trials. These events
        # have no CORRECT token; retain the original PN trial set.
        correct = np.ones(len(names), dtype=bool)
        values[level] = data[level][correct]
        assert len(values[level]) >= 2, (subject, level, len(values[level]))
        for trial, (name, keep) in enumerate(zip(names, correct)):
            trial_rows.append(dict(subject=subject, condition=level, native_trial_index=trial,
                                   event_name=name, retained_native_QC=bool(keep)))
    destination = OUT / 'subjects' / subject
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trial_rows).to_csv(destination / 'trial_selection.csv', index=False)
    summaries, tables, intervals = [], [], []
    for i, channel in enumerate(channels):
        print(subject, channel, 'testing 2000 permutations', flush=True)
        decision, repeat = values['sound'][:, i], values['image'][:, i]
        mean_decision, mean_repeat = np.nanmean(decision, axis=0), np.nanmean(repeat, axis=0)
        difference = mean_decision - mean_repeat
        # A single channel is passed at a time, so adjacency is time-only.
        # Keep the installed original default t statistic and rank/extent code.
        mask, p_raw = stats.time_perm_cluster(decision, repeat, **PARAMS)
        mask, p_raw = np.asarray(mask, dtype=bool), np.asarray(p_raw, dtype=float)
        assert mask.shape == p_raw.shape == times.shape
        observed = (np.isfinite(decision).sum(axis=0) >= 2) & (np.isfinite(repeat).sum(axis=0) >= 2)
        assert not np.any(observed & ~np.isfinite(p_raw)), (subject, channel, 'nonfinite observed p value')
        # Preserve the original hit rule, then identify positive direction
        # at significant bins, rather than using the whole-window mean.
        selected_mask = observed & mask & (p_raw < .05) & (difference > 0)
        spans = contiguous_intervals(times, selected_mask)
        identity = dict(subject=subject, channel=channel, electrode=str(data['electrode'][i]),
                        functional_cluster=str(data['functional_cluster'][i]))
        assert identity['electrode'] == f'{subject}|{channel}'
        for interval_index, span in enumerate(spans):
            intervals.append(dict(**identity, phase='stimulus', interval=interval_index+1, **span))
        tables.append(pd.DataFrame(dict(**identity, time=times, observed=observed,
            temporal_mask_two_sided=mask, pointwise_p_two_sided=p_raw,
            mean_decision_native_QC_trials=mean_decision, mean_repeat_native_QC_trials=mean_repeat,
            mean_difference=difference, decision_gt_repeat_mask=selected_mask)))
        summaries.append(dict(**identity, n_decision_native=len(data['sound']), n_repeat_native=len(data['image']),
            n_decision_retained=len(decision), n_repeat_retained=len(repeat),
            n_time=len(times), n_observed_bins=int(observed.sum()),
            n_positive_significant_bins=int(selected_mask.sum()), n_intervals=len(spans),
            selected=bool(selected_mask.any()),
            significant_intervals_s='; '.join(f"{span['start']:.4f}–{span['end']:.4f}" for span in spans),
            status='tested' if observed.all() else 'partly_unobserved'))
    pd.DataFrame(summaries).to_csv(destination / 'electrodes.csv', index=False)
    pd.concat(tables, ignore_index=True).to_csv(destination / 'timeperm.csv.gz', index=False)
    pd.DataFrame(intervals, columns=['subject','channel','electrode','functional_cluster','phase','interval',
        'start','end','first_significant_time','last_significant_time','n_bins']).to_csv(destination / 'intervals.csv', index=False)
    assert sha(stats_path) == audit['installed_stats_sha256'], 'Installed pipeline source was modified'
    assert_protected()
    write_json(destination / 'validation.json', dict(completed=True, subject=subject,
        input_sha256=subject_row.sha256, parameters=PARAMS, statistic='ieeg.calc.fast.ttest (original default)',
        trial_rule='Original PN native QC; retained native ListenSpeak trials without accuracy tags', compatibility=compatibility,
        installed_stats_path=stats_path, installed_stats_sha256=sha(stats_path), installed_library_unmodified=True,
        n_tested=len(channels), n_selected=sum(row['selected'] for row in summaries),
        files={name:sha(destination / name) for name in ('electrodes.csv','timeperm.csv.gz','intervals.csv','trial_selection.csv')},
        job_id=os.getenv('SLURM_JOB_ID'), completed_utc=datetime.now(timezone.utc).isoformat()))
    print(subject, 'complete', len(channels), 'tested;', sum(row['selected'] for row in summaries), 'selected', flush=True)


def collect():
    assert_protected()
    subjects = pd.read_csv(OUT / 'subjects.csv')
    all_rows, all_intervals, manifests = [], [], []
    for subject in subjects.subject:
        destination = OUT / 'subjects' / subject
        manifest = json.loads((destination / 'validation.json').read_text())
        assert manifest['completed'] and manifest['parameters'] == PARAMS
        for name, expected in manifest['files'].items():
            assert sha(destination / name) == expected
        all_rows.append(pd.read_csv(destination / 'electrodes.csv'))
        all_intervals.append(pd.read_csv(destination / 'intervals.csv'))
        manifests.append(manifest)
    tested = pd.concat(all_rows, ignore_index=True)
    intervals = pd.concat(all_intervals, ignore_index=True)
    assert len(tested) == 81 and not tested.electrode.duplicated().any()
    assert tested.subject.nunique() == 24
    tested['component_order'] = tested.functional_cluster.map({c:i for i,c in enumerate(CLUSTERS)})
    tested = tested.sort_values(['component_order','subject','channel']).drop(columns='component_order')
    selected = tested.loc[tested.selected].copy()
    assert set(selected.electrode) == set(intervals.electrode)
    counts = []
    for cluster in CLUSTERS:
        group = tested.loc[tested.functional_cluster.eq(cluster)]
        subset = selected.loc[selected.functional_cluster.eq(cluster)]
        counts.append(dict(functional_cluster=cluster, n_tested=len(group), n_selected=len(subset),
            n_subjects_tested=group.subject.nunique(), n_subjects_selected=subset.subject.nunique()))
    counts = pd.DataFrame(counts)
    tested.to_csv(OUT / 'all_tested_electrodes.csv', index=False)
    selected.to_csv(OUT / 'selected_electrodes.csv', index=False)
    counts.to_csv(OUT / 'component_counts.csv', index=False)
    intervals.to_csv(OUT / 'significant_intervals.csv', index=False)
    subject_counts = tested.groupby('subject', as_index=False).agg(n_tested=('electrode','size'), n_selected=('selected','sum'))
    subject_counts.to_csv(OUT / 'subject_counts.csv', index=False)
    lines = ['# Stimulus 0–0.6 s: sound > image electrodes', '',
        'Original per-electrode two-sided time_perm_cluster; 2000 permutations; temporal correction within each electrode.',
        'Direction is positive at significant bins. Interval bounds are sample edges clipped to 0–0.6 s.',
        'Original PN native-QC trials are used for this test. Upper pooled curves remain unchanged.', '',
        '| Component | Tested | Selected | Selected subjects |', '|---|---:|---:|---:|']
    lines += [f'| {r.functional_cluster} | {r.n_tested} | {r.n_selected} | {r.n_subjects_selected} |' for r in counts.itertuples()]
    lines += ['', f'**Total: {len(selected)} of {len(tested)} sites, {selected.subject.nunique()} subjects.**', '',
        '| Subject | Electrode | Component | Significant intervals (s) |', '|---|---|---|---|']
    lines += [f'| {r.subject} | {r.channel} | {r.functional_cluster} | {r.significant_intervals_s} |' for r in selected.itertuples()]
    (OUT / 'selection_summary.md').write_text('\n'.join(lines)+'\n')
    write_json(OUT / 'selection_manifest.json', dict(completed=True, window=[0,.6], parameters=PARAMS,
        statistic='ieeg.calc.fast.ttest (original default)', n_tested=len(tested), n_selected=len(selected),
        n_subjects_tested=24, n_subjects_selected=selected.subject.nunique(),
        component_counts=counts.to_dict(orient='records'),
        trial_rule='Original PN native QC; retained native ListenSpeak trials without accuracy tags',
        selection='positive mean_difference within original temporal mask and original pointwise p<.05',
        correction_scope='Time within each electrode separately; no additional across-electrode correction',
        compatibility_approval_sha256=sha(REVIEW / 'approval.json'),
        input_audit_sha256=sha(OUT / 'input_audit.json'),
        outputs={name:sha(OUT / name) for name in ('all_tested_electrodes.csv','selected_electrodes.csv','significant_intervals.csv','component_counts.csv')},
        installed_library_unmodified=True, main_notebooks_and_NMF_unchanged=True,
        completed_utc=datetime.now(timezone.utc).isoformat()))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int)
    parser.add_argument('--collect', action='store_true')
    args = parser.parse_args()
    if args.collect:
        collect()
    else:
        assert args.index is not None
        fit(args.index)
