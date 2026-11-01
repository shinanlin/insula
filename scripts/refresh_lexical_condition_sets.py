"""Author-approved full Lexical Decision/Repeat cohort; run science on Slurm."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

ROOT = Path('/cwork/ns458/insula-paper')
BASE = ROOT / 'results/supplementary/fig3_lexical_delay_condition_sets'
OUT = BASE / 'full_cohort_2026-10-27'
AUDIT = ROOT / 'results/audit/lexical_condition_sets_full_2026-10-27'
COHORT = ROOT / 'results/audit/fig3_all_anatomical_denominators_2026-10-23'
OLD = ROOT / 'results/supplementary/decision_repeat_test/delay_electrode_selection'
NB = ROOT / 'vizpub/supplementary/fig3_lexical_delay_condition_sets.ipynb'
IMG = ROOT / 'img/supplementary/fig3_lexical_delay_condition_sets'
PHASES = {'stimulus': ('Stimulus', (-.5, .6)), 'delay': ('Delay', (0., 1.)), 'response': ('Response', (-.5, 1.))}
GROUPS = ('Decision only', 'Both', 'Repeat only', 'Neither')
COMPONENTS = ('sensory', 'sustain', 'motor', 'Unassigned')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def protected():
    hashes = json.loads((AUDIT / 'protected.json').read_text())
    changed = [p for p, h in hashes.items() if not Path(p).is_file() or sha(p) != h]
    assert not changed, f'Protected input changed: {changed}'
    return len(hashes)


def setup():
    from ieeg.calc import stats
    from scripts.run_decision_repeat_delay_selection import PARAMS
    AUDIT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if not (AUDIT / 'protected.json').exists():
        paths = list((ROOT / 'vizpub').glob('fig[1-6].ipynb'))
        for folder in ('img/fig1', 'img/fig2', 'results/nmf'):
            paths += [p for p in (ROOT / folder).rglob('*') if p.is_file()]
        paths += list(OLD.glob('*.json')) + list(OLD.glob('*.csv'))
        paths += list((OLD / 'subjects').glob('*/*.csv*')) + list((OLD / 'subjects').glob('*/*.json'))
        paths += [COHORT / 'LexicalDelay_all_anatomical_sites.csv', COHORT / 'native_product_sources.csv',
                  ROOT / 'docs/evidence/sound_only_cohort_2026-10-17/sampled_sites_by_subject_task.csv',
                  ROOT / 'results/supplementary/context_comparisons/waveforms.pkl']
        write(AUDIT / 'protected.json', {str(p): sha(p) for p in sorted(set(paths))})
        archive = AUDIT / 'before'
        archive.mkdir()
        shutil.copy2(NB, archive / NB.name)
        for folder, target in ((IMG, archive / 'img'), (BASE, archive / 'results')):
            target.mkdir()
            for p in folder.glob('*'):
                if p.is_file():
                    shutil.copy2(p, target / p.name)
    protected()
    audited = pd.read_csv(COHORT / 'LexicalDelay_all_anatomical_sites.csv')
    sites = audited.loc[audited.eligible].copy()
    assert len(audited) == 214 and len(sites) == 187 and sites.subject.nunique() == 35
    assert not sites.native_qc_bad.any() and sites.paired_statistics_complete.all()
    assert sites.electrode.is_unique
    assignments = pd.read_csv(ROOT / 'results/nmf/channel_assignments.csv')
    assignments = assignments[['subject', 'channel', 'functional_cluster', 'x', 'y', 'z', 'x_native', 'y_native', 'z_native']]
    sites = sites.merge(assignments, on=['subject', 'channel'], how='left', validate='one_to_one')
    assert sites.functional_cluster.notna().sum() == 135
    sites['functional_cluster'] = sites.functional_cluster.fillna('Unassigned')
    # Accepted, task-specific coordinates for non-NMF sites; retain all existing
    # frozen assignment coordinates for already assigned sites.
    coord_source = ROOT / 'results/audit/sound_only_cohort_2026-10-17/all_coordinate_rows.pkl.gz'
    coordinates = pd.read_pickle(coord_source)
    coordinates = coordinates.loc[coordinates.task.eq('LexicalDelay')]
    coord_rows = []
    for row in sites.itertuples():
        candidate = coordinates.loc[coordinates.subject.eq(row.subject) & coordinates.channel.eq(row.channel)]
        assert len(candidate), (row.subject, row.channel, 'No frozen task coordinates')
        xyz = candidate[['x', 'y', 'z']].to_numpy(float)
        assert np.isfinite(xyz).all() and np.allclose(xyz, xyz[0], atol=1e-5, rtol=0), row.channel
        if row.functional_cluster != 'Unassigned':
            # The fixed NMF table averages template locations across tasks.
            # Establish physical identity in native space and preserve its
            # already accepted template location verbatim for old sites.
            native_xyz = candidate[['x_native','y_native','z_native']].to_numpy(float)
            np.testing.assert_allclose([row.x_native,row.y_native,row.z_native],native_xyz[0],atol=1e-5,rtol=0)
            chosen = [row.x, row.y, row.z]
            origin = 'Frozen NMF assignment coordinates; native task identity verified'
        else:
            chosen = xyz[0]
            origin = 'Frozen task-specific accepted coordinate records; no NMF label assigned'
        coord_rows.append(dict(electrode=row.electrode, x=chosen[0], y=chosen[1], z=chosen[2],
                               coordinate_origin=origin, coordinate_source=str(coord_source)))
    sites = sites.drop(columns=['x', 'y', 'z']).merge(pd.DataFrame(coord_rows), on='electrode', validate='one_to_one')
    assert np.isfinite(sites[['x', 'y', 'z']]).all().all() and sites.x.ne(0).all()
    old = pd.read_csv(OLD / 'all_tested_electrodes.csv')
    old_manifest = json.loads((OLD / 'selection_manifest.json').read_text())
    assert old_manifest['parameters'] == PARAMS
    assert sha(OLD / 'all_tested_electrodes.csv') == old_manifest['outputs']['all_tested_electrodes.csv']
    assert set(old.electrode) <= set(sites.electrode)
    sites['test_origin'] = np.where(sites.electrode.isin(old.electrode), 'Reused original two-sided test', 'New test, identical parameters')
    assert sites.test_origin.str.startswith('New').sum() == 64
    sites['baseline_class'] = np.select([sites.decision_responsive & ~sites.repeat_responsive,
            sites.decision_responsive & sites.repeat_responsive,
            ~sites.decision_responsive & sites.repeat_responsive], GROUPS[:3], default='Neither')
    assert sites.baseline_class.value_counts().to_dict() == {'Neither':109, 'Both':48, 'Decision only':19, 'Repeat only':11}
    sites = sites.sort_values(['subject', 'channel']).reset_index(drop=True)
    sites.to_csv(OUT / 'cohort.csv', index=False)
    subjects = sites.groupby('subject', sort=True).agg(n_sites=('electrode','size'),
            n_new_tests=('test_origin', lambda v: int(v.str.startswith('New').sum()))).reset_index()
    subjects.to_csv(OUT / 'subjects.csv', index=False)
    write(AUDIT / 'run_manifest.json', dict(approved_user_request='Update 187 sites, test remaining 64 in both directions, integrate supplementary notebook; freeze Fig1/Fig2/NMF',
          parameters=PARAMS, trial_rule='Original exact CORRECT-only remark for per-electrode tests',
          waveform_trial_rule='All retained native epochs, exactly as the previous waveform cache',
          window=[0,1], n_sites=187, n_subjects=35, reused_tests=123, new_tests=64,
          n_assigned=135, n_unassigned=52, coordinate_source=str(coord_source),
          installed_stats_path=inspect.getsourcefile(stats.time_perm_cluster),
          installed_stats_sha256=sha(inspect.getsourcefile(stats.time_perm_cluster)),
          input_cohort_sha256=sha(COHORT / 'LexicalDelay_all_anatomical_sites.csv'),
          cohort_sha256=sha(OUT / 'cohort.csv'), old_manifest_sha256=sha(OLD / 'selection_manifest.json'),
          job_id=os.environ['SLURM_JOB_ID'], started_utc=datetime.now(timezone.utc).isoformat()))
    print(subjects.to_string(index=True), flush=True)


def phase_product(subject, phase, condition):
    val_path = ROOT / 'results/native_preprocessing_restore_2026-10-18/LexicalDelay' / subject / 'output_validation.json'
    val = json.loads(val_path.read_text())
    assert val['validation_passed']
    products = [p for p in val['products'] if (p['phase'],p['modality'],p['condition']) == (phase,'sound',condition)]
    assert len(products) == 1, (subject, phase, condition)
    return products[0], val_path


def worker(index):
    import h5py
    import mne
    from ieeg.calc import stats
    from scripts.run_decision_repeat_delay_selection import PARAMS, install_reviewed_runtime_compatibility, contiguous_intervals
    protected()
    manifest = json.loads((AUDIT / 'run_manifest.json').read_text())
    assert sha(inspect.getsourcefile(stats.time_perm_cluster)) == manifest['installed_stats_sha256']
    compatibility = install_reviewed_runtime_compatibility()
    subjects = pd.read_csv(OUT / 'subjects.csv')
    subject = subjects.iloc[index].subject
    sites = pd.read_csv(OUT / 'cohort.csv')
    group = sites.loc[sites.subject.eq(subject)].copy()
    dest = OUT / 'subjects' / subject
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'completed.json').exists():
        previous = json.loads((dest / 'completed.json').read_text())
        assert previous['cohort_sha256'] == sha(OUT / 'cohort.csv')
        assert all(sha(dest / f) == h for f,h in previous['outputs'].items())
        print(subject, 'verified existing completion', flush=True)
        return
    old_path = OLD / 'subjects' / subject / 'timeperm.csv.gz'
    old = pd.read_csv(old_path) if old_path.exists() else pd.DataFrame()
    old_inputs = np.load(OLD / 'inputs' / f'{subject}.npz', allow_pickle=False) if (OLD / 'inputs' / f'{subject}.npz').exists() else None
    traces, sources, tested_tables, trial_rows = [], [], [], []
    for phase, (native_phase, limits) in PHASES.items():
        native = {}
        for condition in ('Decision', 'Repeat'):
            product, val_path = phase_product(subject, native_phase, condition)
            path = Path(product['zscore_path'])
            epochs = mne.read_epochs(path, preload=False, verbose='error')
            channels = group.channel.tolist()
            assert set(channels) <= set(epochs.ch_names) and not set(channels) & set(epochs.info['bads'])
            inv = {v:k for k,v in epochs.event_id.items()}
            names = np.asarray([inv[int(n)] for n in epochs.events[:,2]])
            assert not any('INCORRECT' in name.split('/') or 'bad' in name.lower().split('/') for name in names)
            keep = (epochs.times >= limits[0]) & (epochs.times <= limits[1])
            times = epochs.times[keep]
            values = epochs.get_data(picks=channels, copy=True, verbose='error')[:,:,keep]
            assert not np.isinf(values).any()
            mean = np.nanmean(values, axis=0)
            with h5py.File(product['statistics_path'],'r') as h:
                ch_names = [v.decode() if isinstance(v,bytes) else str(v) for v in h['ch_names'][:]]
                assert ch_names == epochs.ch_names
                mask = np.asarray(h['mask'][:], dtype=bool)[[ch_names.index(ch) for ch in channels]][:,keep]
            for i,row in enumerate(group.itertuples()):
                traces.append(pd.DataFrame(dict(electrode=row.electrode, subject=subject, channel=row.channel,
                    phase=phase, condition=condition, time=times, value=mean[i], baseline_mask=mask[i],
                    n_finite_trials=np.isfinite(values[:,i]).sum(axis=0))))
                if phase == 'delay':
                    assert bool(mask[i].any()) == bool(getattr(row, f'{condition.lower()}_responsive')), (subject,row.channel,'Baseline classification drift')
            sources.append(dict(subject=subject,phase=phase,condition=condition,modality='sound',
                    epoch_path=str(path),epoch_sha256=sha(path),statistics_path=product['statistics_path'],
                    statistics_sha256=sha(product['statistics_path']), validation=str(val_path),validation_sha256=sha(val_path),n_trials=len(values)))
            if phase == 'delay':
                correct = np.asarray(['/'.join(name.split('/')[4:]) == 'CORRECT' for name in names])
                assert correct.sum() >= 2
                native[condition] = values[correct]
                for trial,(name,retain) in enumerate(zip(names,correct)):
                    trial_rows.append(dict(subject=subject,condition=condition,native_trial_index=trial,event_name=name,retained_CORRECT=bool(retain)))
                if old_inputs is not None:
                    np.testing.assert_array_equal(names, old_inputs[f'{condition}_event_names'])
                    for i,ch in enumerate(channels):
                        match = np.flatnonzero(old_inputs['channel'] == ch)
                        if len(match):
                            np.testing.assert_array_equal(times, old_inputs['time'])
                            np.testing.assert_allclose(values[:,i], old_inputs[condition][:,match[0]], atol=0,rtol=0,equal_nan=True)
            del epochs, values
        if phase != 'delay':
            continue
        assert len(times) == 129 and times[0] == 0 and times[-1] == 1
        for i,row in enumerate(group.itertuples()):
            if row.test_origin.startswith('Reused'):
                saved = old.loc[old.electrode.eq(row.electrode)].sort_values('time').copy()
                assert len(saved) == 129
                np.testing.assert_array_equal(saved.time, times)
                np.testing.assert_allclose(saved.mean_difference,
                        np.nanmean(native['Decision'][:,i],axis=0)-np.nanmean(native['Repeat'][:,i],axis=0),atol=1e-12,rtol=0)
                table = saved
            else:
                print(subject,row.channel,'testing 2000 original two-sided permutations',flush=True)
                decision, repeat = native['Decision'][:,i], native['Repeat'][:,i]
                mask, p = stats.time_perm_cluster(decision, repeat, **PARAMS)
                observed = (np.isfinite(decision).sum(axis=0)>=2) & (np.isfinite(repeat).sum(axis=0)>=2)
                assert not np.any(observed & ~np.isfinite(p))
                table = pd.DataFrame(dict(electrode=row.electrode,subject=subject,channel=row.channel,
                    functional_cluster=row.functional_cluster,time=times,observed=observed,
                    temporal_mask_two_sided=np.asarray(mask,dtype=bool),pointwise_p_two_sided=p,
                    mean_decision_correct_trials=np.nanmean(decision,axis=0),
                    mean_repeat_correct_trials=np.nanmean(repeat,axis=0),
                    mean_difference=np.nanmean(decision,axis=0)-np.nanmean(repeat,axis=0)))
            valid = table.observed & table.temporal_mask_two_sided & table.pointwise_p_two_sided.lt(.05)
            table['decision_gt_repeat_mask'] = valid & table.mean_difference.gt(0)
            table['repeat_gt_decision_mask'] = valid & table.mean_difference.lt(0)
            table['test_origin'] = row.test_origin
            table['n_decision_correct'] = len(native['Decision'])
            table['n_repeat_correct'] = len(native['Repeat'])
            tested_tables.append(table)
    pd.concat(traces,ignore_index=True).to_csv(dest / 'waveforms.csv.gz',index=False)
    pd.concat(tested_tables,ignore_index=True).to_csv(dest / 'timeperm.csv.gz',index=False)
    pd.DataFrame(sources).to_csv(dest / 'native_sources.csv',index=False)
    pd.DataFrame(trial_rows).to_csv(dest / 'trial_selection.csv',index=False)
    protected()
    write(dest / 'completed.json',dict(subject=subject,n_sites=len(group),n_new_tests=int(group.test_origin.str.startswith('New').sum()),
          cohort_sha256=sha(OUT/'cohort.csv'),parameters=PARAMS,compatibility=compatibility,
          outputs={f:sha(dest/f) for f in ('waveforms.csv.gz','timeperm.csv.gz','native_sources.csv','trial_selection.csv')},
          job_id=os.environ['SLURM_JOB_ID'],completed_utc=datetime.now(timezone.utc).isoformat()))
    print(subject,'complete',flush=True)


def collect():
    from scripts.run_decision_repeat_delay_selection import PARAMS, contiguous_intervals
    protected()
    sites = pd.read_csv(OUT / 'cohort.csv')
    waveforms, tests, sources = [], [], []
    for subject in sorted(sites.subject.unique()):
        dest = OUT / 'subjects' / subject
        done = json.loads((dest/'completed.json').read_text())
        assert done['parameters'] == PARAMS and done['cohort_sha256'] == sha(OUT/'cohort.csv')
        assert all(sha(dest/f)==h for f,h in done['outputs'].items())
        waveforms.append(pd.read_csv(dest/'waveforms.csv.gz'))
        tests.append(pd.read_csv(dest/'timeperm.csv.gz'))
        sources.append(pd.read_csv(dest/'native_sources.csv'))
    tests = pd.concat(tests,ignore_index=True)
    waves = pd.concat(waveforms,ignore_index=True)
    assert tests.electrode.nunique()==187 and len(tests)==187*129
    assert not tests.duplicated(['electrode','time']).any() and tests.observed.all()
    assert not waves.duplicated(['electrode','phase','condition','time']).any()
    intervals, rows = [], []
    for electrode, group in tests.groupby('electrode',sort=False):
        group = group.sort_values('time')
        row = sites.loc[sites.electrode.eq(electrode)].iloc[0].to_dict()
        row['n_decision_correct'] = int(group.n_decision_correct.iloc[0])
        row['n_repeat_correct'] = int(group.n_repeat_correct.iloc[0])
        for direction, column in (('Decision > Repeat','decision_gt_repeat_mask'),('Repeat > Decision','repeat_gt_decision_mask')):
            field = 'decision_gt_repeat' if direction.startswith('Decision') else 'repeat_gt_decision'
            spans = contiguous_intervals(group.time.to_numpy(),group[column].to_numpy(bool))
            row[field] = bool(spans)
            row[field+'_intervals'] = '; '.join(f"{s['start']:.4f}–{s['end']:.4f}" for s in spans)
            row[field+'_n_bins'] = int(group[column].sum())
            intervals.extend(dict(electrode=electrode,subject=row['subject'],channel=row['channel'],functional_cluster=row['functional_cluster'],direction=direction,interval=i+1,**s) for i,s in enumerate(spans))
        row['difference_class'] = ('Both directions' if row['decision_gt_repeat'] and row['repeat_gt_decision'] else
                'Decision > Repeat' if row['decision_gt_repeat'] else 'Repeat > Decision' if row['repeat_gt_decision'] else 'No detected difference')
        rows.append(row)
    table = pd.DataFrame(rows)
    # Existing results must remain exactly the same positive-site set.
    previous = set(pd.read_csv(OLD/'selected_electrodes.csv').electrode)
    reused = table.test_origin.str.startswith('Reused')
    assert set(table.loc[reused & table.decision_gt_repeat,'electrode']) == previous
    table.to_csv(OUT/'electrodes.csv',index=False)
    tests.to_csv(OUT/'timeperm.csv.gz',index=False)
    waves.to_csv(OUT/'waveforms.csv.gz',index=False)
    pd.concat(sources,ignore_index=True).to_csv(OUT/'native_sources.csv',index=False)
    pd.DataFrame(intervals).to_csv(OUT/'significant_intervals.csv',index=False)
    count_rows = []
    for name in GROUPS:
        group = table.loc[table.baseline_class.eq(name)]
        count_rows.append(dict(set=name,n_electrodes=len(group),n_participants=group.subject.nunique(),participants=';'.join(sorted(group.subject.unique()))))
    for name, flag in (('Decision > Repeat','decision_gt_repeat'),('Repeat > Decision','repeat_gt_decision')):
        group = table.loc[table[flag]]
        count_rows.append(dict(set=name,n_electrodes=len(group),n_participants=group.subject.nunique(),participants=';'.join(sorted(group.subject.unique()))))
        group.to_csv(OUT/(flag+'_electrodes.csv'),index=False)
    counts = pd.DataFrame(count_rows)
    counts.to_csv(OUT/'set_counts.csv',index=False)
    cross = table.groupby(['baseline_class','difference_class'],observed=True).agg(n_electrodes=('electrode','size'),n_participants=('subject','nunique')).reset_index()
    cross.to_csv(OUT/'response_difference_counts.csv',index=False)
    component_counts = []
    for set_name, group in [('All 187',table)] + [(n,table.loc[table.baseline_class.eq(n)]) for n in GROUPS] + [(n,table.loc[table[f]]) for n,f in [('Decision > Repeat','decision_gt_repeat'),('Repeat > Decision','repeat_gt_decision')]]:
        for c in COMPONENTS:
            part = group.loc[group.functional_cluster.eq(c)]
            population = table.loc[table.functional_cluster.eq(c)]
            component_counts.append(dict(set=set_name,component=c,n_electrodes=len(part),n_participants=part.subject.nunique(),
                component_denominator=len(population),set_denominator=len(group),
                fraction_within_component=len(part)/len(population),fraction_of_set=len(part)/len(group) if len(group) else np.nan))
    pd.DataFrame(component_counts).to_csv(OUT/'component_composition.csv',index=False)
    subject_counts = table.groupby('subject').agg(n_sites=('electrode','size'),decision_gt_repeat=('decision_gt_repeat','sum'),repeat_gt_decision=('repeat_gt_decision','sum')).reset_index()
    subject_counts.to_csv(OUT/'subject_counts.csv',index=False)
    # Per-site values are retained so group waveforms can be regenerated for
    # any displayed set without rereading single-trial recordings.
    deltas = tests.groupby('electrode').agg(delay_mean_difference=('mean_difference','mean'),n_decision_correct=('n_decision_correct','first'),n_repeat_correct=('n_repeat_correct','first')).reset_index()
    deltas.merge(table[['electrode','subject','functional_cluster','baseline_class','difference_class']],on='electrode',validate='one_to_one').to_csv(OUT/'descriptive_delay_effects.csv',index=False)
    summary = dict(n_sites=187,n_subjects=35,n_reused_tests=123,n_new_tests=64,n_assigned=135,n_unassigned=52,
        parameters=PARAMS,window=[0,1],counts=counts.to_dict('records'),
        n_both_directions=int((table.decision_gt_repeat & table.repeat_gt_decision).sum()),
        correction='Time within each electrode; original two-sided test, no additional across-electrode correction',
        direction_rule='Observed finite bins, original temporal mask and pointwise p<.05; sign of Decision minus Repeat',
        waveform_rule='Retained native trial means; equally weighted bipolar electrodes, SEM over electrode means',
        component_summary='Descriptive only; 52 Unassigned sites retained, no NMF refit or new component inference',
        original_123_positive_set_unchanged=True,protected_files_unchanged=protected(),
        job_id=os.environ['SLURM_JOB_ID'],completed_utc=datetime.now(timezone.utc).isoformat())
    write(OUT/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    assert os.environ.get('SLURM_JOB_ID'), 'Run through Slurm'
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['setup','worker','collect'])
    p.add_argument('--index',type=int)
    args=p.parse_args()
    if args.mode=='setup': setup()
    elif args.mode=='worker': worker(args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID']))
    else: collect()
