#!/usr/bin/env python3
"""Run original RT estimators on the approved task-specific sound cohort."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.connectivity.pairwise.config import DEFAULT_DATASETS
from src.main_analysis import COHORT_DIR
from src.paper_trial_inputs import paper_bids_root,VIEW

OUT = ROOT / 'results/fig5'
RT = ROOT / 'results/rt'
SAMPLE = COHORT_DIR / 'sampled_sites_by_subject_task.csv'
TASKS = ('PhonemeSequence', 'LexicalDelay', 'PictureNaming')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matrix():
    sites = pd.read_csv(SAMPLE)
    return sites.loc[sites.task.isin(TASKS), ['task', 'subject']].drop_duplicates().sort_values(['task', 'subject']).to_dict('records')


def fit(index):
    from src.reaction_time.run_insula_rt_ridge import run_subject
    row = matrix()[index]
    os.environ['INSULA_RT_COHORT'] = str(SAMPLE)
    assert row['subject'] != 'D0121'
    records = []
    coverage=pd.read_csv(VIEW/'subject_task_coverage.csv')
    state=coverage.loc[coverage.task.eq(row['task'])&coverage.subject.eq(row['subject'])].iloc[0]
    if state.status=='native_QC_zero_usable':
        (OUT/f'model_{index:03d}.json').write_text(json.dumps(dict(index=index,**row,
            completed=True,status='skipped_native_QC_zero_usable',outputs=[],cohort_sha256=sha(SAMPLE),
            reason='All sampled channels already excluded by original native bad-channel QC; no channel reinstated'),indent=2)+'\n')
        return
    if state.status!='validated':raise RuntimeError(f'Native input not validated: {row}')
    for phases in (('Delay', 'Go'), ('Response',)):
        outputs = run_subject(bids_root=paper_bids_root(row['task']), output_root=RT,
            task=row['task'], subject=row['subject'], phases=phases, description='Repeat',
            band='highgamma', ref='bipolar', atlas='hammers', window_s=.2, step_s=.02,
            n_folds=10, inner_folds=5, n_perm=1000, random_state=42,
            n_jobs=int(os.environ.get('SLURM_CPUS_PER_TASK', '10')), raw_sfreq=2048., rt_min_s=.05,
            overwrite=True)
        if not outputs:
            raise RuntimeError(f'No RT models produced for accepted subject/task: {row}, {phases}')
        records.extend(dict(path=str(p), sha256=sha(p)) for p in outputs)
    (OUT / f'model_{index:03d}.json').write_text(json.dumps(dict(index=index, **row,
        completed=True, outputs=records, cohort_sha256=sha(SAMPLE),
        condition='Repeat', modality='sound', HGA_significance_required=False,
        correction_families=[['Delay', 'Go'], ['Response']]), indent=2) + '\n')


def summarize():
    import h5py
    import numpy as np
    from src.reaction_time.summarize_insula_rt_ridge import summarize_results
    from src.reaction_time.summarize_insula_rt_encoding import summarize_hga_rt_encoding
    os.environ['INSULA_RT_COHORT'] = str(SAMPLE)
    sites = pd.read_csv(SAMPLE)
    records, trials, membership = [], [], []
    assignments = ROOT / 'results/nmf/channel_assignments.csv'
    for i, row in enumerate(matrix()):
        receipt = OUT / f'model_{i:03d}.json'
        if not receipt.exists():
            raise FileNotFoundError(f'Incomplete RT model: {receipt}')
        receipt_data = json.loads(receipt.read_text())
        assert receipt_data['cohort_sha256'] == sha(SAMPLE)
        accepted = set(sites.loc[sites.task.eq(row['task']) & sites.subject.eq(row['subject']), 'site_id'])
        for result in receipt_data['outputs']:
            path = Path(result['path'])
            assert sha(path) == result['sha256']
            with h5py.File(path, 'r') as stream:
                decode = lambda a: [x.decode() if isinstance(x, bytes) else str(x) for x in a]
                channels = decode(stream['channels/channel'][:])
                assert set(channels) <= accepted
                assert stream.attrs['modality'] == 'sound' and stream.attrs['description'] == 'Repeat'
                assert stream.attrs['sample_selection_sha256'] == sha(SAMPLE)
                assert stream.attrs['n_perm'] == 1000 and stream.attrs['n_folds_requested'] == 10
                if row['task'] == 'PictureNaming':
                    assert set(decode(stream['trials/recording'][:])) == {'sound'}
                phase = str(stream.attrs['phase'])
                n = len(stream['trials/rt_log'])
                for ch in channels:
                    membership.append(dict(**row, phase=phase, channel=ch, source_h5=str(path)))
                for name in stream['trials']:
                    value = stream['trials'][name][:]
                    if value.dtype.kind in 'SO':
                        value = decode(value)
                    if len(value) == n:
                        for k, v in enumerate(value):
                            trials.append(dict(**row, phase=phase, row=k, field=name,
                                value=str(v), source_h5=str(path)))
                records.append(dict(**row, phase=phase, n_sites=len(channels), n_trials=n, source_h5=str(path)))
    pd.DataFrame(records).to_csv(OUT / 'input_counts.csv', index=False)
    pd.DataFrame(membership).to_csv(OUT / 'input_membership.csv', index=False)
    pd.DataFrame(trials).to_csv(OUT / 'trial_metadata.csv.gz', index=False)
    coverage, electrodes, clusters = summarize_results(RT, assignments_path=assignments)
    summary = RT / 'summaries'
    summary.mkdir(parents=True, exist_ok=True)
    for name, frame in [('coverage', coverage), ('electrodes', electrodes), ('significant_clusters', clusters)]:
        frame.to_csv(summary / f'{name}.csv', index=False)
    summarize_hga_rt_encoding(RT, assignments_path=assignments,
        bids_roots={task:Path(paper_bids_root(task)) for task in TASKS})
    for phase in ('go', 'delay', 'response'):
        frame = pd.read_csv(summary / f'hga_rt_{phase}_traces.csv')
        if frame.empty:
            continue
        assert 'channel' in frame
        group = frame.groupby(['subject', 'channel', 'functional_cluster', 'time'], as_index=False)[['hga_fast', 'hga_slow']].mean()
        long = group.melt(id_vars=['subject', 'channel', 'functional_cluster', 'time'], value_vars=['hga_fast', 'hga_slow'],
                          var_name='rt_split', value_name='hga')
        summary_curve = long.groupby(['functional_cluster', 'rt_split', 'time']).hga.agg(['mean', 'sem', 'count']).reset_index()
        group.to_csv(OUT / f'{phase}_electrode_traces.csv.gz', index=False)
        summary_curve.to_csv(OUT / f'{phase}_electrode_mean_sem.csv', index=False)
    unassigned = electrodes.loc[electrodes.functional_cluster.isna()]
    unassigned.to_csv(OUT / 'modeled_sites_without_nmf_assignment.csv', index=False)
    (OUT / 'method_validation.json').write_text(json.dumps(dict(passed=True, modality='sound', condition='Repeat',
        model_cohort='accepted task-specific anatomy; no HGA responsiveness requirement',
        n_subject_task_models=len(matrix()), HGA_group_unit='electrode', HGA_error='SEM', Go_retained=True,
        original_Ridge_CV_and_permutation_unchanged=True, n_perm=1000,
        correction_families=[['Delay', 'Go'], ['Response']],
        n_sites_modeled=int(electrodes.channel.nunique()), n_sites_without_nmf_assignment=int(unassigned.channel.nunique()),
        assignment_sha256=sha(assignments), cohort_sha256=sha(SAMPLE)), indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['matrix', 'fit', 'summarize'])
    parser.add_argument('index', type=int, nargs='?')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == 'matrix':
        (OUT / 'job_matrix.json').write_text(json.dumps(matrix(), indent=2))
        print(len(matrix()))
    elif args.mode == 'fit':
        fit(args.index)
    else:
        summarize()


if __name__ == '__main__':
    main()
