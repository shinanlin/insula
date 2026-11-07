#!/usr/bin/env python3
"""Prepare and fit the original Figure 4 analyses with frozen new assignments."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.connectivity.pairwise.config import DEFAULT_DATASETS
from src.paper_trial_inputs import paper_bids_root

ROIS = ('Sensory', 'Sustain', 'Motor')
PHASES = ('Stimulus', 'Delay', 'Response')
CONTROLS = ('STGl', 'SMCl', 'MFGl', 'IFGl')
OUT = ROOT / 'results/fig4'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def matrix():
    rows = []
    for roi in ROIS:
        for phase in PHASES:
            rows.append(dict(mode='window', task='PhonemeSequence', roi=roi, phase=phase, description='Repeat'))
    for roi in ROIS:
        for description in ('Repeat', 'Decision'):
            for phase in PHASES:
                rows.append(dict(mode='resolved', task='LexicalDelay', roi=roi, phase=phase, description=description))
    for roi in ROIS:
        for train, test in (('Repeat', 'Decision'), ('Decision', 'Repeat')):
            for phase in ('Delay',):
                rows.append(dict(mode='cross', task='LexicalDelay', roi=roi, phase=phase, train=train, test=test))
    return rows


def prepare(task):
    import h5py
    import pandas as pd
    from src.decoding.prepare_functional_decoding_dataset import run
    target = ROOT / 'results/decoding_inputs' / task
    target.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(bids_root=paper_bids_root(task), bids_task=task,
        assignments=str(ROOT / 'results/nmf/channel_assignments.csv'), reference='bipolar',
        band='highgamma', output_root=str(target),
        intersection_output_root=str(ROOT / 'results/decoding_inputs_intersection' / task),
        manifest=str(OUT / f'{task}_prepare_manifest.csv'), expected_assignment_rows=231,
        dry_run=False, overwrite=True, log_level='INFO')
    run(args)
    copied = []
    original_inputs = Path(DEFAULT_DATASETS[task]) / 'derivatives/decoding(bipolar)'
    for roi in CONTROLS:
        for src in sorted((original_inputs / f'sub-{roi}').glob('**/*.h5')):
            dst = target / src.relative_to(original_inputs)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            digest = sha(src)
            assert digest == sha(dst)
            copied.append(dict(kind='unchanged_anatomical_input', source=str(src), destination=str(dst), sha256=digest))
    from src.paths import resolve_decoding_task_root
    bases = [Path('/hpc/group/coganlab/nanlinshi/insula-functional/results/decoding'),
             Path('/hpc/group/coganlab/nanlinshi/insula/results')]
    seen = set()
    for base in bases:
        task_root = resolve_decoding_task_root(base, task)
        if task_root is None:
            continue
        for roi in CONTROLS:
            for src in sorted((task_root / f'sub-{roi}').glob('**/*.h5')):
                relative = src.relative_to(task_root)
                if str(relative) in seen:
                    continue
                seen.add(str(relative))
                dst = ROOT / 'results/decoding' / task / relative
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                digest = sha(src)
                assert digest == sha(dst)
                copied.append(dict(kind='unchanged_anatomical_result', source=str(src), destination=str(dst), sha256=digest))
    pd.DataFrame(copied).to_csv(OUT / f'{task}_reused_controls.csv', index=False)
    assignment = pd.read_csv(ROOT / 'results/nmf/channel_assignments.csv').set_index('channel')
    members, counts = [], []
    for path in sorted(target.glob('sub-*/*/*.h5')):
        roi = path.parts[-3].removeprefix('sub-')
        with h5py.File(path, 'r') as stream:
            decode = lambda values: [v.decode() if isinstance(v, bytes) else str(v) for v in values]
            channels = decode(stream['channel'][:])
            labels = decode(stream['label'][:])
            trials = decode(stream['trial'][:]) if 'trial' in stream else []
            groups = decode(stream['condition'][:])
            if roi in ROIS:
                assert all(ch in assignment.index for ch in channels)
                assert set(assignment.loc[channels, 'functional_cluster']) == {roi.lower()}
                assert stream.attrs['assignment_sha256'] == sha(ROOT / 'results/nmf/channel_assignments.csv')
                assert stream['X'].shape[1] == len(channels)
            phase = path.name.split('_proc-')[1].split('_')[0]
            description = path.name.split('_desc-')[1].split('_')[0]
            feature = path.parent.name
            count = dict(path=str(path), task=task, roi=roi, phase=phase, description=description, feature=feature,
                n_sites=len(channels), n_subjects=len({ch.split('_')[0] for ch in channels}),
                n_pooled_trials=len(labels), n_items=len(set(groups)),
                class_counts=json.dumps({label: labels.count(label) for label in sorted(set(labels))}),
                sha256=sha(path), assignment_sha256=sha(ROOT / 'results/nmf/channel_assignments.csv') if roi in ROIS else '')
            counts.append(count)
            for ch in channels:
                members.append(dict(task=task, roi=roi, phase=phase, description=description, feature=feature,
                    subject=ch.split('_')[0], channel=ch, path=str(path)))
            if roi in ROIS:
                trial_path = OUT / 'trials' / path.relative_to(target).with_suffix('.json')
                trial_path.parent.mkdir(parents=True, exist_ok=True)
                trial_path.write_text(json.dumps(dict(trial=trials, item=groups, label=labels), indent=2))
    pd.DataFrame(counts).to_csv(OUT / f'{task}_input_counts.csv', index=False)
    pd.DataFrame(members).to_csv(OUT / f'{task}_input_membership.csv', index=False)
    print(task, 'prepared files', len(counts), 'reused controls', len(copied), flush=True)


def fit(index, verify_previous=False):
    import h5py
    row = matrix()[index]
    if row['mode'] == 'window':
        # The approved phonetic specification has one standard source engine
        # for stage and time-resolved analyses. Historical frozen-fold fits
        # are retained as diagnostics, never invoked by a current refresh.
        if verify_previous:
            raise ValueError('Old windows/null are historical; formal validation uses the frozen input identities')
        command = [sys.executable, '-u', '-m', 'src.decoding.figure4_formal',
            'fit', '--mode', 'stage', '--index', str(index),
            '--n-jobs', os.environ.get('SLURM_CPUS_PER_TASK', '1')]
        subprocess.run(command, cwd=ROOT, check=True)
        return
    task, roi, phase = row['task'], row['roi'], row['phase']
    os.environ['INSULA_DECODING_INPUT_ROOT'] = str(ROOT / 'results/decoding_inputs' / task)
    feature = 'lexicality'
    root = Path(os.environ['INSULA_DECODING_INPUT_ROOT']) / f'sub-{roi}' / feature
    descriptions = [row['description']] if row['mode'] != 'cross' else [row['train'], row['test']]
    inputs = []
    for description in descriptions:
        recording = '_recording-1' if task == 'PhonemeSequence' else ''
        path = root / f'sub-{roi}_task-{task}_proc-{phase}{recording}_desc-{description}_highgamma.h5'
        if not path.is_file():
            raise FileNotFoundError(f'Expected approved Figure 4 input missing: {path}')
        with h5py.File(path, 'r') as stream:
            assert stream.attrs['assignment_sha256'] == sha(ROOT / 'results/nmf/channel_assignments.csv')
        inputs.append(dict(path=str(path), sha256=sha(path)))
    jobs = os.environ.get('SLURM_CPUS_PER_TASK', '16')
    common = ['--bids_root', paper_bids_root(task), '--subject', roi, '--ref', 'bipolar',
              '--phase', phase, '--band', 'highgamma', '--datatype', feature, '--n_folds', '5', '--n_jobs', jobs]
    if row['mode'] == 'resolved':
        command = [sys.executable, '-u', 'src/decoding/run_decoding_resolved_lda.py', *common,
            '--description', row['description'], '--n_bins', '5', '--window', '0.30', '--step', '0.03',
            '--n_perm', '5000', '--cv_scheme', 'group']
    else:
        command = [sys.executable, '-u', 'src/decoding/run_cross_condition_resolved_lda.py', *common,
            '--train_on', row['train'], '--test_on', row['test'], '--n_bins', '5', '--window', '0.30', '--step', '0.03',
            '--n_perm', '5000', '--cv_scheme', 'group']
    record = dict(index=index, **row, inputs=inputs, command=command,
        assignment_sha256=sha(ROOT / 'results/nmf/channel_assignments.csv'))
    print(json.dumps(record, indent=2), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    record['completed'] = True
    (OUT / f'model_{index:02d}.json').write_text(json.dumps(record, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'fit'])
    parser.add_argument('value')
    parser.add_argument('--verify-previous', action='store_true',
                        help='Validate all observed fold scores against the archived pre-refresh results')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == 'prepare':
        prepare(args.value)
    else:
        fit(int(args.value), verify_previous=args.verify_previous)


if __name__ == '__main__':
    main()
