#!/usr/bin/env python3
"""Trace significance supporting saved NMF sites outside Figure 1's response set."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pandas as pd

SOURCE = Path('/hpc/group/coganlab/nanlinshi/insula-functional/results')
OUT = Path('/cwork/ns458/insula-paper/results/audit/fig1_nmf_selection_2026-10-17')
TASKS = ['PhonemeSequence', 'LexicalDelay', 'PictureNaming', 'SentenceRep']
WINDOWS = {'stimulus': (0., 1.), 'delay': (0., 1.), 'response': (-.5, .5)}
META_COLS = ['channel', 'subject', 'description', 'phase', 'modality', 'roi', 'mix', 'label']


def scan_file(path):
    frame = pd.read_csv(path, usecols=META_COLS + ['time', 'mask', 'value'])
    frame = frame.loc[frame.channel.isin(TARGETS) & frame['mask'].eq(True)].copy()
    frame['phase'] = frame.phase.astype(str).str.lower().replace({'audio': 'stimulus', 'resp': 'response'})
    frame['task'] = path.parents[2].name
    frame['in_fig_window'] = False
    for phase, (lo, hi) in WINDOWS.items():
        frame.loc[frame.phase.eq(phase) & frame.time.gt(lo) & frame.time.lt(hi), 'in_fig_window'] = True
    records = []
    for keys, group in frame.groupby(META_COLS + ['task'], dropna=False):
        record = dict(zip(META_COLS + ['task'], keys))
        record.update(n_significant_samples=len(group), time_min=float(group.time.min()),
                      time_max=float(group.time.max()),
                      n_positive_significant_samples=int(group.value.gt(0).sum()),
                      n_samples_in_fig_window=int(group.in_fig_window.sum()), source_path=str(path))
        records.append(record)
    return records


if __name__ == '__main__':
    main = pd.read_csv(OUT / 'site_selection_audit.csv')
    TARGETS = set(main.loc[main.saved_nmf & ~main.fig_sig_union & main.fig1_sampled, 'channel'])
    subjects = {c.split('_')[0] for c in TARGETS}
    files = [p for t in TASKS for s in sorted(subjects)
             for p in sorted((SOURCE / 'hga' / t / f'sub-{s}' / 'HGA').glob('*_time.csv'))]
    print(f'Trace {len(TARGETS)} sites in {len(files)} files across all conditions', flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, records in enumerate(pool.map(scan_file, files), 1):
            rows.extend(records)
            if i % 100 == 0:
                print(f'Read {i}/{len(files)}', flush=True)
    evidence = pd.DataFrame(rows)
    evidence.to_csv(OUT / 'nmf_14_significance_evidence.csv', index=False)
    summaries = []
    for channel in sorted(TARGETS):
        one = evidence.loc[evidence.channel.eq(channel)]
        repeat_sound = one.description.eq('Repeat') & one.modality.eq('sound')
        other = ~one.description.eq('Repeat') & one.modality.eq('sound')
        summaries.append({
            'channel': channel,
            'Repeat_sound_Go_significance': bool((repeat_sound & one.phase.eq('go')).any()),
            'Repeat_sound_outside_Fig1_windows': bool((repeat_sound & ~one.phase.eq('go')
                                                      & one.n_samples_in_fig_window.eq(0)).any()),
            'other_condition_sound_significance': bool(other.any()),
            'any_sound_significance': bool(one.modality.eq('sound').any()),
            'evidence_descriptions': '|'.join(sorted(set(one.description))),
            'evidence_phases': '|'.join(sorted(set(one.phase))),
        })
    result = pd.DataFrame(summaries)
    result.to_csv(OUT / 'nmf_14_exception_reasons.csv', index=False)
    print(result.to_string(index=False), flush=True)
