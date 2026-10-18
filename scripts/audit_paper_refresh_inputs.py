#!/usr/bin/env python3
"""Census original inputs without substituting old models or averaged trials."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.connectivity.pairwise.config import DEFAULT_DATASETS
from src.main_analysis import COHORT_DIR
from refresh_figure4 import matrix as decoding_matrix
from refresh_figure5 import matrix as rt_matrix

OUT = ROOT / 'docs/evidence/figures3_6_refresh_2026-10-17'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assignments = pd.read_csv(ROOT / 'results/nmf/channel_assignments.csv')
    sampled = pd.read_csv(COHORT_DIR / 'sampled_sites_by_subject_task.csv')
    rows = []
    condition_map = dict(PhonemeSequence=['Repeat'], LexicalDelay=['Repeat', 'Decision'],
        LexicalNoDelay=['Repeat', 'Decision', 'Passive'], PictureNaming=['Repeat', 'Passive'])
    for task, conditions in condition_map.items():
        if task == 'LexicalNoDelay':
            packaged = Path('/hpc/group/coganlab/nanlinshi/insula-functional/results/hga') / task
            subjects = sorted({p.name.removeprefix('sub-') for p in packaged.glob('sub-*')} & set(assignments.subject))
        else:
            subjects = sorted(set(sampled.loc[sampled.task.eq(task), 'subject']))
        bids = Path(DEFAULT_DATASETS[task])
        for subject in subjects:
            for datatype in ('epoch(band)(zscore)', 'epoch(band)(sig)'):
                folder = bids / 'derivatives/epoch(bipolar)' / f'sub-{subject}' / datatype
                for phase in ('Stimulus', 'Delay', 'Go', 'Response'):
                    if task == 'LexicalNoDelay' and phase in ('Delay', 'Go'):
                        continue
                    for condition in conditions:
                        modalities = ('sound', 'image') if task == 'PictureNaming' and condition == 'Repeat' else ('sound',)
                        for modality in modalities:
                            found = sorted(folder.glob(f'*_proc-{phase}_*desc-{condition}_highgamma.h5'))
                            if task == 'PictureNaming':
                                found = [p for p in found if f'_recording-{modality}_' in p.name]
                            rows.append(dict(task=task, subject=subject, phase=phase, condition=condition,
                                modality=modality, datatype=datatype, expected_folder=str(folder),
                                n_files=len(found), available=bool(found), paths='|'.join(map(str, found))))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'original_epoch_availability.csv', index=False)
    summary = table.groupby(['task', 'datatype'], as_index=False).agg(
        expected_subject_condition_phase_cells=('available', 'size'),
        available_cells=('available', 'sum'), subjects=('subject', 'nunique'))
    summary['missing_cells'] = summary.expected_subject_condition_phase_cells - summary.available_cells
    summary.to_csv(OUT / 'original_epoch_availability_summary.csv', index=False)
    table.loc[~table.available].to_csv(OUT / 'missing_original_inputs.csv', index=False)
    (OUT / 'figure4_job_matrix.json').write_text(json.dumps(decoding_matrix(), indent=2))
    (OUT / 'figure5_job_matrix.json').write_text(json.dumps(rt_matrix(), indent=2))
    # Compare the computational cores, excluding only the explicitly changed IO/config.
    old = Path('/hpc/group/coganlab/nanlinshi/insula-functional')
    unchanged = []
    for rel in ('src/decoding/decoder.py', 'src/decoding/run_decoding_resolved_lda.py',
        'src/decoding/run_cross_condition_resolved_lda.py', 'src/reaction_time/insula_ridge.py',
        'src/decoding/prepare_functional_decoding_dataset.py', 'src/nmf/whole_brain_projection_viz.py'):
        assert (ROOT / rel).read_bytes() == (old / rel).read_bytes(), rel
        unchanged.append(rel)
    def function_ast(path, name):
        tree = ast.parse(path.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        return ast.dump(fn, include_attributes=False)
    rel = 'src/reaction_time/run_insula_rt_ridge.py'
    assert function_ast(ROOT / rel, 'fit_phase') == function_ast(old / rel, 'fit_phase')
    (OUT / 'method_core_validation.json').write_text(json.dumps(dict(passed=True,
        unchanged_files=unchanged, RT_fit_phase_unchanged=True,
        scientific_parameters='Original Figure 3–6 analysis definitions retained; approved changes recorded in MAIN_ANALYSIS.md',
        source_missing_inputs_explicit=True), indent=2))
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
