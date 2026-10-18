#!/usr/bin/env python3
"""Focused checks of approved sample selection and electrode HGA weighting."""
from __future__ import annotations
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reaction_time.insula_rt_data import load_strict_insula_parcellation
from src.reaction_time.summarize_insula_rt_encoding import aligned_fast_slow_traces, median_split_traces


def main():
    (ROOT / 'results/fig5').mkdir(parents=True, exist_ok=True)
    paths = [ROOT / rel for rel in (
        'src/paths.py', 'src/decoding/run_decoding.py', 'src/decoding/pooled_io.py',
        'src/reaction_time/insula_rt_data.py', 'src/reaction_time/run_insula_rt_ridge.py',
        'src/reaction_time/summarize_insula_rt_encoding.py', 'scripts/refresh_figure4.py',
        'scripts/refresh_figure5.py', 'scripts/execute_paper_notebook.py')]
    for path in paths:
        compile(path.read_text(), str(path), 'exec')
    for number in range(3, 7):
        nb = json.loads((ROOT / f'vizpub/fig{number}.ipynb').read_text())
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                source = '\n'.join(l for l in ''.join(cell['source']).splitlines() if not l.startswith('%'))
                compile(source, f'fig{number}:cell{i}', 'exec')
    # Accepted task-specific anatomy includes both responsive and nonresponsive
    # sites; a third formerly strict-AIC contact is absent from the new cohort.
    with tempfile.TemporaryDirectory(dir=ROOT / 'results/fig5') as tmp:
        base = Path(tmp)
        folder = base / 'derivatives/parcellation/sub-D0001/bipolar'
        folder.mkdir(parents=True)
        pd.DataFrame(dict(name=['D0001_A1-2', 'D0001_A2-3', 'D0001_A3-4'],
            roi=['AIC'] * 3, hemi=['L'] * 3, x=[1., 2., 3.], y=[0.] * 3, z=[0.] * 3)).to_csv(folder / 'sub-D0001_hammers.csv', index=False)
        cohort = base / 'cohort.csv'
        pd.DataFrame(dict(subject=['D0001'] * 2, task=['PhonemeSequence'] * 2,
            site_id=['D0001_A1-2', 'D0001_A2-3'], confirmed_responsive=[True, False])).to_csv(cohort, index=False)
        prior = os.environ.get('INSULA_RT_COHORT')
        os.environ['INSULA_RT_COHORT'] = str(cohort)
        actual = load_strict_insula_parcellation(base, subject='D0001', task='PhonemeSequence')
        assert set(actual.channel) == {'D0001_A1-2', 'D0001_A2-3'}
        if prior is None:
            os.environ.pop('INSULA_RT_COHORT')
        else:
            os.environ['INSULA_RT_COHORT'] = prior
    # Verify separate electrode traces retain the exact original common trial
    # filter and within-subject RT split, including missing-data behavior.
    rt = np.log(np.linspace(.2, 1., 20))
    array = np.arange(20 * 2 * 3, dtype=float).reshape(20, 2, 3) / 10
    array[3, 0, 1] = np.nan
    data = SimpleNamespace(X=array, times=np.arange(3.),
        trial_meta=pd.DataFrame({'rt_log': rt}),
        channel_meta=pd.DataFrame({'channel': ['a', 'b']}))
    traces = pd.DataFrame(aligned_fast_slow_traces(data, task='PS', subject='D0001',
        retained_clusters=['motor'], by_cluster={'motor': ['a', 'b']}, min_trials=16))
    legacy_fast, legacy_slow, _, _ = median_split_traces(array.mean(axis=1), rt)
    average = traces.groupby('time')[['hga_fast', 'hga_slow']].mean()
    assert traces.channel.nunique() == 2
    assert np.allclose(average.hga_fast, legacy_fast) and np.allclose(average.hga_slow, legacy_slow)
    # Run the notebook's actual reshaping function: duplicate tasks at one
    # electrode cannot create a second observation in the group SEM.
    nb = json.loads((ROOT / 'vizpub/fig5.ipynb').read_text())
    tree = ast.parse(''.join(nb['cells'][9]['source']))
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == '_hga_rt_long')
    namespace = {'pd': pd}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<notebook-hga>', 'exec'), namespace)
    frame = pd.DataFrame(dict(task=['PS', 'LD', 'PS', 'PS'], subject=['s1', 's1', 's1', 's2'],
        channel=['a', 'a', 'b', 'c'], time=[0.] * 4, hga_fast=[0., 2., 4., 10.], hga_slow=[2., 4., 6., 12.]))
    actual = namespace['_hga_rt_long'](frame)
    fast = actual.loc[actual.rt_split.eq('fast'), 'hga'].to_numpy()
    assert len(fast) == 3 and np.allclose(np.sort(fast), [1., 4., 10.])
    assert np.isclose(fast.mean(), 5.)
    assert np.isclose(pd.Series(fast).sem(), np.std([1., 4., 10.], ddof=1) / np.sqrt(3))
    result = dict(passed=True, changed_python_and_notebook_sources_compile=True,
        RT_task_specific_anatomy_used_without_HGA_significance=True,
        HGA_original_RT_split_and_finite_trial_filter_preserved=True,
        HGA_per_electrode_waveforms_retain_original_within_subject_mean=True,
        HGA_tasks_collapsed_before_electrode_mean_and_SEM=True,
        RT_and_functional_decoding_models_fitted=False, limitation='Original task epochs missing; RT/decoding fits not claimed')
    path = ROOT / 'docs/evidence/figures3_6_refresh_2026-10-17/approved_method_checks.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
