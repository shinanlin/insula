#!/usr/bin/env python3
"""Validate the standalone Fig 3 memory copy on a Slurm compute node.

Outputs go under logs/fig3_memory_validation, never the publication figures.
Run --full to also execute every cell of the optimized notebook.
"""
import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath

ORIGINAL = ROOT / 'vizpub/fig3.ipynb'
OPTIMIZED = ROOT / 'vizpub/fig3_memory_optimized.ipynb'
RUN = ROOT / 'logs/fig3_memory_validation' / os.environ.get('SLURM_JOB_ID', 'manual')
RUN.mkdir(parents=True, exist_ok=True)
notebooks = [json.loads(p.read_text()) for p in (ORIGINAL, OPTIMIZED)]
report = {'source_sha256': hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(), 'checks': []}


def source(which, i):
    return ''.join(notebooks[which]['cells'][i]['source'])


def definitions(which, i, ns):
    tree = ast.parse(source(which, i))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    exec(compile(tree, f'{which}:cell{i}:functions', 'exec'), ns)


def record(name, **kwargs):
    item = {'name': name, **kwargs}
    report['checks'].append(item)
    print(json.dumps(item), flush=True)
    (RUN / 'validation.json').write_text(json.dumps(report, indent=2))


def init(which):
    ns = {'__name__': '__main__'}
    exec('\n'.join(l for l in source(which, 1).splitlines() if not l.startswith('%')), ns)
    def test_img_dir(name):
        p = RUN / 'figures' / name
        p.mkdir(parents=True, exist_ok=True)
        return p
    ns['img_dir'] = test_img_dir
    exec(source(which, 2), ns)
    exec(source(which, 4), ns)
    ns['HEATMAP_VMAX_Q'] = 0.95
    definitions(which, 20, ns)
    definitions(which, 27, ns)
    return ns


def assert_table_equal(a, b):
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def synthetic_epochs(old, new):
    rng = np.random.default_rng(73)
    for nt in (1, 17, 320):
        data = rng.standard_normal((97, 4, nt))
        data[::3, 1, :] = np.nan
        data[:, 2, 0] = np.nan
        data[0, 0, 0] = np.inf
        data[0, 3, 0] = -np.inf
        info = mne.create_info(['c','a','b','d'], 200., ch_types='seeg')
        epo = mne.EpochsArray(data, info, verbose=False)
        picks = ['b','a','missing','c','d']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = old['evoked_to_long'](epo, picks)
            b = new['evoked_to_long'](epo, picks)
        assert_table_equal(a,b)
        assert new['evoked_to_long'](epo, ['absent']) is None
        record('synthetic_trial_mean_exact', timepoints=nt, rows=len(a))


class CSVPath:
    def __init__(self, file):
        self.file = file
        self.subject = 'D0001'
        self.description = 'Repeat'
        self.recording = 'sound'
        self.processing = 'Stimulus'
    def __fspath__(self):
        return str(self.file)


def mask_edge_cases(old, new):
    keys = ['subject','channel','time','phase','modality','description']
    base = pd.DataFrame({
        'subject':['D0001']*4, 'channel':['A','A','B','absent'],
        'time':[0.,0.,0.1,0.2], 'phase':['stimulus']*4,
        'modality':['sound']*4, 'description':['Repeat']*4,
        'value':[0.1,0.1,np.nan,0.3], 'mask':[True]*4,
    }, index=[5,8,9,12])
    # Duplicates cross the reader's 20,000 row chunk boundary and file boundary.
    frame = pd.DataFrame({
        'subject':['sub-D0001']*20003, 'channel':['irrelevant']*20003,
        'time':[0.]*20003, 'phase':['Audio']*20003,
        'modality':['sound']*20003, 'description':['Repeat']*20003,
        'mask':[True]*20003,
    })
    frame.loc[0, ['channel','mask']] = ['A',False]
    frame.loc[20000, ['channel','mask']] = ['A',True]
    frame.loc[1, ['channel','time','mask']] = ['B',0.10000001,np.nan]
    frame.loc[20001, ['channel','time','mask']] = ['B',0.1,True]
    csv = RUN / 'mask_fixture.csv'
    frame.to_csv(csv,index=False)
    empty = RUN / 'mask_empty.csv'
    frame.iloc[:0].to_csv(empty,index=False)
    unrelated = RUN / 'mask_unrelated.csv'
    frame.iloc[[2]].to_csv(unrelated,index=False)
    fixtures = {
        'duplicate_false_nan_across_chunks': [CSVPath(csv),CSVPath(csv)],
        'no_files': [],
        'empty_file': [CSVPath(empty)],
        'file_without_requested_channel': [CSVPath(unrelated)],
    }
    for case, paths in fixtures.items():
        class FakeBIDS:
            def __init__(self, **kwargs): pass
            def match(self): return paths
        for ns in (old,new):
            ns['BIDSPath'] = FakeBIDS
            ns['TASK'] = 'PictureNaming'
        for helper in ('attach_packaged_mask','rfig3_add_packaged_mask'):
            kwargs = {} if helper == 'attach_packaged_mask' else {'task':'PictureNaming'}
            a = old[helper](base, **kwargs)
            b = new[helper](base, **kwargs)
            assert_table_equal(a,b)
            record('mask_fixture_exact', helper=helper, case=case, rows=len(a))
    for ns in (old,new): ns['BIDSPath'] = BIDSPath


def real_epochs_and_masks(old, new):
    assignments = old['load_assignments']()
    for task in old['TASKS']:
        channels = assignments.groupby('subject')['channel'].agg(list).to_dict()
        paths = old['list_zscore_paths'](old['TASK_BIDS'][task])
        candidates = [p for p in paths if p.subject in channels
                      and p.description == 'Repeat' and old['_modality_of'](p) == 'sound'
                      and old['_norm_phase'](p.processing) == 'stimulus']
        # Representative largest file exercises the original preload/copy path.
        path = max(candidates, key=lambda p: Path(p.fpath).stat().st_size)
        epo = mne.read_epochs(path, preload=True, verbose=False)
        picks = channels[path.subject]
        a = old['evoked_to_long'](epo,picks)
        full_bytes = epo.get_data(copy=False).nbytes
        n_trials = len(epo.events)
        del epo
        gc.collect()
        epo = mne.read_epochs(path, preload=False, verbose=False)
        b = new['evoked_to_long'](epo,picks)
        assert_table_equal(a,b)
        subject, phase, description, modality = path.subject, old['_norm_phase'](path.processing), path.description, old['_modality_of'](path)
        record('real_trial_mean_exact',task=task,path=str(path.fpath),trials=n_trials,rows=len(a),full_epoch_bytes=full_bytes)
        del epo
        for col,value in [('subject',subject),('phase',phase),('description',description),('modality',modality)]:
            b[col] = value
        for helper in ('attach_packaged_mask','rfig3_add_packaged_mask'):
            for ns in (old,new): ns['TASK'] = task
            kwargs = {} if helper == 'attach_packaged_mask' else {'task':task}
            expected = old[helper](b,**kwargs)
            actual = new[helper](b,**kwargs)
            assert_table_equal(expected,actual)
            record('real_packaged_mask_exact',task=task,helper=helper,rows=len(actual),significant=int(actual['mask'].sum()))
        del a,b,expected,actual
        gc.collect()


def full_run():
    ns = {'__name__':'__main__'}
    for i,c in enumerate(notebooks[1]['cells']):
        if c['cell_type'] != 'code': continue
        text = '\n'.join(l for l in source(1,i).splitlines() if not l.startswith('%'))
        print(f'FULL RUN cell {i} start',flush=True)
        started = time.monotonic()
        exec(compile(text, f'fig3_memory_optimized.ipynb:cell{i}', 'exec'),ns)
        if i == 1:
            def test_img_dir(name):
                p = RUN / 'figures' / name
                p.mkdir(parents=True,exist_ok=True)
                return p
            ns['img_dir'] = test_img_dir
            # Headless equivalent of inline display; save calls stay untouched.
            plt.show = lambda *a,**k: plt.close('all')
        record('full_cell_completed',cell=i,seconds=round(time.monotonic()-started,2),peak_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1))
    record('full_notebook_completed',svg_count=len(list((RUN/'figures').rglob('*.svg'))))


def compare_all_tables():
    """Compare complete scientific inputs to every plot, without rendering."""
    baseline = None
    for which in (0, 1):
        ns = init(which)
        fingerprints = {}
        def capture(name):
            frame = ns[name]
            fingerprint = {
                'rows': len(frame),
                'columns': list(frame.columns),
                'dtypes': [str(d) for d in frame.dtypes],
                'sha256': hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest(),
            }
            fingerprints[name] = fingerprint
            if which:
                assert fingerprint == baseline[name], (name, baseline[name], fingerprint)
            record('complete_table_exact' if which else 'baseline_table',table=name,**fingerprint)
        exec(source(which,6),ns)
        capture('working')
        capture('pooled_df')
        # Run the original delta construction unchanged, stopping before plots.
        delta_source = source(which,10).split('for contrast in contrasts:')[0]
        exec(delta_source,ns)
        capture('delta_df')
        for name in ('hga_parts','pooled_parts','pn','pn_resp','part','chunk','delta_src','delta_parts'):
            ns.pop(name,None)
        gc.collect()
        exec(source(which,19),ns)
        capture('pn_repeat')
        ns['legacy_masked'] = ns['attach_packaged_mask'](ns['pn_repeat'])
        capture('legacy_masked')
        del ns['legacy_masked']
        ns['RFIG3_PHASE_ORDER'] = ('stimulus','delay','response')
        exec(source(which,30),ns)
        for name in ('rfig3_input_df','rfig3_delay_df','rfig3_output_df'):
            capture(name)
        # Spatial electrode membership has a separate streamed CSV reader.
        definitions(which,22,ns)
        ns['spatial_membership'] = pd.DataFrame({
            modality: pd.Series(sorted(ns['significant_channels']('Stimulus',modality,0.0,0.6,set(ns['assignments']['channel'].astype(str)))))
            for modality in ('sound','image')
        })
        capture('spatial_membership')
        if which == 0:
            baseline = fingerprints
        del ns
        gc.collect()
    record('all_scientific_tables_match',tables=len(baseline))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--compare-tables',action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT/'vizpub')
    if args.compare_tables:
        compare_all_tables()
        return
    old,new = init(0),init(1)
    for i,c in enumerate(notebooks[1]['cells']):
        if c['cell_type']=='code':
            compile('\n'.join(l for l in source(1,i).splitlines() if not l.startswith('%')),str(i),'exec')
    assert report['source_sha256'] == notebooks[1]['metadata']['memory_optimization']['source_sha256']
    synthetic_epochs(old,new)
    mask_edge_cases(old,new)
    real_epochs_and_masks(old,new)
    del old,new
    gc.collect()
    if args.full: full_run()
    record('validation_passed',peak_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1))

if __name__ == '__main__': main()
