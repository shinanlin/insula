#!/usr/bin/env python3
"""Prepare three-phase NMF inputs from the reviewed Figure 1 cohort."""
from pathlib import Path
import hashlib
import json
import pickle
import re
import sys
import mne
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/nmf'
FIG1 = ROOT / 'results/fig1'
COHORT = ROOT / 'docs/evidence/subject_task_site_lists_2026-10-17/nmf_candidate_sites_not_final_inputs.csv'
WINDOWS = {'stimulus': (0., .6), 'delay': (0., 1.), 'response': (0., 1.)}
DISPLAY = {'stimulus': (-.5, .6), 'delay': (0., 1.), 'response': (-.5, 1.5)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(COHORT)
    ids = set(candidates.site_id)
    assert len(ids) == 227 and candidates.subject.nunique() == 64
    with (FIG1 / 'available_evidence_stream.pkl').open('rb') as f:
        rows = pickle.load(f)['trace_rows'].copy()
    rows['subject'] = rows.channel.str.split('_').str[0]
    supplement = json.loads((FIG1 / 'average_hga_supplemented_epochs.json').read_text())
    provenance = []
    for item in supplement:
        path = Path(item['path'])
        phase = item['phase']
        lo, hi = DISPLAY[phase]
        epochs = mne.read_epochs(path, preload=False, verbose='error')
        epochs.load_data()
        epochs.pick(item['channels'])
        evoked = epochs.average(method=lambda data: np.nanmean(data, axis=0))
        keep = (evoked.times >= lo) & (evoked.times <= hi)
        frame = pd.DataFrame(evoked.data[:, keep], index=evoked.ch_names, columns=evoked.times[keep])
        frame = frame.rename_axis('channel').reset_index().melt(id_vars='channel', var_name='time', value_name='value')
        modality = re.search(r'_recording-([^_]+)', path.name)
        task = re.search(r'_task-([^_]+)', path.name)[1]
        frame = frame.assign(subject=frame.channel.str.split('_').str[0], task=task, phase=phase,
                             modality=modality[1] if modality else 'sound')
        rows = pd.concat([rows, frame], ignore_index=True)
        provenance.append(dict(path=str(path), sha256=sha(path), channels=item['channels'], phase=phase))
    rows = rows.loc[rows.channel.isin(ids)].copy()
    rows = rows.groupby(['subject','channel','task','phase','modality','time'], as_index=False).value.mean()
    rows.groupby(['subject','channel','task','phase','modality']).size().rename('n_time_samples').to_csv(OUT/'waveform_source_membership.csv')
    site_task = rows.groupby(['subject','channel','task','phase','time'], as_index=False).value.mean()
    means = site_task.groupby(['subject','channel','phase','time'], as_index=False).value.mean()
    means = means.loc[np.isfinite(means.value)]
    coverage = means.groupby(['phase','time']).channel.nunique()
    shared = coverage[coverage.eq(len(ids))].reset_index()[['phase','time']]
    means = means.merge(shared, on=['phase','time'], validate='many_to_one')
    assert set(means.channel) == ids
    assert set(means.phase) == set(WINDOWS)
    means.to_csv(OUT/'site_waveforms.csv.gz', index=False)
    # This reproduces the Figure 1 group mean, providing an independent cohort/weighting check.
    group = means.groupby(['subject','phase','time']).value.mean().groupby(['phase','time']).mean()
    fig1_group = pd.read_csv(FIG1/'average_hga_group_trace.csv').set_index(['phase','time'])['mean']
    assert len(group) == len(fig1_group)
    assert np.allclose(group.sort_index(), fig1_group.sort_index(), atol=1e-10)
    blocks, features = [], []
    for phase, (lo, hi) in WINDOWS.items():
        phase_rows = means.loc[means.phase.eq(phase) & means.time.ge(lo) & means.time.le(hi)]
        matrix = phase_rows.pivot(index='channel', columns='time', values='value').sort_index().sort_index(axis=1)
        assert set(matrix.index) == ids and np.isfinite(matrix.to_numpy()).all()
        blocks.append(matrix.to_numpy(float))
        features.extend(dict(phase=phase, time=float(t)) for t in matrix.columns)
    raw = np.concatenate(blocks, axis=1)
    positive = np.clip(raw, 0, None)
    norms = np.linalg.norm(positive, axis=1)
    assert np.all(norms > np.finfo(float).eps), 'Do not silently drop all-zero sites'
    X = positive / norms[:,None]
    channels = matrix.index.to_numpy(str)
    coords = pd.read_csv(FIG1/'insula_plot_coordinates.csv').set_index('channel')
    metadata = coords.loc[channels].reset_index()
    assert metadata.channel.is_unique and np.isfinite(metadata[['x','y','z']]).all().all()
    metadata.to_csv(OUT/'input_sites.csv', index=False)
    pd.DataFrame(features).to_csv(OUT/'input_features.csv', index=False)
    np.savez_compressed(OUT/'input_matrix.npz', X=X, raw=raw, channels=channels, positive_l2_norm=norms)
    manifest = dict(n_sites=len(channels), n_subjects=int(metadata.subject.nunique()), n_features=X.shape[1],
        phases=list(WINDOWS), fit_windows=WINDOWS, significance_windows={'stimulus':[0,.6],'delay':[0,1],'response':[-.5,.5]},
        phase_samples=pd.DataFrame(features).groupby('phase').size().to_dict(), condition='Repeat',
        modalities=sorted(rows.modality.unique().tolist()),
        weighting='equal modalities within site/task; equal available tasks within site; one NMF row per site',
        preprocessing='negative HGA clipped at zero; L2 normalization across full concatenated waveform',
        excluded_subjects=['D0121'], old_manual_exclusions_applied=False,
        anatomy_selection='Figure 1 reviewed midpoint insular gray matter; endpoints insular GM or confirmed WM',
        incomplete_response_evidence_sites=94, missing_evidence_interpretation='excluded pending evidence; not negative',
        source_cohort=str(COHORT), source_cohort_sha256=sha(COHORT),
        figure1_cache_sha256=sha(FIG1/'available_evidence_stream.pkl'),
        figure1_notebook_sha256=sha(ROOT/'vizpub/fig1.ipynb'),
        figure1_group_mean_reproduced=True, supplemented_epochs=provenance,
        outputs={p.name:sha(p) for p in [OUT/'input_matrix.npz', OUT/'input_sites.csv', OUT/'input_features.csv', OUT/'site_waveforms.csv.gz']})
    (OUT/'input_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['supplemented_epochs','outputs']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
