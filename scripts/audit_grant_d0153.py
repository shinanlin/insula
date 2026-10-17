"""Read-only upstream audit for the temporary D0153 grant HGA figures.

Run on Slurm. Writes only the isolated grant result directory.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import h5py
import mne
import numpy as np
import pandas as pd

ROOT = Path('/hpc/group/coganlab/nanlinshi/insula-functional')
BIDS = Path('/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS')
DERIV = BIDS / 'derivatives'
EPOCH = DERIV / 'epoch(bipolar)/sub-D0153'
ANATOMY = Path('/cwork/ns458/insula_recon_bipolar/derivatives/parcellation/sub-D0153/bipolar/sub-D0153_aparc2009sHammers.csv')
OUT = ROOT / 'results/grant_d0153_highgamma'
CONDITIONS = ('Decision', 'Repeat')
PHASES = ('Stimulus', 'Delay', 'Go', 'Response')
TEST_WINDOWS = {p: ((-.5, .5) if p == 'Response' else (0., .5)) for p in PHASES}

def epoch_path(kind, phase, condition):
    task = '' if kind == 'sig' else '_task-LexicalDelay'
    datatype = 'epoch(raw)' if kind == 'raw' else f'epoch(band)({kind})'
    suffix = 'raw' if kind == 'raw' else 'highgamma'
    return EPOCH / datatype / f'sub-D0153{task}_proc-{phase}_desc-{condition}_{suffix}.h5'

def stats_path(phase, condition):
    return DERIV / 'statistics/sub-D0153/bipolar' / f'sub-D0153_task-LexicalDelay_proc-{phase}_desc-{condition}_highgamma.h5'

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read_stats(path):
    with h5py.File(path) as f:
        return (f['mask'][:].astype(bool), f['pvals'][:],
                f['ch_names'].asstr()[:].tolist(), f['sig_ch_names'].asstr()[:].tolist())

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inventory, details, source_files = [], [], []
    def require(path, stage):
        assert path.is_file() and path.stat().st_size > 0, str(path)
        st = path.stat()
        inventory.append(dict(stage=stage, path=str(path), bytes=st.st_size, mtime_ns=st.st_mtime_ns))
        source_files.append(path)

    for stage, base in [('raw', BIDS), ('clean', DERIV / 'clean'), ('bipolar', DERIV / 'bipolar')]:
        folder = base / 'sub-D0153/ieeg'
        recordings = sorted(folder.glob('*run-*_ieeg.edf'))
        assert len(recordings) == 4, (stage, recordings)
        for edf in recordings:
            require(edf, stage)
            for suffix in ('_ieeg.json', '_events.tsv'):
                require(edf.with_name(edf.name.replace('_ieeg.edf', suffix)), stage)
            channels = edf.with_name(edf.name.replace('_ieeg.edf', '_channels.tsv'))
            if not channels.exists():
                matches = list(folder.glob('*channels.tsv'))
                assert len(matches) == 1
                channels = matches[0]
            require(channels, stage)
            events_json = edf.with_name(edf.name.replace('_ieeg.edf', '_events.json'))
            if not events_json.exists():
                candidates = list(folder.glob('*events.json'))
                assert len(candidates) == 1
                events_json = candidates[0]
            require(events_json, stage)
        for suffix in ('electrodes.tsv', 'coordsystem.json'):
            matches = list(folder.glob('*' + suffix))
            assert len(matches) == 1
            require(matches[0], stage)
    pairs_path = DERIV / 'qc/bipolar_pairing/sub-D0153_pairs.tsv'
    for suffix in ('pairs', 'segments', 'orphans', 'skipped'):
        require(pairs_path.with_name(f'sub-D0153_{suffix}.tsv'), 'pairing')
    require(ANATOMY, 'external_anatomy')
    parc = pd.read_csv(ANATOMY)
    pairs = pd.read_csv(pairs_path, sep='\t')
    coords_path = next((DERIV / 'bipolar/sub-D0153/ieeg').glob('*electrodes.tsv'))
    coords = pd.read_csv(coords_path, sep='\t').set_index('name')
    assert parc.name.is_unique and pairs.bipolar_name.is_unique
    matched = parc.merge(pairs, left_on='name', right_on='bipolar_name', validate='one_to_one')
    assert len(matched) == len(pairs) == 192, (len(matched), len(pairs))
    assert (matched.contact_1 == matched.ch1).all() and (matched.contact_2 == matched.ch2).all()
    assert np.allclose(matched[['x', 'y', 'z']], coords.loc[matched.name, ['x', 'y', 'z']].to_numpy(float) * 1000, atol=1e-6)
    strict = parc.roi.isin(['AIC', 'PIC', 'INS']) & ~parc.mix
    selected = parc.loc[strict | parc.name.eq('D0153_RFMI8-9')].copy()
    selected['selection'] = np.where(selected.name.eq('D0153_RFMI8-9'), 'user_selected_IFG_INS_boundary', 'nonmixed_insula')
    assert selected.loc[selected.name.eq('D0153_RFMI8-9'), 'roi'].iloc[0] == 'IFG–INS'
    assert len(selected) > 0
    selected.to_csv(OUT / 'insula_electrodes.csv', index=False)
    parc.loc[parc.roi.str.contains('AIC|PIC|INS') & parc.mix & ~parc.name.isin(selected.name)].to_csv(OUT / 'mixed_boundary_electrodes_excluded.csv', index=False)
    channels = None
    for condition in CONDITIONS:
        for phase in ('baseline',) + PHASES:
            for kind in ('raw', 'power') + (() if phase == 'baseline' else ('raw_band', 'zscore', 'sig')):
                path = epoch_path('raw' if kind == 'raw_band' else kind, phase, condition)
                if kind == 'raw_band':
                    path = EPOCH / 'epoch(band)(raw)' / path.name.replace('_raw.h5', '_highgamma.h5')
                require(path, kind)
            power = mne.read_epochs(epoch_path('power', phase, condition), preload=True, verbose='error')
            data = power.get_data(copy=False)
            assert data.shape == (168, 192, 64 if phase == 'baseline' else 320), (condition, phase, data.shape)
            assert power.info['sfreq'] == 128 and np.isfinite(data).all()
            if channels is None:
                channels = power.ch_names
                assert set(channels) == set(pairs.bipolar_name)
            assert channels == power.ch_names
            details.append(dict(condition=condition, phase=phase, kind='power', n_trials=len(power), n_channels=len(channels), n_times=len(power.times), n_nan=0))
            del data, power
            if phase == 'baseline':
                continue
            path = stats_path(phase, condition)
            require(path, 'statistics')
            mask, pvals, stat_channels, sig_channels = read_stats(path)
            z = mne.read_epochs(epoch_path('zscore', phase, condition), preload=True, verbose='error')
            assert channels == stat_channels == z.ch_names
            assert mask.shape == pvals.shape == (192, 320)
            assert np.isfinite(pvals).all() and ((pvals >= 0) & (pvals <= 1)).all()
            start, stop = z.time_as_index(TEST_WINDOWS[phase])
            expected = np.array(channels)[(mask[:, start:stop] & (pvals[:, start:stop] < .05)).any(axis=1)].tolist()
            assert sig_channels == expected, (condition, phase, 'significant channel mismatch')
            data = z.get_data(copy=False)
            assert data.shape == (168, 192, 320) and not np.isinf(data).any()
            selected_data = z.get_data(picks=selected.name.tolist())
            assert (np.isfinite(selected_data).sum(axis=0) >= 2).all()
            sig = mne.read_epochs(epoch_path('sig', phase, condition), preload=True, verbose='error')
            assert sig.ch_names == sig_channels and len(sig) == len(z)
            np.testing.assert_allclose(sig.get_data(copy=False), z.get_data(picks=sig_channels), equal_nan=True)
            details.append(dict(condition=condition, phase=phase, kind='zscore', n_trials=len(z), n_channels=len(channels), n_times=len(z.times), n_nan=int(np.isnan(data).sum()), n_significant=len(sig_channels), insula_significant=';'.join(c for c in selected.name if c in sig_channels)))
            print(condition, phase, 'OK', 'insula significant:', details[-1]['insula_significant'], flush=True)
            del data, z, sig, selected_data
    pd.DataFrame(inventory).drop_duplicates('path').to_csv(OUT / 'file_inventory.csv', index=False)
    pd.DataFrame(details).to_csv(OUT / 'epoch_audit.csv', index=False)
    report = dict(subject='D0153', task='LexicalDelay', completed_utc=datetime.now(timezone.utc).isoformat(),
                  status='passed_for_requested_visualization', signal_stages_complete=True, significance_complete=True,
                  conditions=CONDITIONS, phases=PHASES, trials_per_condition=168, n_bipolar_channels=192,
                  sfreq=128, insula_channels=selected.name.tolist(),
                  anatomy_source=str(ANATOMY), anatomy_sha256=digest(ANATOMY),
                  task_parcellation_present=(DERIV / 'parcellation/sub-D0153').exists(),
                  effective_sig_epoch_count=len(list((EPOCH / 'epoch(band)(sig)(effective)').glob('*.h5'))),
                  full_pipeline_complete=False,
                  anatomy_note='Existing recon-only hybrid aparc2009s/Hammers table; all task pair identities and native midpoints verified. ROI AIC/PIC/INS, mix=False, plus user-selected RFMI8-9 labeled IFG–INS. Other mixed boundary pairs excluded; no significance selection.',
                  upstream_file_count=len(set(source_files)), test_windows=TEST_WINDOWS,
                  significance_rule='Saved cluster mask AND pvals < 0.05; channel inclusion uses half-open phase analysis window; no new tests.',
                  limitation='Complete for temporal HGA visualization using existing external anatomy. Task-local parcellation and effective-significant epoch exports are not complete; these are not required for plotting all anatomically selected insula channels from z-score epochs. TFR and decoding are outside this audit.')
    (OUT / 'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    main()
