"""Read-only Picture Naming audit and RTAS5-6 summaries; execute on Slurm."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import h5py
import mne
import numpy as np
import pandas as pd

PN_ROOT = Path('/hpc/group/coganlab/nanlinshi/insula-functional')
PN_BIDS = Path('/cwork/ns458/BIDS-1.3_PictureNaming/BIDS')
PN_DERIV = PN_BIDS / 'derivatives'
PN_EPOCH = PN_DERIV / 'epoch(bipolar)/sub-D0149'
PN_OUT = PN_ROOT / 'results/grant_d0149_highgamma'
PN_ANATOMY = PN_DERIV / 'parcellation/sub-D0149/bipolar/sub-D0149_aparc2009s.csv'
PN_CHANNEL = 'D0149_RTAS5-6'
PN_CONDITIONS = ('Passive', 'Repeat')
PN_MODALITIES = ('image', 'sound', 'text')
PN_PHASES = ('Stimulus', 'Delay', 'Go', 'Response')
PN_TEST_WINDOWS = {p: ((-.5, .5) if p == 'Response' else (0., .5)) for p in PN_PHASES}

def pn_digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def pn_epoch_path(kind, phase, modality, condition):
    task = '' if kind in ('sig', 'effective') else '_task-PictureNaming'
    folder = ('epoch(raw)' if kind == 'raw' else 'epoch(band)(raw)' if kind == 'raw_band'
              else 'epoch(band)(sig)(effective)' if kind == 'effective' else f'epoch(band)({kind})')
    suffix = 'raw' if kind == 'raw' else 'highgamma'
    stem = f'sub-D0149{task}_proc-{phase}_recording-{modality}_desc-{condition}_{suffix}.h5'
    path = PN_EPOCH / folder / stem
    if kind == 'effective' and not path.exists():
        path = path.with_name(stem.replace('sub-D0149_', 'sub-D0149_task-PictureNaming_', 1))
    return path

def pn_stats_path(phase, modality, condition):
    return PN_DERIV / 'statistics/sub-D0149/bipolar' / f'sub-D0149_task-PictureNaming_proc-{phase}_recording-{modality}_desc-{condition}_highgamma.h5'

def pn_intervals(mask, times, sfreq=128):
    changes = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return [(float(times[a]), float(times[b-1] + 1/sfreq))
            for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))]

def pn_audit():
    PN_OUT.mkdir(parents=True, exist_ok=True)
    inventory, details, summaries, intervals, input_hashes = [], [], [], [], []
    def require(path, stage):
        assert path.is_file() and path.stat().st_size > 0, str(path)
        st = path.stat()
        inventory.append(dict(stage=stage, path=str(path), bytes=st.st_size, mtime_ns=st.st_mtime_ns))
    for stage, base in [('raw', PN_BIDS)] + [(s, PN_DERIV/s) for s in ('clean','delay','bipolar')]:
        folder = base/'sub-D0149/ieeg'
        recordings = sorted(folder.glob('*run-*_ieeg.edf'))
        assert len(recordings) == 5, (stage, len(recordings))
        for edf in recordings:
            require(edf, stage)
            for suffix in ('_ieeg.json', '_events.tsv', '_channels.tsv', '_events.json'):
                path = edf.with_name(edf.name.replace('_ieeg.edf', suffix))
                if not path.exists():
                    matches = list(folder.glob('*' + suffix))
                    assert len(matches) == 1, (stage, suffix, matches)
                    path = matches[0]
                require(path, stage)
        for suffix in ('electrodes.tsv','coordsystem.json'):
            paths = list(folder.glob('*'+suffix))
            assert paths, (stage, suffix)
            for path in paths:
                require(path, stage)
    require(PN_ANATOMY, 'parcellation')
    parc = pd.read_csv(PN_ANATOMY)
    assert parc.name.is_unique
    selected = parc.loc[parc.name.eq(PN_CHANNEL)]
    assert len(selected) == 1 and selected.roi.iloc[0] == 'INS' and not selected.mix.iloc[0]
    assert selected.hemi.iloc[0] == 'R'
    pairs_path = PN_DERIV/'qc/bipolar_pairing/sub-D0149_pairs.tsv'
    for suffix in ('pairs','segments','orphans','skipped'):
        require(pairs_path.with_name(f'sub-D0149_{suffix}.tsv'), 'pairing')
    pairs = pd.read_csv(pairs_path, sep='\t')
    pair = pairs.loc[pairs.bipolar_name.eq(PN_CHANNEL)]
    assert len(pair) == 1 and pair.ch1.iloc[0] == 'RTAS5' and pair.ch2.iloc[0] == 'RTAS6'
    assert selected.contact_1.iloc[0] == 'RTAS5' and selected.contact_2.iloc[0] == 'RTAS6'
    for path in (PN_DERIV/'bipolar/sub-D0149/ieeg').glob('*electrodes.tsv'):
        coords = pd.read_csv(path, sep='\t').set_index('name')
        np.testing.assert_allclose(selected[['x','y','z']].to_numpy(float)[0], coords.loc[PN_CHANNEL,['x','y','z']].to_numpy(float)*1000, atol=1e-6)
    selected.to_csv(PN_OUT/'electrode_anatomy.csv', index=False)
    channels = None
    for modality in PN_MODALITIES:
        for condition in PN_CONDITIONS:
            for phase in ('baseline',) + PN_PHASES:
                for kind in ('raw','power') + (() if phase == 'baseline' else ('raw_band','zscore')):
                    require(pn_epoch_path(kind,phase,modality,condition),kind)
                power = mne.read_epochs(pn_epoch_path('power',phase,modality,condition),preload=True,verbose='error')
                pdata = power.get_data(copy=False)
                assert len(power) > 1 and power.info['sfreq'] == 128
                assert pdata.shape[2] == (64 if phase == 'baseline' else 320) and np.isfinite(pdata).all()
                if channels is None:
                    channels = power.ch_names
                    assert PN_CHANNEL in channels and set(channels).issubset(set(pairs.bipolar_name))
                assert channels == power.ch_names
                if phase == 'baseline':
                    details.append(dict(modality=modality,condition=condition,phase=phase,n_trials=len(power),n_channels=len(channels),n_times=len(power.times)))
                    del pdata,power
                    continue
                path = pn_stats_path(phase,modality,condition)
                require(path,'statistics')
                with h5py.File(path) as f:
                    mask,pvals = f['mask'][:].astype(bool),f['pvals'][:]
                    stat_channels = f['ch_names'].asstr()[:].tolist()
                    sig_channels = f['sig_ch_names'].asstr()[:].tolist()
                zpath = pn_epoch_path('zscore',phase,modality,condition)
                z = mne.read_epochs(zpath,preload=True,verbose='error')
                zdata = z.get_data(copy=False)
                assert channels == stat_channels == z.ch_names
                assert zdata.shape == pdata.shape and np.array_equal(z.events,power.events)
                assert np.array_equal(z.times,power.times) and not np.isinf(zdata).any()
                assert mask.shape == pvals.shape == zdata.shape[1:]
                assert np.isfinite(pvals).all() and ((pvals>=0)&(pvals<=1)).all()
                a,b = z.time_as_index(PN_TEST_WINDOWS[phase])
                window = np.zeros(len(z.times),bool);window[a:b]=True
                expected = np.array(channels)[(mask[:,a:b]&(pvals[:,a:b]<.05)).any(axis=1)].tolist()
                assert expected == sig_channels, (modality,condition,phase,'stored significance mismatch')
                if sig_channels:
                    sigpath = pn_epoch_path('sig',phase,modality,condition)
                    require(sigpath,'sig')
                    sig = mne.read_epochs(sigpath,preload=True,verbose='error')
                    assert sig.ch_names == sig_channels and np.array_equal(z.events,sig.events)
                    np.testing.assert_allclose(sig.get_data(copy=False),z.get_data(picks=sig_channels),equal_nan=True)
                    del sig
                effective_path = pn_epoch_path('effective',phase,modality,condition)
                require(effective_path,'effective')
                effective = mne.read_epochs(effective_path,preload=True,verbose='error')
                assert set(effective.ch_names).issubset(sig_channels)
                np.testing.assert_allclose(effective.get_data(copy=False),z.get_data(picks=effective.ch_names),equal_nan=True)
                del effective
                idx = channels.index(PN_CHANNEL)
                x = zdata[:,idx,:]
                n = np.isfinite(x).sum(axis=0)
                assert (n>=2).all()
                mean,sem = np.nanmean(x,axis=0),np.nanstd(x,axis=0,ddof=1)/np.sqrt(n)
                sig_target = mask[idx]&(pvals[idx]<.05)&window
                assert bool(sig_target.any()) == (PN_CHANNEL in sig_channels)
                summaries.append(pd.DataFrame(dict(subject='D0149',channel=PN_CHANNEL,roi='INS',
                    modality=modality,condition=condition,phase=phase,time=z.times,mean=mean,sem=sem,
                    n_valid=n,n_trials=len(z),cluster_mask=mask[idx],pvalue=pvals[idx],
                    in_analysis_window=window,significant=sig_target)))
                for start,stop in pn_intervals(sig_target,z.times):
                    intervals.append(dict(modality=modality,condition=condition,phase=phase,start_s=start,stop_exclusive_s=stop))
                details.append(dict(modality=modality,condition=condition,phase=phase,n_trials=len(z),
                    n_channels=len(channels),n_times=len(z.times),n_nan_all_channels=int(np.isnan(zdata).sum()),
                    min_valid_target=int(n.min()),max_valid_target=int(n.max()),n_significant=len(sig_channels),
                    target_significant=bool(sig_target.any())))
                input_hashes.extend(dict(path=str(p),sha256=pn_digest(p)) for p in (path,zpath))
                print(modality,condition,phase,'OK; RTAS5-6 significant:',bool(sig_target.any()),flush=True)
                del pdata,power,zdata,z,x
    frame = pd.concat(summaries,ignore_index=True)
    assert len(frame) == 3*2*4*320 and np.isfinite(frame[['mean','sem']]).all().all()
    frame.to_csv(PN_OUT/'timecourse_mean_sem.csv',index=False)
    pd.DataFrame(details).to_csv(PN_OUT/'epoch_audit.csv',index=False)
    pd.DataFrame(intervals,columns=['modality','condition','phase','start_s','stop_exclusive_s']).to_csv(PN_OUT/'significant_intervals.csv',index=False)
    pd.DataFrame(inventory).drop_duplicates('path').to_csv(PN_OUT/'file_inventory.csv',index=False)
    report = dict(subject='D0149',display_subject='D149',task='PictureNaming',channel=PN_CHANNEL,
        status='passed',completed_utc=datetime.now(timezone.utc).isoformat(),n_recordings=5,
        n_bipolar_channels=len(channels),conditions=PN_CONDITIONS,modalities=PN_MODALITIES,
        phases=PN_PHASES,statistics_count=24,effective_epoch_count=24,anatomy_source=str(PN_ANATOMY),
        anatomy_sha256=pn_digest(PN_ANATOMY),input_hashes=input_hashes,
        passive_response='Synthetic alignment at GoCue plus subject median Repeat reaction time; not a vocal onset.',
        note='All requested channel trials retained regardless of significance. Raw EDF data are inventoried, not reprocessed. TFR/decoding are outside this temporal HGA audit.')
    (PN_OUT/'audit.json').write_text(json.dumps(report,indent=2))
    print('AUDIT PASSED:',PN_CHANNEL,flush=True)
    return report,frame,pd.DataFrame(details)

if __name__ == '__main__':
    pn_audit()
