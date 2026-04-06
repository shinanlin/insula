# Cross-correlation analysis between Insula and IFG electrodes
# to detect volume conduction (peak at lag=0)

import argparse
from typing import List, Tuple
import numpy as np
from mne_bids import BIDSPath
import pandas as pd
import mne
import logging
from tqdm import tqdm
from scipy.signal import correlate
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from ieeg.viz.mri import force2frame
import re
import xarray as xr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_xcorr_matrix(
    xdata: np.ndarray,
    sfreq: float,
    max_lag_s: float = 1,
    method: str = "fft",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute trial×chan×chan×lag cross-correlation (z-scored, squared).

    Args:
        xdata: (n_trials, n_chan, n_time)
        sfreq: sampling rate
        max_lag_s: symmetric window in seconds (default 1s ⇒ [-1, 1])
        method: 'fft' (fast) or 'loop'

    Returns:
        xcorr: (n_trials, n_chan, n_chan, n_lags) variance-explained matrix
        lag_times: (n_lags,) lag axis in seconds
    """
    n_trials, n_chan, n_time = xdata.shape
    max_lag = int(max_lag_s * sfreq)
    lags = np.arange(-max_lag, max_lag + 1)
    lag_times = lags / sfreq

    # z-score per trial/channel across time; guard zero std
    mean = xdata.mean(axis=2, keepdims=True)
    std = np.maximum(xdata.std(axis=2, keepdims=True), np.finfo(float).eps)
    zdata = (xdata - mean) / std

    if method == "fft":
        # pad length for linear correlation
        pad_len = 1 << int(np.ceil(np.log2(n_time + 2 * max_lag)))
        freqs = np.fft.rfft(zdata, n=pad_len, axis=2)
        xcorr = np.empty((n_trials, n_chan, n_chan, len(lags)), dtype=np.float32)
        for t in range(n_trials):
            F = freqs[t]  # (chan, freq)
            cs = F[:, None, :] * np.conj(F[None, :, :])  # (chan, chan, freq)
            corr_full = np.fft.irfft(cs, n=pad_len, axis=-1)
            # zero-lag at index 0; reorder to [-lag, +lag]
            neg = corr_full[..., -max_lag:]
            pos = corr_full[..., : max_lag + 1]
            seg = np.concatenate([neg, pos], axis=-1) / n_time
            idx = np.abs(seg).argmax(axis=-1)[..., None]
            peak_sign = np.take_along_axis(seg, idx, axis=-1)[..., 0]
            peak_sign = np.sign(peak_sign)
            peak_sign[peak_sign == 0] = 1.0
            xcorr[t] = seg * peak_sign[..., None]
    else:
        xcorr = np.empty((n_trials, n_chan, n_chan, len(lags)), dtype=np.float32)
        for t in range(n_trials):
            for i in range(n_chan):
                xi = zdata[t, i]
                for j in range(n_chan):
                    xj = zdata[t, j]
                    full = correlate(xi, xj, mode='full', method='auto')
                    mid = len(full) // 2
                    start = mid - max_lag
                    stop = mid + max_lag + 1
                    seg = full[start:stop] / n_time
                    # flip by peak sign to make maximal peak positive
                    peak_sign = np.sign(seg[np.argmax(np.abs(seg))]) if seg.size else 1.0
                    if peak_sign == 0:
                        peak_sign = 1.0
                    xcorr[t, i, j] = seg * peak_sign

    # square to express variance explained
    xcorr = xcorr ** 2
    return xcorr, lag_times


def package_xcorr(
    peak_val: np.ndarray,
    peak_lag: np.ndarray,
    ch_names: List[str],
    cord_df: pd.DataFrame,
    insula_chans: List[str] | None = None,
) -> pd.DataFrame:
    """Flatten peak metrics to long dataframe per trial/source/target.

    Args:
        peak_val: (trial, chan, chan)
        peak_lag: (trial, chan, chan)
        ch_names: channel order matching peak arrays
        cord_df: indexed metadata (channel, roi, label, x, y, z) aligned to ch_names
    """
    n_trials, n_chan, _ = peak_val.shape
    pair_n = n_chan * n_chan

    src_idx = np.repeat(np.arange(n_chan), n_chan)
    tgt_idx = np.tile(np.arange(n_chan), n_chan)

    trial_col = np.repeat(np.arange(n_trials), pair_n)

    meta = cord_df.loc[ch_names]
    src_meta = meta.iloc[src_idx]
    tgt_meta = meta.iloc[tgt_idx]

    peak_val_flat = peak_val.reshape(-1)
    peak_lag_flat = peak_lag.reshape(-1)

    # expand per-trial to avoid duplicated rows without trial context
    src_flat = np.broadcast_to(src_meta.index.values, (n_trials, pair_n)).reshape(-1)
    tgt_flat = np.broadcast_to(tgt_meta.index.values, (n_trials, pair_n)).reshape(-1)
    src_meta_rep = {k: np.broadcast_to(src_meta[k].values, (n_trials, pair_n)).reshape(-1) for k in ['roi', 'label', 'x', 'y', 'z']}
    tgt_meta_rep = {k: np.broadcast_to(tgt_meta[k].values, (n_trials, pair_n)).reshape(-1) for k in ['roi', 'label', 'x', 'y', 'z']}

    if insula_chans:
        insula_set = set(insula_chans)
        # keep only rows where source is insula to avoid symmetric duplicates
        mask = np.isin(src_flat, list(insula_set))
    else:
        mask = slice(None)

    df = pd.DataFrame(
        {
            'trial': trial_col[mask],
            'source': src_flat[mask],
            'target': tgt_flat[mask],
            'source_roi': src_meta_rep['roi'][mask],
            'source_label': src_meta_rep['label'][mask],
            'source_x': src_meta_rep['x'][mask],
            'source_y': src_meta_rep['y'][mask],
            'source_z': src_meta_rep['z'][mask],
            'target_roi': tgt_meta_rep['roi'][mask],
            'target_label': tgt_meta_rep['label'][mask],
            'target_x': tgt_meta_rep['x'][mask],
            'target_y': tgt_meta_rep['y'][mask],
            'target_z': tgt_meta_rep['z'][mask],
            'peak_val': peak_val_flat[mask],
            'peak_lag_s': peak_lag_flat[mask],
        }
    )
    return df


def main(
    bids_root: str,
    band: str,
    reference: str,
    recon_dir: str
):

    datatype = 'epoch(raw)' if band=='raw' else 'epoch(band)(raw)'
    raw_pts = BIDSPath(
        root=bids_root + f"derivatives/epoch({reference})",
        datatype=datatype,
        suffix=band,
        extension='.h5',
        check=False
    )
    
    for raw_pt in tqdm(raw_pts.match(), desc='Processing subjects'):
        
        subject = raw_pt.subject
        phase = raw_pt.processing
        desc = raw_pt.description
        task = raw_pt.task
        
        raw = mne.read_epochs(raw_pt, preload=True)
        
        # load parcellation file
        try:
            parc_path = raw_pt.copy().update(
                root=str(raw_pt.root).replace(f'epoch({reference})', 'parcellation'),
                datatype=reference,
                task=None,
                description=None,
                recording=None,
                processing='3mm',
                suffix='aparc2009s',
                extension='.csv',
            ).match()[0]
            parc = pd.read_csv(parc_path)
        except IndexError:
            logger.warning(f"No parcellation file found for subject {subject}")
            continue
        
        # merge parcellation with raw
        parc.rename(columns={'name': 'channel'}, inplace=True)
        
        # Check if required columns exist in parcellation
        required_cols = ['channel', 'label', 'roi', 'hemi']
        if not all(col in parc.columns for col in required_cols):
            print(f"Skipping {subject}: missing columns in parcellation file")
            continue
        
        parc_sub = parc[['channel', 'label','roi','hemi']]
        
        montage = raw.get_montage()
        sub_id = re.sub(r'^D0+', 'D', subject)
        to_fsaverage = mne.read_talxfm(sub_id, recon_dir)
        trans = mne.transforms.Transform(fro='head', to='mri',
                                        trans=to_fsaverage['trans'])
        force2frame(montage, trans.from_str)  
        montage.apply_trans(trans) 
        pos_m = montage.get_positions()['ch_pos']
        
        cord_df = pd.DataFrame(pos_m).T
        cord_df.columns = ['x', 'y', 'z']
        cord_df[['x','y','z']] *= 1000
        cord_df = cord_df.reset_index().rename(columns={'index': 'channel'})
        cord_df = cord_df[cord_df.channel.isin(parc_sub.channel)]
        cord_df = cord_df.merge(parc_sub, on='channel', how='left')
        
        # remove white matter, intersection and unknown electrodes
        cord_df = cord_df[~cord_df['roi'].str.contains('white|intersection|unknown|WM', case=False, na=False)]
        cord_df = cord_df[~cord_df['roi'].isin([
            'OPC', 'sOccG', 'OccS', 'mOccG',
            'WM','LinG','PhG','Cb','iOccGs',
            'GRect','Amyg','Hipp','LinGs',
            'CG','OFCs','CGs',
            'Thal','Right-Pallidum','Left-Pallidum',
            'Calcarine','CollatAnt','VDC','InfLatV','LatV',
            'CC_Central','BrainStem','CollatPost','Caud'
        ])]
        
        # pick channels
        raw.pick_channels(cord_df.channel.unique().tolist())
        
        # pass if there is no insula channel
        if len(cord_df[cord_df.roi.str.contains('ins', case=False, na=False)]) == 0:
            logger.warning(f"No insula channel found for subject {subject}")
            continue
        
        # crop
        xdata = raw.get_data()
        
        xcorr, lag_times = compute_xcorr_matrix(xdata, raw.info['sfreq'])
        
        peak_idx = xcorr.argmax(axis=-1)                      # (trial, chan, chan)
        peak_lag = lag_times[peak_idx]                        # (trial, chan, chan)
        peak_val = xcorr.max(axis=-1)                         # (trial, chan, chan)

        cord_ordered = cord_df.set_index('channel').loc[raw.ch_names]
        insula_chans = [ch for ch, roi in cord_ordered['roi'].items() if 'ins' in str(roi).lower()]
        conn_df = package_xcorr(peak_val, peak_lag, raw.ch_names, cord_ordered, insula_chans)
        
        conn_df['subject'] = subject
        conn_df['phase'] = phase
        conn_df['description'] = desc
        conn_df['band'] = band
        conn_df['task'] = task
        
        # save df
        save_path = BIDSPath(
            root=f'results/{task}({reference})',
            description=desc,
            datatype='xcorr',
            suffix=band,
            recording=raw_pt.recording,
            task=task,
            subject=subject,
            processing=phase,
            extension=".csv",
            check=False,
        )
        conn_df.to_csv(save_path, index=False)
        
        # Save trial-averaged 3D waveform as xarray Dataset to reduce size
        xcorr_mean = xcorr.mean(axis=0)  # (source, target, lag)
        chans = raw.ch_names
        
        da = xr.DataArray(
            xcorr_mean,
            dims=['source', 'target', 'lag'],
            coords={
                'source': chans,
                'target': chans,
                'lag': lag_times
            },
            name='xcorr_wave'
        )
        
        # Attach channel metadata as dataset variables
        ds = xr.Dataset({'xcorr': da})
        ds = ds.assign_coords({
            'roi': ('source', cord_ordered['roi'].values),
            'hemi': ('source', cord_ordered['hemi'].values),
            'label': ('source', cord_ordered['label'].values),
            'x': ('source', cord_ordered['x'].values),
            'y': ('source', cord_ordered['y'].values),
            'z': ('source', cord_ordered['z'].values)
        })
        
        # Add global attributes
        ds.attrs['subject'] = subject
        ds.attrs['task'] = task
        ds.attrs['phase'] = phase
        ds.attrs['band'] = band
        ds.attrs['description'] = desc
        ds.attrs['n_trials_averaged'] = int(xcorr.shape[0])
        
        # Save to NetCDF
        nc_save_path = save_path.update(suffix='wave', extension='.nc')
        ds.to_netcdf(nc_save_path, engine='h5netcdf')
        logger.info(f"Saved full xarray dataset to {nc_save_path}")
    
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/", type=str)
    # parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.0_LexicalDecRepNoDelay/BIDS/", type=str)
    parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/", type=str)
    # parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.3_PictureNaming/BIDS/", type=str)
    # parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.4_SentenceRep/BIDS/", type=str)
    parser.add_argument("--band", type=str, default="highgamma", choices=['highgamma', 'lowband', 'raw'])
    parser.add_argument("--reference", type=str, default='bipolar', choices=['bipolar', 'car'])
    parser.add_argument("--recon_dir", type=str, default=r'/cwork/ns458/ECoG_Recon/',
                        help='path to the recon-all directory')
    args = parser.parse_args()
    main(**vars(args))
