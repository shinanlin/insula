"""Compute Insula–IFG cross-correlation curves and save long-format CSV.

This script computes trial-level xcorr (z-scored, squared) but saves the
trial-averaged curve for each Insula→IFG pair. Results are stored under
BIDSPath datatype=xcorr with suffix=wave.
"""

import argparse
import logging
import re
from typing import Dict, List, Tuple

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath
from scipy.signal import correlate
from tqdm import tqdm
from ieeg.viz.mri import force2frame

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


EXCLUDE_ROIS = {
    'OPC', 'sOccG', 'OccS', 'mOccG', 'WM', 'LinG', 'PhG', 'Cb', 'iOccGs',
    'GRect', 'Amyg', 'Hipp', 'LinGs', 'CG', 'OFCs', 'SFGs', 'CGs',
    'Thal', 'Right-Pallidum', 'Left-Pallidum', 'Calcarine', 'CollatAnt',
    'VDC', 'InfLatV', 'LatV', 'CC_Central', 'BrainStem', 'CollatPost',
    'Caud',
}


def compute_xcorr_matrix(
    xdata: np.ndarray,
    sfreq: float,
    max_lag_s: float = 0.5,
    method: str = "fft",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute trial×chan×chan×lag cross-correlation (z-scored, squared)."""
    n_trials, n_chan, n_time = xdata.shape
    max_lag = int(max_lag_s * sfreq)
    lags = np.arange(-max_lag, max_lag + 1)
    lag_times = lags / sfreq

    mean = np.nanmean(xdata, axis=2, keepdims=True)
    std = np.nanstd(xdata, axis=2, keepdims=True)
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0)
    std = np.maximum(std, np.finfo(float).eps)
    zdata = (np.nan_to_num(xdata, nan=0.0) - mean) / std

    if method == "fft":
        pad_len = 1 << int(np.ceil(np.log2(n_time + 2 * max_lag)))
        freqs = np.fft.rfft(zdata, n=pad_len, axis=2)
        xcorr = np.empty((n_trials, n_chan, n_chan, len(lags)), dtype=np.float32)
        for t in range(n_trials):
            F = freqs[t]
            cs = F[:, None, :] * np.conj(F[None, :, :])
            corr_full = np.fft.irfft(cs, n=pad_len, axis=-1)
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
                    peak_sign = np.sign(seg[np.argmax(np.abs(seg))]) if seg.size else 1.0
                    if peak_sign == 0:
                        peak_sign = 1.0
                    xcorr[t, i, j] = seg * peak_sign

    xcorr = xcorr ** 2
    return xcorr, lag_times


def parse_bipolar_name(ch_name: str) -> Tuple[str, int, int] | None:
    """Parse bipolar channel name into (shank_prefix, i1, i2)."""
    match = re.match(r"^[^_]+_(.*?)(\d+)-(\d+)$", ch_name)
    if not match:
        return None
    shank = match.group(1)
    i1 = int(match.group(2))
    i2 = int(match.group(3))
    return shank, i1, i2


def is_neighbor_pair(
    source: str,
    target: str,
    meta: pd.DataFrame,
) -> bool:
    """Return True if source/target are adjacent bipolar contacts on same hemi."""
    if source not in meta.index or target not in meta.index:
        return False
    hemi_src = str(meta.loc[source, 'hemi'])
    hemi_tgt = str(meta.loc[target, 'hemi'])
    if hemi_src != hemi_tgt:
        return False
    parsed_src = parse_bipolar_name(source)
    parsed_tgt = parse_bipolar_name(target)
    if not parsed_src or not parsed_tgt:
        return False
    shank_src, s1, s2 = parsed_src
    shank_tgt, t1, t2 = parsed_tgt
    if shank_src != shank_tgt:
        return False
    return len({s1, s2}.intersection({t1, t2})) > 0


def build_long_xcorr_df(
    xcorr_mean: np.ndarray,
    lag_times: np.ndarray,
    insula_chans: List[str],
    ifg_chans: List[str],
    ch_names: List[str],
    meta: pd.DataFrame,
    subject: str,
    task: str,
    phase: str,
    desc: str,
    band: str,
) -> pd.DataFrame:
    """Build long-format dataframe with one row per lag."""
    rows = []
    idx_map = {ch: idx for idx, ch in enumerate(ch_names)}
    for source in insula_chans:
        if source not in idx_map:
            continue
        src_meta = meta.loc[source]
        for target in ifg_chans:
            if target not in idx_map:
                continue
            tgt_meta = meta.loc[target]
            curve = xcorr_mean[idx_map[source], idx_map[target]]
            pair_df = pd.DataFrame({
                'subject': subject,
                'task': task,
                'phase': phase,
                'description': desc,
                'band': band,
                'source': source,
                'target': target,
                'source_roi': src_meta['roi'],
                'source_label': src_meta['label'],
                'source_hemi': src_meta['hemi'],
                'source_x': src_meta['x'],
                'source_y': src_meta['y'],
                'source_z': src_meta['z'],
                'target_roi': tgt_meta['roi'],
                'target_label': tgt_meta['label'],
                'target_hemi': tgt_meta['hemi'],
                'target_x': tgt_meta['x'],
                'target_y': tgt_meta['y'],
                'target_z': tgt_meta['z'],
                'is_neighbor': is_neighbor_pair(source, target, meta),
                'lag': lag_times,
                'xcorr': curve,
            })
            rows.append(pair_df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def main(
    bids_root: str,
    reference: str,
    recon_dir: str,
):
    band = "highgamma"
    datatype = 'epoch(band)(zscore)'
    bids_kwargs = dict(
        root=bids_root + f"derivatives/epoch({reference})",
        datatype=datatype,
        suffix=band,
        processing='Response',
        extension='.h5',
        check=False,
    )
    raw_pts = BIDSPath(**bids_kwargs)

    for raw_pt in tqdm(raw_pts.match(), desc='Processing subjects'):
        subject = raw_pt.subject
        phase = raw_pt.processing
        desc = raw_pt.description
        task = raw_pt.task

        epochs = mne.read_epochs(raw_pt, preload=True)

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

        parc.rename(columns={'name': 'channel'}, inplace=True)
        required_cols = ['channel', 'label', 'roi', 'hemi']
        if not all(col in parc.columns for col in required_cols):
            logger.warning(f"Skipping {subject}: missing columns in parcellation file")
            continue

        parc_sub = parc[required_cols]

        montage = epochs.get_montage()
        sub_id = re.sub(r'^D0+', 'D', subject)
        to_fsaverage = mne.read_talxfm(sub_id, recon_dir)
        trans = mne.transforms.Transform(fro='head', to='mri', trans=to_fsaverage['trans'])
        force2frame(montage, trans.from_str)
        montage.apply_trans(trans)
        pos_m = montage.get_positions()['ch_pos']

        cord_df = pd.DataFrame(pos_m).T
        cord_df.columns = ['x', 'y', 'z']
        cord_df[['x', 'y', 'z']] *= 1000
        cord_df = cord_df.reset_index().rename(columns={'index': 'channel'})
        cord_df = cord_df[cord_df.channel.isin(parc_sub.channel)]
        cord_df = cord_df.merge(parc_sub, on='channel', how='left')

        cord_df = cord_df[~cord_df['roi'].str.contains('white|intersection|unknown|WM', case=False, na=False)]
        cord_df = cord_df[~cord_df['roi'].isin(EXCLUDE_ROIS)]

        epochs.pick_channels(cord_df.channel.unique().tolist())

        insula_mask = cord_df['roi'].str.contains('ins', case=False, na=False)
        ifg_mask = cord_df['roi'].str.contains('ifg', case=False, na=False)
        insula_chans = cord_df[insula_mask]['channel'].tolist()
        ifg_chans = cord_df[ifg_mask]['channel'].tolist()

        if not insula_chans or not ifg_chans:
            logger.info(f"{subject}: Insula={len(insula_chans)}, IFG={len(ifg_chans)}, skipping")
            continue

        pick_chans = [ch for ch in epochs.ch_names if ch in set(insula_chans + ifg_chans)]
        epochs.pick_channels(pick_chans)
        ch_names = epochs.ch_names

        insula_chans = [ch for ch in insula_chans if ch in ch_names]
        ifg_chans = [ch for ch in ifg_chans if ch in ch_names]

        if not insula_chans or not ifg_chans:
            logger.info(f"{subject}: no overlapping Insula/IFG channels in epochs")
            continue

        xdata = epochs.get_data()
        xcorr, lag_times = compute_xcorr_matrix(xdata, epochs.info['sfreq'], max_lag_s=0.5)
        xcorr_mean = xcorr.mean(axis=0)

        meta = cord_df.set_index('channel').loc[ch_names]
        conn_df = build_long_xcorr_df(
            xcorr_mean,
            lag_times,
            insula_chans,
            ifg_chans,
            ch_names,
            meta,
            subject,
            task,
            phase,
            desc,
            band,
        )

        if conn_df.empty:
            logger.info(f"{subject}: no xcorr rows to save")
            continue

        save_path = raw_pt.copy().update(
            root=f'results/{task}({reference})',
            datatype='xcorr',
            suffix='wave',
            extension='.csv',
            check=False,
        )
        save_path.mkdir(exist_ok=True)
        conn_df.to_csv(save_path, index=False)
        logger.info(f"Saved {len(conn_df)} rows to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--bids_root",
    #     default="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/",
    #     type=str,
    # )
    
    parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.4_SentenceRep/BIDS/", type=str)
    # parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.3_PictureNaming/BIDS/", type=str)
    # parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.0_LexicalDecRepNoDelay/BIDS/", type=str)
    # parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/", type=str)
    parser.add_argument("--reference", type=str, default='bipolar', choices=['bipolar', 'car'])
    parser.add_argument(
        "--recon_dir",
        type=str,
        default=r"/cwork/ns458/ECoG_Recon/",
        help="path to the recon-all directory",
    )
    args = parser.parse_args()
    main(**vars(args))
