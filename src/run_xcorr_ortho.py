"""Orthogonalize Insula vs IFG neighbor pairs and save xcorr + traces.

For each subject/phase/description, identify neighboring Insula–IFG pairs
(same hemisphere, same shank prefix, adjacent bipolar indices). Orthogonalize
Insula signals against IFG signals (per trial), then compute trial-mean xcorr
curves (variance explained). Save long-format CSV (suffix=ortho) and save
per-trial traces as HDF5 via xarray (suffix=ortho, extension .h5).
"""

import argparse
import logging
import re
from typing import List, Tuple

import mne
import numpy as np
import pandas as pd
import xarray as xr
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


def parse_bipolar_name(ch_name: str) -> Tuple[str, int, int] | None:
    """Parse bipolar channel name into (shank_prefix, i1, i2)."""
    match = re.match(r"^[^_]+_(.*?)(\d+)-(\d+)$", ch_name)
    if not match:
        return None
    shank = match.group(1)
    i1 = int(match.group(2))
    i2 = int(match.group(3))
    return shank, i1, i2


def is_neighbor_pair(source: str, target: str, meta: pd.DataFrame) -> bool:
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


def orthogonalize_insula(ins_data: np.ndarray, ifg_data: np.ndarray) -> np.ndarray:
    """Remove IFG component from Insula per trial.

    Args:
        ins_data: (n_trials, n_time)
        ifg_data: (n_trials, n_time)

    Returns:
        ins_ortho: (n_trials, n_time)
    """
    n_trials, _ = ins_data.shape
    ins_ortho = np.zeros_like(ins_data)
    for i in range(n_trials):
        a_i = ins_data[i]
        b_i = ifg_data[i]
        beta = np.dot(a_i, b_i) / (np.dot(b_i, b_i) + 1e-10)
        ins_ortho[i] = a_i - beta * b_i
    return ins_ortho


def compute_pair_xcorr(
    ins_data: np.ndarray,
    ifg_data: np.ndarray,
    sfreq: float,
    max_lag_s: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute trial×lag xcorr for a single Insula–IFG pair (z-scored, squared)."""
    n_trials, n_time = ins_data.shape
    max_lag = int(max_lag_s * sfreq)
    lags = np.arange(-max_lag, max_lag + 1)
    lag_times = lags / sfreq

    mean_ins = ins_data.mean(axis=1, keepdims=True)
    std_ins = np.maximum(ins_data.std(axis=1, keepdims=True), np.finfo(float).eps)
    mean_ifg = ifg_data.mean(axis=1, keepdims=True)
    std_ifg = np.maximum(ifg_data.std(axis=1, keepdims=True), np.finfo(float).eps)
    ins_z = (ins_data - mean_ins) / std_ins
    ifg_z = (ifg_data - mean_ifg) / std_ifg

    xcorr = np.empty((n_trials, len(lags)), dtype=np.float32)
    for t in range(n_trials):
        full = correlate(ins_z[t], ifg_z[t], mode='full', method='auto')
        mid = len(full) // 2
        start = mid - max_lag
        stop = mid + max_lag + 1
        seg = full[start:stop] / n_time
        peak_sign = np.sign(seg[np.argmax(np.abs(seg))]) if seg.size else 1.0
        if peak_sign == 0:
            peak_sign = 1.0
        xcorr[t] = seg * peak_sign

    xcorr = xcorr ** 2
    return xcorr, lag_times


def build_long_xcorr_df(
    lag_times: np.ndarray,
    xcorr_mean: np.ndarray,
    source: str,
    target: str,
    meta: pd.DataFrame,
    subject: str,
    task: str,
    phase: str,
    desc: str,
    band: str,
    variant: str,
) -> pd.DataFrame:
    """Build long-format dataframe for one pair."""
    src_meta = meta.loc[source]
    tgt_meta = meta.loc[target]
    return pd.DataFrame({
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
        'is_neighbor': True,
        'variant': variant,
        'lag': lag_times,
        'xcorr': xcorr_mean,
    })


def main(
    bids_root: str,
    reference: str,
    recon_dir: str,
):
    band = "highgamma"
    datatype = 'epoch(band)(zscore)'
    raw_pts = BIDSPath(
        root=bids_root + f"derivatives/epoch({reference})",
        datatype=datatype,
        suffix=band,
        processing='Response',
        extension='.h5',
        check=False,
    )

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

        meta = cord_df.set_index('channel').loc[ch_names]

        neighbor_pairs: List[Tuple[str, str]] = []
        for ins in insula_chans:
            for ifg in ifg_chans:
                if is_neighbor_pair(ins, ifg, meta):
                    neighbor_pairs.append((ins, ifg))

        if not neighbor_pairs:
            logger.info(f"{subject}: no neighbor Insula–IFG pairs")
            continue

        xdata = epochs.get_data()
        times = epochs.times
        sfreq = epochs.info['sfreq']

        xcorr_before_means = []
        xcorr_after_means = []
        pair_names = []
        ins_traces = []
        ifg_traces = []
        ortho_traces = []
        source_list = []
        target_list = []

        name_to_idx = {ch: idx for idx, ch in enumerate(ch_names)}
        for ins, ifg in neighbor_pairs:
            ins_data = xdata[:, name_to_idx[ins], :]
            ifg_data = xdata[:, name_to_idx[ifg], :]
            ins_ortho = orthogonalize_insula(ins_data, ifg_data)

            xcorr_before, lag_times = compute_pair_xcorr(ins_data, ifg_data, sfreq)
            xcorr_after, _ = compute_pair_xcorr(ins_ortho, ifg_data, sfreq)

            xcorr_before_means.append(xcorr_before.mean(axis=0))
            xcorr_after_means.append(xcorr_after.mean(axis=0))

            pair_names.append(f"{ins}__{ifg}")
            source_list.append(ins)
            target_list.append(ifg)
            ins_traces.append(ins_data)
            ifg_traces.append(ifg_data)
            ortho_traces.append(ins_ortho)

        xcorr_before_arr = np.stack(xcorr_before_means, axis=0)
        xcorr_after_arr = np.stack(xcorr_after_means, axis=0)
        xcorr_arr = np.stack([xcorr_before_arr, xcorr_after_arr], axis=1)

        ds = xr.Dataset(
            data_vars={
                'insula': (('pair', 'trial', 'time'), np.stack(ins_traces, axis=0)),
                'ifg': (('pair', 'trial', 'time'), np.stack(ifg_traces, axis=0)),
                'insula_ortho': (('pair', 'trial', 'time'), np.stack(ortho_traces, axis=0)),
                'xcorr': (('pair', 'variant', 'lag'), xcorr_arr),
            },
            coords={
                'pair': np.array(pair_names),
                'trial': np.arange(xdata.shape[0]),
                'time': times,
                'lag': lag_times,
                'variant': np.array(['before', 'after']),
                'source': ('pair', np.array(source_list)),
                'target': ('pair', np.array(target_list)),
                'is_neighbor': ('pair', np.ones(len(pair_names), dtype=bool)),
            },
            attrs={
                'subject': subject,
                'task': task,
                'phase': phase,
                'description': desc,
                'band': band,
            },
        )

        trace_path = raw_pt.copy().update(
            root=f'results/{task}({reference})',
            datatype='xcorr',
            suffix='ortho',
            extension='.h5',
            check=False,
        )
        trace_path.mkdir(exist_ok=True)

        try:
            ds.to_netcdf(trace_path, engine='h5netcdf')
        except ModuleNotFoundError:
            try:
                ds.to_netcdf(trace_path, engine='netcdf4')
            except ModuleNotFoundError:
                import h5py

                with h5py.File(trace_path, 'w') as h5:
                    h5.create_dataset('insula', data=ds['insula'].values)
                    h5.create_dataset('ifg', data=ds['ifg'].values)
                    h5.create_dataset('insula_ortho', data=ds['insula_ortho'].values)
                    h5.create_dataset('xcorr', data=ds['xcorr'].values)
                    h5.create_dataset('time', data=ds['time'].values)
                    h5.create_dataset('lag', data=ds['lag'].values)
                    h5.create_dataset('pair', data=ds['pair'].values.astype('S'))
                    h5.create_dataset('variant', data=ds['variant'].values.astype('S'))
                    h5.create_dataset('source', data=ds['source'].values.astype('S'))
                    h5.create_dataset('target', data=ds['target'].values.astype('S'))
                    h5.create_dataset('is_neighbor', data=ds['is_neighbor'].values)
                    for key, value in ds.attrs.items():
                        h5.attrs[key] = value

        logger.info(f"Saved traces to {trace_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--bids_root",
    #     default="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/",
    #     type=str,
    # )
    
    # parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.4_SentenceRep/BIDS/", type=str)
    # parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.3_PictureNaming/BIDS/", type=str)
    # parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.0_LexicalDecRepNoDelay/BIDS/", type=str)
    parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/", type=str)
    parser.add_argument("--reference", type=str, default='bipolar', choices=['bipolar', 'car'])
    parser.add_argument(
        "--recon_dir",
        type=str,
        default=r"/cwork/ns458/ECoG_Recon/",
        help="path to the recon-all directory",
    )
    args = parser.parse_args()
    main(**vars(args))
