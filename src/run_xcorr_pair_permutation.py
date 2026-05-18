import argparse
import logging
import os
import re
from itertools import combinations
from typing import Any, Dict, List, Tuple

import mne
import numpy as np
import pandas as pd
from scipy.signal import correlate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _zscore_trials(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = np.maximum(x.std(axis=1, keepdims=True), np.finfo(float).eps)
    return (x - mean) / std


def compute_pair_xcorr_trials(
    source_trials: np.ndarray,
    target_trials: np.ndarray,
    sfreq: float,
    max_lag_s: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute trial-wise xcorr curve for one electrode pair.

    Args:
        source_trials: (n_trials, n_time)
        target_trials: (n_trials, n_time)
        sfreq: sampling rate
        max_lag_s: symmetric lag window in seconds

    Returns:
        xcorr_trials: (n_trials, n_lag), squared and sign-normalized xcorr
        lag_times: (n_lag,)
    """
    if source_trials.shape != target_trials.shape:
        raise ValueError("source_trials and target_trials must have same shape")

    n_trials, n_time = source_trials.shape
    max_lag = int(max_lag_s * sfreq)
    lags = np.arange(-max_lag, max_lag + 1)
    lag_times = lags / sfreq

    zsrc = _zscore_trials(source_trials)
    ztgt = _zscore_trials(target_trials)

    xcorr_trials = np.empty((n_trials, len(lags)), dtype=np.float32)
    for t in range(n_trials):
        full = correlate(zsrc[t], ztgt[t], mode='full', method='auto')
        mid = len(full) // 2
        start = mid - max_lag
        stop = mid + max_lag + 1
        seg = full[start:stop] / n_time

        peak_sign = np.sign(seg[np.argmax(np.abs(seg))]) if seg.size else 1.0
        if peak_sign == 0:
            peak_sign = 1.0

        xcorr_trials[t] = (seg * peak_sign) ** 2

    return xcorr_trials, lag_times


def build_trial_shuffle_null(
    source_trials: np.ndarray,
    target_trials: np.ndarray,
    sfreq: float,
    max_lag_s: float = 1.0,
    n_perm: int = 1000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build trial-shuffle null distribution of mean xcorr curves.

    Each permutation shuffles target trial order and recomputes trial-averaged xcorr.
    """
    rng = np.random.RandomState(random_state)
    n_trials = source_trials.shape[0]

    null_curves = []
    lag_times = None

    for _ in range(n_perm):
        perm_idx = rng.permutation(n_trials)
        xcorr_perm, lag_times = compute_pair_xcorr_trials(
            source_trials,
            target_trials[perm_idx],
            sfreq=sfreq,
            max_lag_s=max_lag_s,
        )
        null_curves.append(xcorr_perm.mean(axis=0))

    return np.asarray(null_curves, dtype=np.float32), lag_times


def build_circular_shift_null(
    source_trials: np.ndarray,
    target_trials: np.ndarray,
    sfreq: float,
    max_lag_s: float = 1.0,
    n_perm: int = 1000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build circular-shift null distribution of mean xcorr curves.

    Each permutation circularly shifts target in time independently per trial.
    """
    rng = np.random.RandomState(random_state)
    n_trials, n_time = target_trials.shape
    if n_time < 2:
        raise ValueError("target_trials must contain at least 2 time points for circular shift null.")

    null_curves = []
    lag_times = None

    for _ in range(n_perm):
        shifted = np.empty_like(target_trials)
        shifts = rng.randint(1, n_time, size=n_trials)
        for t, s in enumerate(shifts):
            shifted[t] = np.roll(target_trials[t], int(s))

        xcorr_perm, lag_times = compute_pair_xcorr_trials(
            source_trials,
            shifted,
            sfreq=sfreq,
            max_lag_s=max_lag_s,
        )
        null_curves.append(xcorr_perm.mean(axis=0))

    return np.asarray(null_curves, dtype=np.float32), lag_times


def _find_true_clusters(mask: np.ndarray) -> List[Tuple[int, int]]:
    clusters: List[Tuple[int, int]] = []
    start = None

    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            clusters.append((start, i))
            start = None

    if start is not None:
        clusters.append((start, len(mask)))

    return clusters


def cluster_permutation_test(
    observed_curve: np.ndarray,
    null_curves: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, object]:
    """Cluster permutation test over lag dimension using trial-shuffle null.

    Tail is one-sided (observed > null), suited to squared xcorr measure.
    """
    if null_curves.ndim != 2:
        raise ValueError("null_curves must be 2D: (n_perm, n_lag)")

    null_mean = null_curves.mean(axis=0)
    point_thresh = np.quantile(null_curves, 1.0 - alpha, axis=0)

    observed_mask = observed_curve > point_thresh
    observed_stat = observed_curve - null_mean
    observed_clusters = _find_true_clusters(observed_mask)
    observed_masses = [float(observed_stat[s:e].sum()) for s, e in observed_clusters]

    max_masses = np.zeros(null_curves.shape[0], dtype=np.float32)
    for p in range(null_curves.shape[0]):
        perm_curve = null_curves[p]
        perm_mask = perm_curve > point_thresh
        perm_stat = perm_curve - null_mean
        perm_clusters = _find_true_clusters(perm_mask)
        if perm_clusters:
            masses = [perm_stat[s:e].sum() for s, e in perm_clusters]
            max_masses[p] = float(np.max(masses))

    clusters = []
    for (start, end), mass in zip(observed_clusters, observed_masses):
        p_cluster = (np.sum(max_masses >= mass) + 1.0) / (len(max_masses) + 1.0)
        clusters.append(
            {
                'start_idx': int(start),
                'end_idx': int(end),
                'mass': float(mass),
                'p_value': float(p_cluster),
                'significant': bool(p_cluster < alpha),
            }
        )

    pointwise_p = (1.0 + (null_curves >= observed_curve[None, :]).sum(axis=0)) / (null_curves.shape[0] + 1.0)

    return {
        'null_mean': null_mean,
        'point_threshold': point_thresh,
        'pointwise_p': pointwise_p,
        'clusters': clusters,
        'max_cluster_masses': max_masses,
    }


def load_pair_trials(
    bids_root: str,
    subject: str,
    phase: str,
    description: str,
    band: str,
    reference: str,
) -> Tuple[mne.Epochs, Any, str]:
    """Load significant HGA epochs for one subject/condition."""
    from mne_bids import BIDSPath

    if band == 'raw':
        raise ValueError("This script requires significant HGA input; --band raw is not supported.")

    root = os.path.join(bids_root, f'derivatives/epoch({reference})')
    datatypes = ['epoch(band)(sig)(effective)']

    matches = []
    for datatype in datatypes:
        bp = BIDSPath(
            root=root,
            datatype=datatype,
            subject=subject,
            task=None,
            processing=phase,
            description=description,
            suffix=band,
            extension='.h5',
            check=False,
        )
        matches = bp.match()
        if len(matches) > 0:
            break

    if len(matches) == 0:
        candidate_bp = BIDSPath(
            root=root,
            datatype='epoch(band)(sig)(effective)',
            subject=subject,
            task=None,
            processing=None,
            description=None,
            suffix=band,
            extension='.h5',
            check=False,
        )
        candidates = candidate_bp.match()
        candidate_msg = ", ".join(str(p) for p in candidates[:5]) if candidates else "none"
        raise FileNotFoundError(
            f"No significant epoch file found for subject={subject}, phase={phase}, "
            f"description={description}, band={band}. "
            f"Available candidates ({len(candidates)}): {candidate_msg}"
        )

    task_set = sorted({m.task for m in matches if m.task is not None})
    if len(task_set) != 1:
        show = ", ".join(str(m) for m in matches[:5])
        raise ValueError(f"Expected one task in {root}, found tasks={task_set}. Matches: {show}")

    epoch_file = matches[0]
    epochs = mne.read_epochs(epoch_file, preload=True, verbose='error')
    return epochs, epoch_file, str(epoch_file.task)


def _safe_name(x: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '-', x).strip('-')


def _parse_bipolar_contacts(channel: str) -> Tuple[str, int, int] | None:
    m = re.match(r'^(.*?)(\d+)-(\d+)$', channel)
    if m is None:
        return None
    stem = m.group(1)
    c1 = int(m.group(2))
    c2 = int(m.group(3))
    return stem, c1, c2


def is_adjacent_bipolar_pair(source_channel: str, target_channel: str) -> bool:
    src = _parse_bipolar_contacts(source_channel)
    tgt = _parse_bipolar_contacts(target_channel)
    if src is None or tgt is None:
        return False
    src_stem, s1, s2 = src
    tgt_stem, t1, t2 = tgt
    if src_stem != tgt_stem:
        return False
    return len({s1, s2}.intersection({t1, t2})) > 0


def main(
    bids_root: str,
    subject: str,
    phase: str,
    description: str,
    band: str,
    reference: str,
    max_lag_s: float,
    n_perm: int,
    alpha: float,
    random_state: int,
):
    import xarray as xr

    logger.info('Loading subject data: %s', subject)
    epochs, epoch_path, task = load_pair_trials(
        bids_root=bids_root,
        subject=subject,
        phase=phase,
        description=description,
        band=band,
        reference=reference,
    )

    # Phase-specific time window crop to avoid cross-phase contamination.
    PHASE_WINDOWS = {
        'Stimulus': (0.0, 1.0),
        'Delay':    (0.0, 1.0),
        'Go':       (0.0, 1.0),
        'Response': (-0.5, 0.5),
    }
    if phase not in PHASE_WINDOWS:
        raise ValueError(f"Unknown phase '{phase}'. Expected one of {list(PHASE_WINDOWS)}.")
    tmin_crop, tmax_crop = PHASE_WINDOWS[phase]
    epochs.crop(tmin=tmin_crop, tmax=tmax_crop)
    logger.info('Cropped epochs to phase=%s window [%.2f, %.2f]s', phase, tmin_crop, tmax_crop)

    data = epochs.get_data()
    ch_names = epochs.ch_names
    sfreq = float(epochs.info['sfreq'])
    n_trials, n_chan, n_time = data.shape
    logger.info('Loaded epochs: %s', str(epoch_path))
    logger.info('Trials: %d, Channels: %d, Time points: %d, sfreq: %.2f', n_trials, n_chan, n_time, sfreq)

    candidate_pairs = list(combinations(range(n_chan), 2))
    pair_indices = [
        (i, j) for i, j in candidate_pairs
        if not is_adjacent_bipolar_pair(ch_names[i], ch_names[j])
    ]

    if len(pair_indices) == 0:
        raise ValueError('No valid channel pairs found after filtering.')

    logger.info('Evaluating %d pairs (adjacent bipolar pairs removed)', len(pair_indices))

    observed_rows = []
    null_mean_rows = []
    point_thresh_rows = []
    pointwise_p_rows = []
    pair_src = []
    pair_tgt = []
    cluster_rows = []
    lag_times = None

    for k, (i, j) in enumerate(pair_indices, start=1):
        source_channel = ch_names[i]
        target_channel = ch_names[j]
        src_trials = data[:, i, :]
        tgt_trials = data[:, j, :]

        xcorr_trials, lag_times = compute_pair_xcorr_trials(
            src_trials,
            tgt_trials,
            sfreq=sfreq,
            max_lag_s=max_lag_s,
        )
        observed_curve = xcorr_trials.mean(axis=0)

        null_curves, _ = build_trial_shuffle_null(
            src_trials,
            tgt_trials,
            sfreq=sfreq,
            max_lag_s=max_lag_s,
            n_perm=n_perm,
            random_state=random_state,
        )

        test_res = cluster_permutation_test(
            observed_curve=observed_curve,
            null_curves=null_curves,
            alpha=alpha,
        )

        pair_src.append(source_channel)
        pair_tgt.append(target_channel)
        observed_rows.append(observed_curve.astype(np.float32))
        null_mean_rows.append(test_res['null_mean'].astype(np.float32))
        point_thresh_rows.append(test_res['point_threshold'].astype(np.float32))
        pointwise_p_rows.append(test_res['pointwise_p'].astype(np.float32))

        for cl in test_res['clusters']:
            if not cl['significant']:
                continue
            s = cl['start_idx']
            e = cl['end_idx']
            cluster_rows.append(
                {
                    'pair_idx': k - 1,
                    'source_channel': source_channel,
                    'target_channel': target_channel,
                    'lag_start_s': float(lag_times[s]),
                    'lag_end_s': float(lag_times[e - 1]),
                    'mass': float(cl['mass']),
                    'p_value': float(cl['p_value']),
                    'significant': bool(cl['significant']),
                }
            )

        if k % 100 == 0 or k == len(pair_indices):
            logger.info('Processed %d/%d pairs', k, len(pair_indices))

    ds = xr.Dataset(
        data_vars={
            'observed_curve': (('pair', 'lag'), np.asarray(observed_rows, dtype=np.float32)),
            'null_mean': (('pair', 'lag'), np.asarray(null_mean_rows, dtype=np.float32)),
            'point_threshold': (('pair', 'lag'), np.asarray(point_thresh_rows, dtype=np.float32)),
            'pointwise_p': (('pair', 'lag'), np.asarray(pointwise_p_rows, dtype=np.float32)),
        },
        coords={
            'pair': np.arange(len(pair_indices), dtype=np.int32),
            'lag': lag_times.astype(np.float32),
            'source_channel': ('pair', np.asarray(pair_src, dtype=str)),
            'target_channel': ('pair', np.asarray(pair_tgt, dtype=str)),
        },
        attrs={
            'subject': subject,
            'task': task,
            'phase': phase,
            'description': description,
            'band': band,
            'reference': reference,
            'null_model': 'trial_shuffle',
            'n_perm': int(n_perm),
            'alpha': float(alpha),
            'skip_adjacent_bipolar': 1,
        },
    )

    from mne_bids import BIDSPath
    out_nc = BIDSPath(
        root=f'results/{task}({reference})',
        description=description,
        datatype='xcorr',
        suffix='perm',
        recording=epoch_path.recording if epoch_path.recording is not None else band,
        task=task,
        subject=subject,
        processing=phase,
        extension='.nc',
        check=False,
    )
    out_clusters = out_nc.copy().update(extension='.csv')

    out_nc.mkdir(exist_ok=True)
    ds.to_netcdf(out_nc)
    cluster_cols = ['pair_idx', 'source_channel', 'target_channel', 'lag_start_s', 'lag_end_s', 'mass', 'p_value', 'significant']
    pd.DataFrame(cluster_rows, columns=cluster_cols).to_csv(out_clusters, index=False)

    total_sig = len(cluster_rows)
    logger.info('Saved waveform dataset: %s', out_nc)
    logger.info('Saved cluster table: %s', out_clusters)
    logger.info('Significant clusters: %d', total_sig)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/", type=str)
    parser.add_argument('--subject', type=str, required=True)
    parser.add_argument('--phase', type=str, default='Response')
    parser.add_argument('--description', type=str, default='Repeat')
    parser.add_argument('--band', type=str, default='highgamma', choices=['highgamma', 'lowband'])
    parser.add_argument('--reference', type=str, default='bipolar', choices=['bipolar', 'car'])
    parser.add_argument('--max_lag_s', type=float, default=1.0)
    parser.add_argument('--n_perm', type=int, default=1000)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--random_state', type=int, default=42)
    return parser


if __name__ == '__main__':
    parser = build_argparser()
    args = parser.parse_args()
    main(**vars(args))
