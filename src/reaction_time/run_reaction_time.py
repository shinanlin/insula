
import argparse
import sys

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath
from sklearn.pipeline import make_pipeline
from himalaya.ridge import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from tqdm import tqdm
import h5py
from ieeg.calc.oversample import mixup
from einops import rearrange

from src.paths import SUPPORTED_ATLASES, hga_results_dir

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)
RANDOM_SEED = 42

def load_subject_data(
    subject: str,
    epoch_paths: BIDSPath,
    ref: str,
    atlas: str = "hammers",
) -> pd.DataFrame:
    """
    Load and preprocess epoch data for a single subject.
    
    Returns a long-format DataFrame with columns:
        time, channel, value, epoch, onset, phase, description, subject, 
        condition, remark, trial, rt
    
    Returns None if data cannot be loaded.
    """
    if atlas not in SUPPORTED_ATLASES:
        raise ValueError(f"atlas must be one of {SUPPORTED_ATLASES}, got {atlas!r}")

    subj_pts = epoch_paths.copy().update(subject=subject).match()
    if not subj_pts:
        logging.warning(f"No epoch files for {subject}")
        return None
    
    # Load parcellation (same pattern as src/hga/package_highgamma.py)
    try:
        parc_matches = subj_pts[0].copy().update(
            root=str(subj_pts[0].root).replace(f'epoch({ref})', 'parcellation'),
            datatype=ref,
            task=None,
            description=None,
            recording=None,
            processing=None,
            suffix=atlas,
            extension='.csv',
        ).match()
        if not parc_matches:
            raise IndexError(f"no {atlas} parcellation file matched")
        parc = pd.read_csv(parc_matches[0])
        parc = parc[~parc['roi'].str.contains('white|intersection|unknown|WM', case=False, na=False)]
        parc = parc.dropna(subset=['roi'])
    except Exception as e:
        logging.warning(f"Parc file not found for {subject}: {e}")
        return None
    
    picks = parc['name'].unique().tolist()
    dfs = []
    
    for pt in subj_pts:
        try:
            epochs = mne.read_epochs(pt, preload=True)
        except Exception as e:
            logging.warning(f"Cannot load {pt}: {e}")
            continue
        
        # Load significance mask
        stats_path = pt.copy().update(
            root=str(pt.root).replace(f'epoch({ref})', 'statistics'),
            datatype=ref,
            extension='.h5',
        )
        
        epochs.metadata = pd.DataFrame(
            {'onset': epochs.events[:, 0] / 2048}, 
            index=epochs.events[:, -1]
        )
        epoch_picks = [ch for ch in picks if ch in epochs.ch_names]
        if not epoch_picks:
            logging.warning(
                f"No {atlas} channels found in {pt}; skipping epoch file"
            )
            continue
        epochs.pick(epoch_picks)
        
        df = epochs.to_data_frame(long_format=True, scalings={'seeg': 1}, verbose=False)
        meta = epochs.metadata.reset_index(names='epoch')
        df = df.merge(meta, on='epoch', how='left')
        df = df.drop(columns=['ch_type'])
        
        # Merge significance mask
        with h5py.File(stats_path, 'r') as stats:
            mask_df = pd.DataFrame(
                index=[ch.decode('utf-8') for ch in stats['ch_names'][:]],
                columns=epochs.times,
                data=stats['mask'][:],
            )
        mask_long = (
            mask_df.reset_index()
            .melt(id_vars='index', var_name='time', value_name='mask')
            .rename(columns={'index': 'channel'})
        )
        mask_long = mask_long[mask_long['channel'].isin(df['channel'])]
        df = df.merge(mask_long, on=['channel', 'time'], how='left')
        df['mask'] = df['mask'].fillna(False).astype(bool)
        
        df['phase'] = pt.processing
        df['description'] = pt.description
        df['subject'] = subject
        df['remark'] = df['condition'].str.split('/').str[4:].str.join('/')
        df['condition'] = df['condition'].str.split('/').str[2:4].str.join('/')
        
        dfs.append(df)
    
    if not dfs:
        return None
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Keep only significant channels
    df['sig'] = df.groupby('channel')['mask'].transform('any')
    df = df[df['sig']].drop(columns=['sig', 'mask'])
    
    # Compute RT per trial
    trial_meta = extract_reaction_time(df)
    df = df.merge(
        trial_meta[['subject', 'description', 'epoch', 'trial', 'rt']],
        on=['subject', 'description', 'epoch'],
        how='left'
    )
    # filter to CORRECT only when available; otherwise keep all rows
    correct_mask = df['remark'].str.contains('CORRECT', na=False)
    if correct_mask.any():
        df = df[correct_mask]
    return df


def extract_reaction_time(epos_df):
    """
    Extract trial-level metadata and reaction time (RT) from a long-format epos DataFrame.
    
    Parameters
    ----------
    epos_df : pd.DataFrame
        Long-format DataFrame with columns:
        ['subject', 'epoch', 'condition', 'description', 'phase', 'onset', 'remark', ...]
        Each row corresponds to a single timepoint within an epoch.
    
    Returns
    -------
    epoch_tbl : pd.DataFrame
        Trial-level table (one row per epoch) with the following columns:
        - subject, epoch, condition, description, phase, onset, remark
        - trial : unique trial number (within each subject)
        - rt    : reaction time (resp_onset - go_onset) for the trial
        Only correct trials are retained.
    """
    # ----------------------------------------------------------------------
    # 1) Build a trial-level table from epoch-level metadata (one row per epoch)
    # ----------------------------------------------------------------------
    epoch_tbl = (
        epos_df[['subject', 'epoch', 'condition', 'description', 'phase', 'onset', 'remark']]
        .drop_duplicates()
    )

    # ----------------------------------------------------------------------
    # 2) Assign a unique trial number (trial) to each real trial.
    #    - Trials are defined by the onset of the Stimulus phase.
    #    - Within each subject, we count Stimulus events in chronological order.
    # ----------------------------------------------------------------------
    epoch_tbl = epoch_tbl.sort_values(['subject', 'onset', 'epoch']).reset_index(drop=True)

    epoch_tbl['trial'] = (
        epoch_tbl
        .groupby(['subject'], sort=False)['phase']
        .transform(lambda s: s.eq('Stimulus').cumsum())
    )
    # Rows before the first Stimulus (if any) will have trial=0; set them to NaN.
    epoch_tbl.loc[epoch_tbl['trial'].eq(0), 'trial'] = np.nan

    # ----------------------------------------------------------------------
    # 3) Extract Go and Response onsets for each trial and compute RT.
    #    - Pivot the long table so each trial has one row with go_onset and resp_onset.
    #    - RT = resp_onset - go_onset.
    # ----------------------------------------------------------------------
    
    # RT rule: if Go exists, RT = Response - Go; otherwise RT = Response - Stimulus
    
    go_resp_onset = (
        epoch_tbl[epoch_tbl['phase'].isin(['Stimulus', 'Go', 'Response'])]
        .pivot_table(
            index=['subject', 'description', 'trial', 'condition'],
            columns='phase',
            values='onset',
            aggfunc='first'
        )
        .reindex(columns=['Stimulus', 'Go', 'Response'])
        .reset_index()
        .rename(columns={'Stimulus': 'stim_onset', 'Go': 'go_onset', 'Response': 'resp_onset'})
    )

    go_resp_onset['rt'] = np.where(
        go_resp_onset['go_onset'].notna(),
        go_resp_onset['resp_onset'] - go_resp_onset['go_onset'],
        go_resp_onset['resp_onset'] - go_resp_onset['stim_onset'],
    )

    # ----------------------------------------------------------------------
    # 4) Merge RT back into the epoch-level table.
    #    - Use subject, description, trial, and condition as the join key.
    # ----------------------------------------------------------------------
    epoch_tbl = epoch_tbl.merge(
        go_resp_onset[['subject', 'description', 'trial', 'condition', 'rt']],
        on=['subject', 'description', 'trial', 'condition'],
        how='left'
    )

    return epoch_tbl


def _prepare_data(
    df: pd.DataFrame,
    phase: str,
    description: str,
):
    """Prepare 3D data array and RT vector from long-format DataFrame.
    
    Returns
    -------
    X_3d : ndarray, shape (n_trials, n_channels, n_times)
    rt : ndarray, shape (n_trials,)
    times : ndarray
    channels : ndarray
    Returns (None, None, None, None) if insufficient data.
    """
    sub = df[(df['phase'] == phase) & (df['description'] == description)]
    if sub.empty:
        return None, None, None, None

    times = np.sort(sub['time'].unique())
    trials = sub['trial'].dropna().unique()
    channels = sub['channel'].unique()
    if len(trials) == 0 or len(channels) == 0 or len(times) == 0:
        return None, None, None, None

    trial_to_rt = sub.groupby('trial')['rt'].first().to_dict()
    rt = np.array([trial_to_rt[t] for t in trials])

    valid = ~np.isnan(rt)

    trials = trials[valid]
    rt = rt[valid]
    if len(trials) < 2:
        return None, None, None, None

    # Build 3D array via pivot. Pandas groupby sorts its coordinates, so select
    # them back into the explicit trial/channel order before dropping xarray's
    # coordinate labels. This keeps X, RT, and saved channel names aligned.
    valid_sub = sub[sub['trial'].isin(trials)]
    X_da = (valid_sub
        .groupby(['trial','channel','time'])['value']
        .mean()
        .to_xarray()
        .transpose('trial', 'channel', 'time')
        .sel(trial=trials, channel=channels, time=times)
    )

    trials = X_da.coords['trial'].values
    channels = X_da.coords['channel'].values
    times = X_da.coords['time'].values
    rt = np.array([trial_to_rt[t] for t in trials])
    X_3d = X_da.values

    expected_shape = (len(rt), len(channels), len(times))
    if X_3d.shape != expected_shape:
        raise RuntimeError(
            "Reaction-time data axes are misaligned: "
            f"X.shape={X_3d.shape}, expected={expected_shape}"
        )

    # mixup, in-place
    mixup(X_3d, obs_axis=0, rng=42)

    return X_3d, rt, times, channels


def cluster_correction(scores, baseline, p_thresh=0.05, tails=1):

    from ieeg.calc.stats import time_cluster, proportion, tail_compare

    # scores: (n_times, n_channels)
    # baseline: (n_times, n_channels, n_perm)
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D (n_times, n_channels), got {scores.shape}")
    if baseline.ndim != 3:
        raise ValueError(f"baseline must be 3D (n_times, n_channels, n_perm), got {baseline.shape}")
    if scores.shape[0] != baseline.shape[0] or scores.shape[1] != baseline.shape[1]:
        raise ValueError(
            f"scores and baseline must match in (n_times, n_channels). "
            f"Got scores={scores.shape}, baseline={baseline.shape}"
        )

    n_times, n_channels = scores.shape
    n_perm = baseline.shape[2]

    mask = np.zeros((n_channels, n_times), dtype=bool)
    p_act = np.ones((n_channels, n_times), dtype=float)

    for ch in range(n_channels):
        sc = scores[:, ch]
        base = baseline[:, ch, :].T  # (n_perm, n_times)

        diff = base - sc[None, :]
        p_ch = (np.sum(diff >= 0, axis=0) + 1) / (diff.shape[0] + 1)
        p_perm = proportion(diff, tail=tails, axis=0)
        b_act = tail_compare(1. - p_ch, 1. - p_thresh, tails)
        b_perm = tail_compare(p_perm, 1. - p_thresh, tails)
        mask[ch, :] = time_cluster(b_act, b_perm, 1 - p_thresh, tails)
        p_act[ch, :] = p_ch

    return mask, p_act



def predict_permutation_scores(
    X,
    rt,
    pipeline,
    cv,
    n_perm=1000,
    random_state=42,
    n_jobs=-1,
):
    """
    Univariate channel-wise prediction with permutation test.
    X: (n_trials, n_channels, n_times)
    rt: (n_trials,)
    Returns:
        obs_scores: (n_channels,) Pearson r for each channel
        perm_scores: (n_channels, n_perm) permutation Pearson r for each channel
        p_values: (n_channels,) one-sided p-value (obs > perm) for each channel
    """
    from sklearn.base import clone

    def _pearson_r(y_true, y_pred):
        if y_true.size < 2:
            return np.nan
        y_true_std = np.std(y_true)
        y_pred_std = np.std(y_pred)
        if y_true_std == 0 or y_pred_std == 0:
            return np.nan
        y_true_z = (y_true - np.mean(y_true)) / y_true_std
        y_pred_z = (y_pred - np.mean(y_pred)) / y_pred_std
        return float(np.mean(y_true_z * y_pred_z))

    n_trials, n_channels, n_times = X.shape
    # Use entire time window as features (no averaging)
    logger.info(f"[predict_permutation_scores] Using entire time window ({n_times} time points) as features")
    
    if n_trials < 2 or n_channels < 1:
        raise ValueError("Insufficient data for prediction")

    obs_scores = np.full(n_channels, np.nan)
    perm_scores = np.full((n_channels, n_perm), np.nan)
    p_values = np.full(n_channels, np.nan)

    effective_n_jobs = n_jobs
    if effective_n_jobs == 0:
        raise ValueError("n_jobs must be non-zero")
    if effective_n_jobs is None or effective_n_jobs < 0:
        effective_n_jobs = -1
    else:
        effective_n_jobs = int(min(effective_n_jobs, n_channels))

    def _one_channel(ch_idx):
        rng = np.random.RandomState(random_state + int(ch_idx))
        x_ch = X[:, ch_idx, :]  # (n_trials, n_times)

        fold_obs = []
        fold_perm = []

        for tr, te in cv.split(x_ch, rt):
            x_train, x_test = x_ch[tr], x_ch[te]
            y_train, y_test = rt[tr], rt[te]

            train_ok = ~np.isnan(x_train).any(axis=1) & ~np.isnan(y_train)
            test_ok = ~np.isnan(x_test).any(axis=1) & ~np.isnan(y_test)
            if train_ok.sum() < 2 or test_ok.sum() < 2:
                continue

            x_train = x_train[train_ok]
            y_train = y_train[train_ok]
            x_test = x_test[test_ok]
            y_test = y_test[test_ok]

            if np.std(y_test) == 0:
                continue

            seeds_fold = rng.randint(0, 2**31 - 1, size=n_perm)
            y_train_all = np.empty((len(y_train), 1 + n_perm))
            y_train_all[:, 0] = y_train

            for i, seed in enumerate(seeds_fold):
                r = np.random.RandomState(seed)
                y_perm = y_train.copy()
                r.shuffle(y_perm)
                y_train_all[:, i + 1] = y_perm

            dec = clone(pipeline)
            dec.fit(x_train, y_train_all)

            y_pred_all = dec.predict(x_test)
            score_all = np.array([_pearson_r(y_test, y_pred_all[:, i]) for i in range(1 + n_perm)])

            fold_obs.append(score_all[0])
            fold_perm.append(score_all[1:])

        if len(fold_obs) == 0:
            return ch_idx, np.nan, np.full(n_perm, np.nan), np.nan

        obs = float(np.nanmean(fold_obs))
        perm = np.nanmean(np.stack(fold_perm, axis=0), axis=0)
        valid_perm = np.isfinite(perm)
        if not np.isfinite(obs) or valid_perm.sum() == 0:
            return ch_idx, np.nan, perm, np.nan
        p_val = (np.sum(perm[valid_perm] >= obs) + 1.0) / (valid_perm.sum() + 1.0)
        return ch_idx, obs, perm, p_val

    results = Parallel(n_jobs=effective_n_jobs, batch_size=1)(
        delayed(_one_channel)(ch_idx) for ch_idx in range(n_channels)
    )
    for ch_idx, obs, perm, p_val in results:
        obs_scores[ch_idx] = obs
        perm_scores[ch_idx, :] = perm
        p_values[ch_idx] = p_val

    return obs_scores, perm_scores, p_values

def main(bids_root: str, 
         ref: str, 
         band: str,
         subject: str,
         window: float,
         step: float,
         n_folds: int,
         n_perm: int,
         n_jobs: int,
         atlas: str = "hammers",
         ):
    
    """Main entry point: load data for all subjects and predict RT."""
    
    epoch_paths = BIDSPath(
        root=bids_root + f"derivatives/epoch({ref})",
        suffix=band,
        subject=subject,
        datatype='epoch(band)(zscore)',
        extension=".h5",
        check=False,
    )
    
    alphas = np.logspace(-3, 3, 10)
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    
    pipeline=make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=alphas, fit_intercept=True),
    )
    
    df = load_subject_data(subject, epoch_paths, ref, atlas=atlas)
    if df is None or df.empty:
        logger.warning(f"No usable data for {subject}; skipping")
        return
    
    # remove any passive epochs
    df = df[df.description != 'Passive']
    if df.empty:
        logger.warning(f"No non-Passive data for {subject}; skipping")
        return
    
    for phase in df.phase.unique():
        
        for desc in df.description.unique():
            
            X, rt, times, channels = _prepare_data(df, phase, desc)
            if X is None or len(rt) < 2:
                logger.warning(
                    f"Insufficient trials for {subject} | {phase} | {desc}; skipping"
                )
                continue
            
            tmin, tmax = times.min(), times.max()
            fs = 128
            time_points = np.arange(tmin + window,
                        tmax + step,
                        step)
            
            window_samples = int(window * fs)
            step_samples = int(step * fs)
            
            scores = np.full((len(time_points), len(channels)), np.nan)
            perm_scores = np.full((len(time_points), len(channels), n_perm), np.nan)
            pvals = np.full((len(time_points), len(channels)), np.nan)
            
            for t_idx, time_end in enumerate(
                tqdm(
                    time_points,
                    desc=f"{subject} | {phase} | {desc}",
                    leave=False,
                )
            ):
                
                end_sample = int((time_end - tmin) * fs) + 1
                start_sample = end_sample - window_samples
                if start_sample < 0 or end_sample > X.shape[-1]:
                    continue
                
                X_segment = X.copy()[..., start_sample:end_sample]
                logger.info(f"Processing time window: {time_end:.3f}s, samples {start_sample}:{end_sample}")
                
                score, permutation_scores, p_value = predict_permutation_scores(
                    X_segment,
                    rt,
                    pipeline,
                    cv,
                    n_perm=n_perm,
                    random_state=42,
                    n_jobs=n_jobs,
                )
                
                # actual score shape: (time, channels)
                scores[t_idx, :] = score
                # permutation score shape: (time, channels, perm)
                perm_scores[t_idx, :, :] = permutation_scores
                # pval shape: (time, channels)
                pvals[t_idx, :] = p_value

            mask, p_values = cluster_correction(
                scores,
                perm_scores,
                p_thresh=0.05,
                tails=1,
            )

            # Save everything in (channels, time) orientation
            scores = rearrange(scores, 't c -> c t')
            pvals = rearrange(pvals, 't c -> c t')
            perm_scores = rearrange(perm_scores, 't c p -> c t p')

            pt = epoch_paths.copy().match()[0]
            task = pt.task
            # Match current packaging: results/{Task}(bipolar)(hammers)/RT/...
            save_path = BIDSPath(
                root=str(hga_results_dir(task, ref, atlas)),
                datatype='RT',
                subject=subject,
                suffix=band,
                processing=phase,
                description=desc,
                extension='.h5',
                check=False
            )
            save_path.mkdir(exist_ok=True)
            logger.info(f"Saving results to: {save_path}")

            with h5py.File(save_path, "w") as f:
                # New canonical score fields
                f.create_dataset(name='score', data=scores)
                f.create_dataset(name='perm_score', data=perm_scores)
                # Backward-compatible aliases for existing notebooks
                f.create_dataset(name='r2', data=scores)
                f.create_dataset(name='perm_r2', data=perm_scores)
                f.create_dataset(name='pval', data=pvals)
                f.create_dataset(name='mask', data=mask)
                f.create_dataset(name='cluster_pval', data=p_values)
                f.create_dataset(name='time', data=time_points)
                f.create_dataset(name='channels', data=np.asarray(channels, dtype='S'))

                f.attrs["fs"] = fs
                f.attrs["tmin"] = tmin
                f.attrs["tmax"] = tmax
                f.attrs["window"] = window
                f.attrs["step"] = step
                f.attrs["n_perm"] = n_perm
                f.attrs["n_folds"] = n_folds
                f.attrs["score_metric"] = 'pearson_r'
                f.attrs["band"] = band
                f.attrs["ref"] = ref
                f.attrs["atlas"] = atlas
                f.attrs["phase"] = phase
                f.attrs["description"] = desc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Channel-wise reaction-time prediction from high-gamma epochs."
    )
    parser.add_argument(
        "--bids_root",
        required=True,
        type=str,
        help="Path to BIDS root (must be set explicitly; no silent SentenceRep default)",
    )
    parser.add_argument('--band', 
                        default='highgamma', 
                        help='Band suffix in epoch files')
    parser.add_argument('--subject', 
                        required=True,
                        help='Subject to process')
    parser.add_argument('--ref', 
                        default='bipolar', 
                        help='Reference name used in derivatives')
    parser.add_argument(
        "--atlas",
        default="hammers",
        choices=list(SUPPORTED_ATLASES),
        help="Parcellation atlas suffix under derivatives/parcellation/",
    )
    parser.add_argument("--window", type=float, default=0.2,
                        help="window length in seconds")
    parser.add_argument("--step", type=float, default=0.02,
                        help="step size in seconds")
    parser.add_argument("--n_perm", type=int, default=500,
                        help="number of permutations")
    parser.add_argument("--n_folds", type=int, default=10,
                        help="number of folds")
    parser.add_argument("--n_jobs", type=int, default=20,
                        help="number of jobs")

    args = parser.parse_args()
    main(**vars(args))
