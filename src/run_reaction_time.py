
import argparse
import logging
import os
from typing import Optional

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath
from himalaya.ridge import RidgeCV
from ieeg.calc.stats import time_perm_cluster
from joblib import Parallel, delayed
from tqdm import tqdm
import h5py


def load_subject_data(subject: str, epoch_paths: BIDSPath, ref: str) -> pd.DataFrame:
    """
    Load and preprocess epoch data for a single subject.
    
    Returns a long-format DataFrame with columns:
        time, channel, value, epoch, onset, phase, description, subject, 
        condition, remark, trial, rt
    
    Returns None if data cannot be loaded.
    """
    subj_pts = epoch_paths.copy().update(subject=subject).match()
    
    # Load parcellation
    try:
        parc_path = subj_pts[0].copy().update(
            root=str(subj_pts[0].root).replace(f'epoch({ref})', 'parcellation'),
            datatype=ref,
            task=None,
            description=None,
            recording=None,
            processing='3mm',
            suffix='aparc2009s',
            extension='.csv',
        ).match()[0]
        parc = pd.read_csv(parc_path)
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
        epochs.pick([ch for ch in picks if ch in epochs.ch_names])
        
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
    df = df[df.remark=='CORRECT']
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
    go_resp_onset = (
        epoch_tbl[epoch_tbl['phase'].isin(['Go', 'Response'])]
        .pivot_table(
            index=['subject', 'description', 'trial', 'condition'],
            columns='phase',
            values='onset',
            aggfunc='first'
        )
        .reset_index()
        .rename(columns={'Go': 'go_onset', 'Response': 'resp_onset'})
    )

    go_resp_onset['rt'] = go_resp_onset['resp_onset'] - go_resp_onset['go_onset']

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
    min_trials: int = 10,
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
    
    times = np.sort(sub['time'].unique())
    trials = sub['trial'].dropna().unique()
    channels = sub['channel'].unique()
    
    trial_to_rt = sub.groupby('trial')['rt'].first().to_dict()
    rt = np.array([trial_to_rt[t] for t in trials])
    
    valid = ~np.isnan(rt)

    
    trials = trials[valid]
    rt = rt[valid]
    
    # Build 3D array via pivot
    valid_sub = sub[sub['trial'].isin(trials)]
    pivot = valid_sub.pivot_table(
        index='trial', columns=['channel', 'time'], values='value', aggfunc='mean'
    )
    # Reshape to (n_trials, n_channels, n_times)
    X_3d = pivot.values.reshape(len(trials), len(channels), len(times))
    
    return X_3d, rt, times, channels


def _window_mean(arr: np.ndarray, center_idx: int, half_win: int) -> np.ndarray:
    """Compute mean over a sliding window centered at center_idx."""
    start = max(0, center_idx - half_win)
    end = min(arr.shape[-1], center_idx + half_win + 1)
    return np.nanmean(arr[..., start:end], axis=-1)


def _fit_ridge_at_time(
    X_3d: np.ndarray,
    rt: np.ndarray,
    t_idx: int,
    half_win: int,
    alphas: np.ndarray,
    cv: int = 5,
    min_samples: int = 10,
):
    """Fit RidgeCV at a single time point and return CV R² and best alpha.
    
    Uses cross_val_score to avoid data leakage (train/test on same data).
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import RidgeCV as SklearnRidgeCV
    
    X_win = _window_mean(X_3d, t_idx, half_win)
    valid_mask = ~np.isnan(X_win).any(axis=1)
    
    if valid_mask.sum() < min_samples:
        return np.nan, np.nan
    
    X_t = X_win[valid_mask]
    y_t = rt[valid_mask]
    
    # Use sklearn RidgeCV to find best alpha
    ridge_cv = SklearnRidgeCV(alphas=alphas, cv=cv, scoring='r2')
    ridge_cv.fit(X_t, y_t)
    best_alpha = ridge_cv.alpha_
    
    # Get unbiased CV score using the best alpha
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=best_alpha, fit_intercept=True)
    cv_scores = cross_val_score(ridge, X_t, y_t, cv=cv, scoring='r2')
    r2 = np.mean(cv_scores)
    
    return r2, best_alpha


def _permutation_r2_all_times(
    X_3d: np.ndarray,
    rt: np.ndarray,
    half_win: int,
    alpha: float,
    seed: int,
    min_samples: int = 10,
):
    """Compute R² for a single permutation across all time points.
    
    Returns array of shape (n_times,).
    """
    from sklearn.linear_model import Ridge
    
    n_times = X_3d.shape[-1]
    r2_perm = np.full(n_times, np.nan)
    
    rng = np.random.default_rng(seed)
    rt_shuf = rng.permutation(rt)
    
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    
    for t_idx in range(n_times):
        X_win = _window_mean(X_3d, t_idx, half_win)
        valid_mask = ~np.isnan(X_win).any(axis=1)
        
        if valid_mask.sum() < min_samples:
            continue
        
        X_t = X_win[valid_mask]
        y_t = rt_shuf[valid_mask]
        
        ridge.fit(X_t, y_t)
        r2_perm[t_idx] = ridge.score(X_t, y_t)
    
    return r2_perm


def sliding_window_rt_prediction(
    df: pd.DataFrame,
    phase: str,
    description: str,
    win_size: float = 0.05,
    n_perm: int = 1000,
    alphas: Optional[np.ndarray] = None,
    n_jobs: int = -1,
    random_state: int = 42,
) -> Optional[pd.DataFrame]:
    """Predict RT using sliding-window HGA with RidgeCV + permutation cluster test.
    
    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame from load_subject_data.
    phase : str
        Phase to analyze (e.g., 'Stimulus', 'Delay', 'Go', 'Response').
    description : str
        Condition to analyze (e.g., 'Decision', 'Repeat').
    win_size : float
        Sliding window size in seconds (default 50ms).
    n_perm : int
        Number of permutations for null distribution.
    alphas : ndarray, optional
        Array of alpha values for RidgeCV. Default: logspace(-3, 3, 10).
    n_jobs : int
        Number of parallel jobs (-1 for all cores).
    random_state : int
        Random seed for reproducibility.
    
    Returns
    -------
    pd.DataFrame with columns: time, r2, best_alpha, r2_null_mean, r2_null_std, mask, pval
    Returns None if insufficient data.
    """
    if alphas is None:
        alphas = np.logspace(-3, 3, 10)
    
    X_3d, rt, times, channels = _prepare_data(df, phase, description)
    
    dt = np.median(np.diff(times))
    half_win = int(np.ceil(win_size / 2 / dt))
    n_times = len(times)
    
    # Step 1: Fit RidgeCV at each time point (parallel)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_ridge_at_time)(X_3d, rt, t_idx, half_win, alphas)
        for t_idx in tqdm(range(n_times), desc="Fitting RidgeCV")
    )
    r2_true = np.array([r[0] for r in results])
    best_alphas = np.array([r[1] for r in results])
    
    # Use median best alpha for permutations (more stable)
    median_alpha = np.nanmedian(best_alphas)
    if np.isnan(median_alpha):
        median_alpha = 1.0
    
    # Step 2: Permutation test (parallel over permutations, serial over time points)
    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, 2**31, size=n_perm)
    
    perm_results = Parallel(n_jobs=n_jobs)(
        delayed(_permutation_r2_all_times)(
            X_3d, rt, half_win, median_alpha, seeds[p]
        )
        for p in tqdm(range(n_perm), desc="Permutations")
    )
    r2_null = np.array(perm_results)  # (n_perm, n_times)
    
    # Step 3: Cluster-based permutation test
    r2_true_2d = r2_true.reshape(1, -1)
    mask, pvals = time_perm_cluster(
        r2_true_2d,
        r2_null,
        p_thresh=0.05,
        n_perm=n_perm,
        n_jobs=1,
    )
    
    result = pd.DataFrame({
        'time': times,
        'r2': r2_true,
        'best_alpha': best_alphas,
        'r2_null_mean': np.nanmean(r2_null, axis=0),
        'r2_null_std': np.nanstd(r2_null, axis=0),
        'mask': mask.flatten(),
        'pval': pvals.flatten(),
    })
    
    return result


def main(bids_root: str, 
         ref: str, 
         band: str):
    """Main entry point: load data for all subjects and predict RT."""
    
    epoch_paths = BIDSPath(
        root=bids_root + f"derivatives/epoch({ref})",
        suffix=band,
        datatype='epoch(band)(zscore)',
        extension=".h5",
        check=False,
    )
    
    subjects = list(set([pt.subject for pt in epoch_paths.match()]))
    
    for subject in tqdm(subjects, desc='Processing subjects'):
        df = load_subject_data(subject, epoch_paths, ref)
        if df is None:
            continue
        
        for phase in df.phase.unique():
            for desc in df.description.unique():
                result = sliding_window_rt_prediction(df, phase, desc)
                if result is None:
                    logging.info(f"Skipping {subject}/{phase}/{desc}: insufficient data")
                    continue
                
                result['subject'] = subject
                result['phase'] = phase
                result['description'] = desc
                
                # Save result
                out_dir = os.path.join(bids_root, 'derivatives', 'rt_prediction', ref)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f'sub-{subject}_phase-{phase}_desc-{desc}_rt-pred.csv')
                result.to_csv(out_path, index=False)
                logging.info(f"Saved: {out_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--bids_root', 
                        default='/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/', 
                        help='Path to BIDS root')
    parser.add_argument('--band', 
                        default='highgamma', 
                        help='Band suffix in epoch files')
    parser.add_argument('--ref', 
                        default='bipolar', 
                        help='Reference name used in derivatives')

    args = parser.parse_args()
    main(**vars(args))