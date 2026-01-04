#!/usr/bin/env python3
"""Reaction time prediction from sEEG high gamma using Ridge regression with permutation testing."""

import rootutils
# add the root path to the python path for importing
path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import argparse
from pathlib import Path
from typing import Tuple, List, Optional

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from mne.decoding import Vectorizer
from joblib import Parallel, delayed
from tqdm import tqdm
import h5py
import sys
import logging
import os

# Simple logging: everything INFO and above to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def load_epoch_data(
    bids_root: str,
    subject: str,
    description: str,
    band: str,
    tmin: float,
    tmax: float,
) -> Tuple[np.ndarray, List[str]]:
    """Load and preprocess neural epoch data from MNE Epochs file.
    
    Parameters
    ----------
    bids_root : str or Path
        Root directory of BIDS dataset containing neural data
    subject : str
        Subject identifier (e.g., 'D0022')
    description : str
        Task description (e.g., 'production', 'perception')
    band : str
        Frequency band (e.g., 'highgamma', 'beta')
    tmin : float
        Start time of the temporal window in seconds
    tmax : float
        End time of the temporal window in seconds
        
    Returns
    -------
    X : ndarray, shape (n_epochs, n_channels, n_times)
        Neural time series data cropped to specified window
    words : list of str
        Word labels for each epoch (for alignment verification)
        
    Raises
    ------
    FileNotFoundError
        If no matching files found for the specified subject
    """
    # Construct path to epoch file
    epoch_path = os.path.join(
        bids_root,
        'derivatives',
        'epoch(bipolar)',
        f'sub-{subject}',
        'epoch(band)(sig)(effective)',
        f'sub-{subject}_task-PhonemeSequence_desc-{description}_{band}.h5'
    )
    
    if not os.path.exists(epoch_path):
        raise FileNotFoundError(f"No epoch file found at {epoch_path}")
    
    # Load MNE Epochs object
    epochs = mne.read_epochs(epoch_path, preload=True, verbose=False)
    
    # Get data and crop to specified time window
    epochs_cropped = epochs.copy().crop(tmin=tmin, tmax=tmax)
    X = epochs_cropped.get_data()
    
    # Extract word labels from event_id
    id_to_word = {v: k for k, v in epochs.event_id.items()}
    words = [id_to_word[e] for e in epochs.events[:, 2]]
    
    logger.info(f"Loaded epoch data: {X.shape} from {epoch_path}")
    logger.info(f"Time window: {tmin}s to {tmax}s, sfreq: {epochs.info['sfreq']} Hz")
    
    return X, words, epochs.info['sfreq']


def load_rt_data(
    bids_root: str,
    subject: str,
) -> Tuple[np.ndarray, List[str]]:
    """Load reaction time data from CSV file.
    
    Parameters
    ----------
    bids_root : str or Path
        Root directory of BIDS dataset
    subject : str
        Subject identifier (e.g., 'D0022')
        
    Returns
    -------
    rt : ndarray, shape (n_trials,)
        Reaction times in seconds
    words : list of str
        Word labels for each trial (for alignment verification)
        
    Raises
    ------
    FileNotFoundError
        If no matching RT file found
    """
    rt_path = os.path.join(
        bids_root,
        'derivatives',
        'features',
        f'sub-{subject}',
        'metadata',
        f'sub-{subject}_task-PhonemeSequence_desc-Repeat_RT.csv'
    )
    
    if not os.path.exists(rt_path):
        raise FileNotFoundError(f"No RT file found at {rt_path}")
    
    rt_df = pd.read_csv(rt_path)
    rt = rt_df['RT'].values
    words = rt_df['word'].tolist()
    
    logger.info(f"Loaded RT data: {len(rt)} trials from {rt_path}")
    logger.info(f"RT range: {rt.min():.3f}s to {rt.max():.3f}s, mean: {rt.mean():.3f}s")
    
    return rt, words


def verify_alignment(epoch_words: List[str], rt_words: List[str]) -> bool:
    """Verify that epoch and RT data are aligned by comparing word sequences.
    
    Parameters
    ----------
    epoch_words : list of str
        Word labels from epoch data
    rt_words : list of str
        Word labels from RT data
        
    Returns
    -------
    aligned : bool
        True if sequences match exactly
    """
    if len(epoch_words) != len(rt_words):
        logger.error(f"Length mismatch: epochs={len(epoch_words)}, RT={len(rt_words)}")
        return False
    
    if epoch_words != rt_words:
        # Find first mismatch
        for i, (e, r) in enumerate(zip(epoch_words, rt_words)):
            if e != r:
                logger.error(f"Word mismatch at trial {i}: epoch='{e}', RT='{r}'")
                break
        return False
    
    logger.info("Epoch and RT data alignment verified successfully")
    return True


def regression_permutation_scores(
    X: np.ndarray,
    y: np.ndarray,
    cv,
    regressor,
    n_jobs: int = -1,
    n_permutations: int = 100,
    scoring: str = "r2",
    random_state: int = 42,
) -> Tuple[List[float], np.ndarray, float]:
    """Compute regression scores with permutation testing.
    
    For each CV fold:
    1. Fit regressor on training data
    2. Evaluate on test data (observed score)
    3. Permute y labels and repeat n_permutations times (null distribution)
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, n_channels, n_times)
        Neural data
    y : ndarray, shape (n_samples,)
        Continuous target variable (reaction times)
    cv : CV splitter
        Cross-validation splitter
    regressor : sklearn estimator
        Regression pipeline
    n_jobs : int
        Number of parallel jobs
    n_permutations : int
        Number of permutations for null distribution
    scoring : str
        Scoring metric ('r2' or 'neg_mean_squared_error')
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    obs_scores : list of float
        Observed scores per fold
    perm_scores : ndarray, shape (n_folds, n_permutations)
        Permutation scores per fold
    p_value : float
        One-sided p-value (proportion of permutation scores >= observed)
    """
    from sklearn.metrics import get_scorer
    
    scorer = get_scorer(scoring)
    splits = list(cv.split(X, y))
    if len(splits) == 0:
        raise ValueError("CV splitter produced no splits")
    
    obs_scores = []
    perm_scores = []
    
    for fold_idx, (tr, te) in enumerate(tqdm(splits, desc="Cross-validation")):
        reg = clone(regressor)
        
        X_train, X_test = X[tr], X[te]
        y_train, y_test = y[tr], y[te]
        
        # Handle NaN values in neural data
        is_nan_train = np.isnan(X_train)
        if is_nan_train.any():
            X_train = X_train.copy()
            X_train[is_nan_train] = 0
            
        is_nan_test = np.isnan(X_test)
        if is_nan_test.any():
            X_test = X_test.copy()
            X_test[is_nan_test] = 0
        
        # Fit and evaluate
        reg.fit(X_train, y_train)
        observed_score = scorer(reg, X_test, y_test)
        obs_scores.append(observed_score)
        
        # Permutation testing
        rng_fold = np.random.RandomState(random_state + fold_idx)
        seeds_fold = rng_fold.randint(0, 2**31 - 1, size=n_permutations)
        
        def one_perm(seed):
            r = np.random.RandomState(seed)
            y_train_perm = y_train.copy()
            r.shuffle(y_train_perm)
            reg_p = clone(regressor)
            reg_p.fit(X_train, y_train_perm)
            return scorer(reg_p, X_test, y_test)
        
        perm_score = np.asarray(
            Parallel(n_jobs=n_jobs)(
                delayed(one_perm)(s) for s in tqdm(seeds_fold, desc=f"Permutations (fold {fold_idx+1})", leave=False)
            )
        )
        perm_scores.append(perm_score)
    
    score = np.mean(obs_scores)
    perm_scores = np.stack(perm_scores)
    
    # p-value (greater is better for R2)
    p_value = (np.sum(perm_scores.mean(axis=0) >= score) + 1.0) / (n_permutations + 1.0)
    
    return obs_scores, perm_scores, p_value


def main(
    bids_root: str,
    subject: str,
    description: str,
    band: str,
    variance: float,
    alpha: float,
    n_perm: int,
    n_folds: int,
    n_jobs: int,
    tmin: float,
    tmax: float,
):
    """Main function for RT prediction from neural data.
    
    Parameters
    ----------
    bids_root : str
        Root directory of BIDS dataset
    subject : str
        Subject identifier
    description : str
        Task description ('production' or 'perception')
    band : str
        Frequency band
    variance : float
        PCA variance to retain (0-1)
    alpha : float
        Ridge regression regularization parameter
    n_perm : int
        Number of permutations
    n_folds : int
        Number of CV folds
    n_jobs : int
        Number of parallel jobs
    tmin : float
        Start time of analysis window
    tmax : float
        End time of analysis window
    """
    # Load data
    X, epoch_words, sfreq = load_epoch_data(
        bids_root, subject, description, band, tmin, tmax
    )
    rt, rt_words = load_rt_data(bids_root, subject)
    
    # Verify alignment
    if not verify_alignment(epoch_words, rt_words):
        raise ValueError("Epoch and RT data are not aligned!")
    
    y = rt
    
    # Build regression pipeline
    # Vectorizer: (n_epochs, n_channels, n_times) -> (n_epochs, n_channels * n_times)
    # StandardScaler: normalize features
    # PCA: dimensionality reduction
    # Ridge: L2-regularized linear regression
    logger.info(f'Building pipeline with PCA variance={variance}, Ridge alpha={alpha}')
    
    regressor = make_pipeline(
        Vectorizer(),
        StandardScaler(),
        PCA(n_components=variance, random_state=42),
        Ridge(alpha=alpha, random_state=42)
    )
    
    # Cross-validation
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Run permutation testing
    obs_scores, perm_scores, p_value = regression_permutation_scores(
        X, y, cv, regressor,
        n_jobs=n_jobs,
        n_permutations=n_perm,
        scoring='r2',
        random_state=42,
    )
    
    # Log results
    mean_r2 = np.mean(obs_scores)
    std_r2 = np.std(obs_scores)
    logger.info(f"Observed R² = {mean_r2:.4f} ± {std_r2:.4f}")
    logger.info(f"Permutation null mean R² = {perm_scores.mean():.4f}")
    logger.info(f"P-value = {p_value:.4f}")
    
    # Save results
    save_dir = os.path.join('results', f'rt_prediction(bipolar)')
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(
        save_dir,
        f'sub-{subject}_task-PhonemeSequence_desc-{description}_{band}_rt-prediction.h5'
    )
    
    logger.info(f'Saving results to {save_path}')
    with h5py.File(save_path, "w") as f:
        f.create_dataset(name="r2_scores", data=obs_scores)
        f.create_dataset(name="perm_scores", data=perm_scores)
        f.create_dataset(name="p_value", data=p_value)
        
        f.attrs["sfreq"] = sfreq
        f.attrs["tmin"] = tmin
        f.attrs["tmax"] = tmax
        f.attrs["variance"] = variance
        f.attrs["alpha"] = alpha
        f.attrs["n_perm"] = n_perm
        f.attrs["n_folds"] = n_folds
        f.attrs["mean_r2"] = mean_r2
        f.attrs["std_r2"] = std_r2

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Predict reaction time from sEEG high gamma using Ridge regression"
    )

    parser.add_argument("--bids_root", type=str,
                        default="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/",
                        help="Root directory of the BIDS dataset")
    parser.add_argument("--subject", type=str, default='D0022',
                        help="Subject to process (e.g., D0022)")
    parser.add_argument("--description", type=str, default='production',
                        choices=['perception', 'production'],
                        help="Task phase: perception or production")
    parser.add_argument("--band", type=str, default='highgamma',
                        help="Frequency band of neural signal")
    parser.add_argument("--variance", type=float, default=0.85,
                        help="PCA variance to retain (0-1)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ridge regression regularization parameter")
    parser.add_argument("--n_perm", type=int, default=100,
                        help="Number of permutations for significance testing")
    parser.add_argument("--n_folds", type=int, default=10,
                        help="Number of cross-validation folds")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel jobs")
    parser.add_argument("--tmin", type=float, default=-0.5,
                        help="Start time of analysis window (seconds)")
    parser.add_argument("--tmax", type=float, default=0.0,
                        help="End time of analysis window (seconds)")

    args = parser.parse_args()
    main(**vars(args))
