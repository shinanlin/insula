#!/usr/bin/env python3
"""Single-electrode RT encoding analysis following Muller et al. (2025) Nature Human Behaviour.

This script implements the reaction time encoding approach where:
1. For each electrode independently, fit a linear regression: neural activity -> RT
2. Evaluate prediction using Spearman rank correlation between predicted and actual RT
3. Use permutation testing (shuffle trials) to establish significance

Reference: Speech sequencing in the human precentral gyrus (Nature Human Behaviour, 2025)
"""

import rootutils
# add the root path to the python path for importing
path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import argparse
from pathlib import Path
from typing import Tuple, List, Optional

import mne
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.base import clone
from joblib import Parallel, delayed
from tqdm import tqdm
import h5py
import sys
import logging
import os
from mne_bids import BIDSPath

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
) -> Tuple[np.ndarray, List[str], float, List[str]]:
    """Load and preprocess neural epoch data from MNE Epochs file.
    
    Parameters
    ----------
    bids_root : str or Path
        Root directory of BIDS dataset containing neural data
    subject : str
        Subject identifier (e.g., 'D0022')
    description : str
        Task description (e.g., 'production', 'repeat')
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
    sfreq : float
        Sampling frequency
    ch_names : list of str
        Channel names
        
    Raises
    ------
    FileNotFoundError
        If no matching files found for the specified subject
    """
    epoch_path = BIDSPath(
        root=os.path.join(bids_root, 'derivatives', 'epoch(bipolar)'),
        subject=subject,
        datatype='epoch(band)(sig)(effective)',
        description=description,
        extension='.h5',
        check=False
    )
    
    epoch_path = epoch_path.match()[0]
    # Load MNE Epochs object
    epochs = mne.read_epochs(epoch_path, preload=True, verbose=False)
    
    # Get data and crop to specified time window
    epochs_cropped = epochs.copy().crop(tmin=tmin, tmax=tmax)
    X = epochs_cropped.get_data()
    
    # Extract word labels from event_id
    id_to_word = {v: k for k, v in epochs.event_id.items()}
    words = [id_to_word[e] for e in epochs.events[:, 2]]
    
    ch_names = epochs.ch_names
    
    logger.info(f"Loaded epoch data: {X.shape} from {epoch_path}")
    logger.info(f"Time window: {tmin}s to {tmax}s, sfreq: {epochs.info['sfreq']} Hz")
    logger.info(f"Channels: {len(ch_names)}")
    
    return X, words, epochs.info['sfreq'], ch_names


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

    rt_path = BIDSPath(
        root=os.path.join(bids_root,'derivatives','features'),
        subject=subject,
        datatype='metadata',
        extension='.csv',
        check=False
    )
    rt_path = rt_path.match()[0]
    
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


def single_electrode_rt_encoding(
    X_elec: np.ndarray,
    y: np.ndarray,
    cv,
    alpha: float = 1.0,
    n_permutations: int = 100,
    random_state: int = 42,
) -> Tuple[float, float, np.ndarray]:
    """Single-electrode RT encoding with cross-validation and permutation testing.
    
    For one electrode:
    1. Flatten time dimension to get features: (n_trials, n_times)
    2. Fit Ridge regression with cross-validation to predict RT
    3. Compute Spearman correlation between predicted and actual RT
    4. Permutation test: shuffle RT labels to build null distribution
    
    Parameters
    ----------
    X_elec : ndarray, shape (n_trials, n_times)
        Neural activity for one electrode across trials
    y : ndarray, shape (n_trials,)
        Reaction times
    cv : CV splitter
        Cross-validation splitter
    alpha : float
        Ridge regularization parameter
    n_permutations : int
        Number of permutations for null distribution
    random_state : int
        Random seed
        
    Returns
    -------
    rho : float
        Observed Spearman correlation between predicted and actual RT
    p_value : float
        Permutation-based p-value (two-sided)
    perm_rhos : ndarray, shape (n_permutations,)
        Null distribution of Spearman correlations
    """
    # Handle NaN values
    X_elec = X_elec.copy()
    is_nan = np.isnan(X_elec)
    if is_nan.any():
        X_elec[is_nan] = 0
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_elec)
    
    # Cross-validated prediction
    regressor = Ridge(alpha=alpha, random_state=random_state)
    y_pred = cross_val_predict(regressor, X_scaled, y, cv=cv)
    
    # Observed Spearman correlation
    rho, _ = spearmanr(y_pred, y)
    
    # Permutation testing
    rng = np.random.RandomState(random_state)
    perm_rhos = np.zeros(n_permutations)
    
    for i in range(n_permutations):
        y_perm = y.copy()
        rng.shuffle(y_perm)
        y_pred_perm = cross_val_predict(regressor, X_scaled, y_perm, cv=cv)
        perm_rhos[i], _ = spearmanr(y_pred_perm, y_perm)
    
    # Two-sided p-value
    p_value = (np.sum(np.abs(perm_rhos) >= np.abs(rho)) + 1) / (n_permutations + 1)
    
    return rho, p_value, perm_rhos


def run_encoding_all_electrodes(
    X: np.ndarray,
    y: np.ndarray,
    cv,
    alpha: float = 1.0,
    n_permutations: int = 100,
    n_jobs: int = 1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run RT encoding analysis for all electrodes.
    
    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_times)
        Neural data
    y : ndarray, shape (n_trials,)
        Reaction times
    cv : CV splitter
        Cross-validation splitter
    alpha : float
        Ridge regularization parameter
    n_permutations : int
        Number of permutations
    n_jobs : int
        Number of parallel jobs
    random_state : int
        Random seed
        
    Returns
    -------
    rhos : ndarray, shape (n_channels,)
        Observed Spearman correlations per electrode
    p_values : ndarray, shape (n_channels,)
        P-values per electrode
    perm_rhos : ndarray, shape (n_channels, n_permutations)
        Null distributions per electrode
    """
    n_trials, n_channels, n_times = X.shape
    
    def process_electrode(ch_idx):
        X_elec = X[:, ch_idx, :]  # (n_trials, n_times)
        rho, p_val, perm = single_electrode_rt_encoding(
            X_elec, y, cv, alpha, n_permutations, random_state + ch_idx
        )
        return ch_idx, rho, p_val, perm
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_electrode)(ch_idx) 
        for ch_idx in tqdm(range(n_channels), desc="Electrodes")
    )
    
    # Organize results
    rhos = np.zeros(n_channels)
    p_values = np.zeros(n_channels)
    perm_rhos = np.zeros((n_channels, n_permutations))
    
    for ch_idx, rho, p_val, perm in results:
        rhos[ch_idx] = rho
        p_values[ch_idx] = p_val
        perm_rhos[ch_idx] = perm
    
    return rhos, p_values, perm_rhos


def main(
    bids_root: str,
    subject: str,
    description: str,
    band: str,
    alpha: float,
    n_perm: int,
    n_folds: int,
    n_jobs: int,
    tmin: float,
    tmax: float,
):
    """Main function for single-electrode RT encoding analysis.
    
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
    X, epoch_words, sfreq, ch_names = load_epoch_data(
        bids_root, subject, description, band, tmin, tmax
    )
    rt, rt_words = load_rt_data(bids_root, subject)
    
    # Verify alignment
    if not verify_alignment(epoch_words, rt_words):
        raise ValueError("Epoch and RT data are not aligned!")
    
    y = rt
    
    # Cross-validation
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    logger.info(f'Running single-electrode RT encoding with Ridge alpha={alpha}')
    logger.info(f'CV folds={n_folds}, permutations={n_perm}')
    
    # Run encoding for all electrodes
    rhos, p_values, perm_rhos = run_encoding_all_electrodes(
        X, y, cv,
        alpha=alpha,
        n_permutations=n_perm,
        n_jobs=n_jobs,
        random_state=42,
    )
    
    # Log results
    n_sig = np.sum(p_values < 0.05)
    logger.info(f"Significant electrodes (p<0.05): {n_sig}/{len(ch_names)}")
    for i, (ch, rho, p) in enumerate(zip(ch_names, rhos, p_values)):
        if p < 0.05:
            logger.info(f"  {ch}: rho={rho:.4f}, p={p:.4f}")
    
    # Save results
    save_dir = os.path.join('results', f'rt_encoding(bipolar)')
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(
        save_dir,
        f'sub-{subject}_task-PhonemeSequence_desc-{description}_{band}_rt-encoding.h5'
    )
    
    logger.info(f'Saving results to {save_path}')
    with h5py.File(save_path, "w") as f:
        f.create_dataset(name="rhos", data=rhos)
        f.create_dataset(name="p_values", data=p_values)
        f.create_dataset(name="perm_rhos", data=perm_rhos)
        
        # Store channel names as variable-length strings
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset(name="ch_names", data=ch_names, dtype=dt)
        
        f.attrs["sfreq"] = sfreq
        f.attrs["tmin"] = tmin
        f.attrs["tmax"] = tmax
        f.attrs["alpha"] = alpha
        f.attrs["n_perm"] = n_perm
        f.attrs["n_folds"] = n_folds
        f.attrs["n_sig"] = n_sig

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    )

    parser.add_argument("--bids_root", type=str,
                        default="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/",
                        help="Root directory of the BIDS dataset")
    parser.add_argument("--subject", type=str, default='D0088',
                        help="Subject to process (e.g., D0088)")
    parser.add_argument("--description", type=str, default='production',
                        choices=['repeat', 'production'],
                        help="Task phase: repeat or production")
    parser.add_argument("--band", type=str, default='highgamma',
                        help="Frequency band of neural signal")
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
