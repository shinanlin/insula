#!/usr/bin/env python3
"""Connectivity analysis script for sEEG data using MVAR-based PDC.

This script computes functional connectivity (Partial Directed Coherence) 
between electrodes for each subject and experimental condition.
Uses the SCoT (Source Connectivity Toolbox) package for MVAR modeling.
"""

import rootutils
path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import argparse
from pathlib import Path
import h5py
import numpy as np
from mne_bids import BIDSPath
import logging
import sys
import mne
import os
import scipy as sp
import json
from joblib import Parallel, delayed
from ieeg.calc.stats import time_perm_cluster


# Import SCoT for MVAR-based connectivity
from scot.var import VAR
from scot.connectivity import Connectivity

# SciPy compatibility patch for SCOT (which uses deprecated top-level scipy aliases)
if not hasattr(sp, 'shape'):
    sp.shape = np.shape
if not hasattr(sp, 'ceil'):
    sp.ceil = np.ceil
if not hasattr(sp, 'atleast_3d'):
    sp.atleast_3d = np.atleast_3d
if not hasattr(sp, 'sign'):
    sp.sign = np.sign
if not hasattr(sp, 'sqrt'):
    sp.sqrt = np.sqrt
if not hasattr(sp, 'zeros'):
    sp.zeros = np.zeros
if not hasattr(sp, 'cov'):
    sp.cov = np.cov
if not hasattr(sp, 'sum'):
    sp.sum = np.sum
if not hasattr(sp, 'eye'):
    sp.eye = np.eye
if not hasattr(sp, 'random'):
    sp.random = np.random

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
    band: str,
    datatype: str = 'epoch(band)(sig)(effective)',
) -> mne.Epochs:
    """Load preprocessed epoch data from HDF5 file saved in MNE format.
    
    Parameters
    ----------
    bids_root : str
        Root directory of BIDS dataset (derivatives path)
    subject : str
        Subject identifier (e.g., 'D0019')
    band : str
        Frequency band (e.g., 'highgamma')
    datatype : str
        Data type folder name
        
    Returns
    -------
    epochs : mne.Epochs
        Loaded epoch data
    """
    epoch_path = BIDSPath(
        root=os.path.join(bids_root, 'derivatives', 'epoch(bipolar)'),
        subject=subject,
        suffix=band,
        datatype=datatype,
        extension='.h5',
        check=False
    )
    
    matched_files = epoch_path.match()
    if not matched_files:
        raise FileNotFoundError(f"No files found for {epoch_path}")
    
    epochs = []
    
    for epoch_file in matched_files:
        this_epoch = mne.read_epochs(epoch_file, verbose='error')
        epochs.append(this_epoch)
        logger.info(f'Loading epoch data from: {epoch_file}')
    
    return epochs, matched_files


def _permute_pdc(seed: int, segment: np.ndarray, p: int, nfft: int = 64) -> np.ndarray:
    """Generate a circular-shift surrogate, fit VAR, and return mean PDC over freq.

    Parameters
    ----------
    seed : int
        RNG seed for reproducibility across workers.
    segment : np.ndarray
        Data segment with shape (n_epochs, n_channels, n_samples).
    p : int
        VAR model order.
    nfft : int
        Number of FFT points for connectivity computation.

    Returns
    -------
    np.ndarray
        Mean PDC over frequency with shape (n_channels, n_channels).
    """
    # Re-apply SciPy compatibility patch for parallel workers
    import scipy as sp
    if not hasattr(sp, 'shape'):
        sp.shape = np.shape
    if not hasattr(sp, 'ceil'):
        sp.ceil = np.ceil
    if not hasattr(sp, 'atleast_3d'):
        sp.atleast_3d = np.atleast_3d
    if not hasattr(sp, 'sign'):
        sp.sign = np.sign
    if not hasattr(sp, 'sqrt'):
        sp.sqrt = np.sqrt
    if not hasattr(sp, 'zeros'):
        sp.zeros = np.zeros
    if not hasattr(sp, 'cov'):
        sp.cov = np.cov
    if not hasattr(sp, 'sum'):
        sp.sum = np.sum
    if not hasattr(sp, 'eye'):
        sp.eye = np.eye
    if not hasattr(sp, 'random'):
        sp.random = np.random
    
    rng = np.random.default_rng(seed)
    n_tr, n_ch, L = segment.shape
    sur = np.empty_like(segment)
    for tr in range(n_tr):
        shifts = rng.integers(1, L, size=n_ch)
        for ch in range(n_ch):
            sur[tr, ch, :] = np.roll(segment[tr, ch, :], int(shifts[ch]))

    m_sur = VAR(model_order=p)
    m_sur.fit(sur)
    conn_sur = Connectivity(m_sur.coef, c=m_sur.rescov, nfft=nfft)
    pdc_sur = conn_sur.PDC()
    return pdc_sur.mean(axis=-1)


def compute_connectivity(
    epochs: mne.Epochs,
    window: float,
    step: float,
    model_order: int = 4,
    nfft: int = 64,
    n_permutations: int = 100,
    n_jobs: int = 1,
) -> dict:
    """Compute time-resolved PDC connectivity with permutation statistics.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Input epoch data (n_epochs, n_channels, n_times)
    window : float
        Window length in seconds
    step : float
        Step size in seconds
    model_order : int
        VAR model order (default: 4)
    nfft : int
        Number of FFT points for connectivity
    n_permutations : int
        Number of permutations for statistical testing
    n_jobs : int
        Number of parallel jobs
        
    Returns
    -------
    results : dict
        Dictionary containing connectivity results and metadata
    """
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    tmin, tmax = epochs.tmin, epochs.tmax
    fs = epochs.info['sfreq']
    
    # Compute time windows
    time_points = np.arange(tmin + window, tmax + step, step)
    window_samples = int(window * fs)
    
    # Compute window indices
    end_samples = (np.round((time_points - tmin) * fs).astype(int)) + 1
    start_samples = end_samples - window_samples
    
    # Filter valid windows
    n_samples_total = data.shape[-1]
    valid = (start_samples >= 0) & (end_samples <= n_samples_total)
    start_samples = start_samples[valid]
    end_samples = end_samples[valid]
    time_points = time_points[valid]
    n_windows = len(start_samples)
    
    _, n_channels, _ = data.shape
    p = model_order
    
    # Pre-allocate result arrays
    coefs = np.empty((n_windows, n_channels, n_channels * p), dtype=float)
    rescovs = np.empty((n_windows, n_channels, n_channels), dtype=float)
    pdcs = np.empty((n_windows, n_channels, n_channels), dtype=float)
    null = np.empty((n_permutations, n_windows, n_channels, n_channels), dtype=float)
    pvals = np.ones((n_windows, n_channels, n_channels), dtype=float)
    
    logger.info(f'Computing connectivity for {n_windows} time windows')
    
    for wi, (ss, es) in enumerate(zip(start_samples, end_samples)):
        segment = data[..., ss:es].copy()
        
        # Fit VAR model
        model = VAR(model_order=p)
        model.fit(segment)
        coefs[wi] = model.coef
        rescovs[wi] = model.rescov
        
        # Compute PDC
        conn = Connectivity(model.coef, c=model.rescov, nfft=nfft)
        pdc = conn.PDC()  # (n_channels, n_channels, nfft)
        pdc_mean_obs = pdc.mean(axis=-1)
        pdcs[wi] = pdc_mean_obs
        
        # Permutation testing
        seed_rng = np.random.default_rng(42 + wi)
        seeds = seed_rng.integers(0, 2**32 - 1, size=n_permutations, dtype=np.uint64)
        
        null[:, wi] = np.asarray(Parallel(n_jobs=n_jobs, prefer='processes')(
            delayed(_permute_pdc)(int(seeds[r]), segment, p, nfft) 
            for r in range(n_permutations)
        ))
        
        # One-sided p-values
        pvals[wi] = (1 + (null[:, wi] >= pdcs[wi]).sum(axis=0)) / (1 + n_permutations)
    
    results = {
        'coef': coefs,
        'rescov': rescovs,
        'pdc': pdcs,
        'pvals': pvals,
        'null': null,
        'start_samples': start_samples,
        'end_samples': end_samples,
        'time_points': time_points,
        'fs': fs,
        'tmin': tmin,
        'tmax': tmax,
        'window': window,
        'step': step,
        'model_order': p,
        'nfft': nfft,
        'ch_names': epochs.ch_names,
    }
    
    return results


def save_connectivity_results(
    results: dict,
    save_path: BIDSPath,
    subject: str,
    band: str,
):
    """Save connectivity results to HDF5 file.
    
    Parameters
    ----------
    results : dict
        Connectivity results from compute_connectivity
    save_path : BIDSPath
        Output path
    subject : str
        Subject identifier
    band : str
        Frequency band
    """
    save_path.mkdir(exist_ok=True)
    
    with h5py.File(save_path.fpath, 'w') as f:
        # Save arrays
        f.create_dataset('coef', data=results['coef'])
        f.create_dataset('rescov', data=results['rescov'])
        f.create_dataset('pdc', data=results['pdc'])
        f.create_dataset('pvals', data=results['pvals'])
        f.create_dataset('null', data=results['null'])
        f.create_dataset('start_samples', data=results['start_samples'])
        f.create_dataset('end_samples', data=results['end_samples'])
        f.create_dataset('time_points', data=results['time_points'])
        
        # Save metadata
        f.attrs['subject'] = subject
        f.attrs['description'] = str(save_path.description)
        f.attrs['phase'] = str(save_path.processing)
        f.attrs['band'] = band
        f.attrs['fs'] = float(results['fs'])
        f.attrs['tmin'] = float(results['tmin'])
        f.attrs['tmax'] = float(results['tmax'])
        f.attrs['window'] = float(results['window'])
        f.attrs['step'] = float(results['step'])
        f.attrs['model_order'] = int(results['model_order'])
        f.attrs['nfft'] = int(results['nfft'])
        f.attrs['channel_names_json'] = json.dumps(results['ch_names'])
    
    logger.info(f'Saved connectivity results to: {save_path}')


def main(
    bids_root: str,
    subject: str,
    band: str,
    datatype: str,
    window: float,
    step: float,
    model_order: int,
    n_permutations: int,
    n_jobs: int,
):
    """Main function for connectivity analysis.
    
    Parameters
    ----------
    bids_root : str
        Root directory of BIDS dataset
    subject : str
        Subject identifier
    band : str
        Frequency band
    datatype : str
        Data type folder
    window : float
        Window length in seconds
    step : float
        Step size in seconds
    model_order : int
        VAR model order
    n_permutations : int
        Number of permutations
    n_jobs : int
        Number of parallel jobs
    """
    # Load epoch data
    epochs, paths = load_epoch_data(
        bids_root=bids_root,
        subject=subject,
        band=band,
        datatype=datatype,
    )
    
    for epoch, path in zip(epochs, paths):
        logger.info(f'Loading epoch data from: {path}')
        
        logger.info(f'Loaded epochs: {epoch.get_data().shape}')
        logger.info(f'Channels: {epoch.ch_names}')
        logger.info(f'Time range: {epoch.tmin} to {epoch.tmax} s')
        
        # Compute connectivity
        results = compute_connectivity(
            epochs=epoch,
            window=window,
            step=step,
            model_order=model_order,
            n_permutations=n_permutations,
            n_jobs=n_jobs,
        )

        save_path = BIDSPath(
            root=f'./results/{path.task}(bipolar)',
            subject=subject,
            suffix='pdc',
            datatype='connectivity',
            description=path.description,
            processing=path.processing,
            task=path.task,
            extension='.h5',
            check=False
        )
        
        save_connectivity_results(
            results=results,
            save_path=save_path,
            subject=subject,
            band=band,
        )
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute MVAR-based PDC connectivity for sEEG data"
    )
    
    parser.add_argument("--bids_root", type=str,
                        default="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/",
                        help="Root directory of the BIDS dataset")
    parser.add_argument("--subject", type=str, default='D0023',
                        help="Subject to process")
    parser.add_argument("--band", type=str, default='highgamma',
                        help="Frequency band")
    parser.add_argument("--datatype", type=str, default='epoch(band)(sig)(effective)',
                        help="Data type folder name")
    parser.add_argument("--window", type=float, default=0.1,
                        help="Window length in seconds")
    parser.add_argument("--step", type=float, default=0.02,
                        help="Step size in seconds")
    parser.add_argument("--model_order", type=int, default=4,
                        help="VAR model order")
    parser.add_argument("--n_permutations", type=int, default=200,
                        help="Number of permutations for statistics")
    parser.add_argument("--n_jobs", type=int, default=20,
                        help="Number of parallel jobs")
    
    args = parser.parse_args()
    main(**vars(args))
