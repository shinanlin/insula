#!/usr/bin/env python3
"""
This script computes time-resolved decoder patterns (Haufe et al., 2014) 
instead of regular decoding accuracies. It helps interpret the spatial 
distribution (neurophysiological sources) of the classifier weights.

Reference: Haufe et al., 2014. "On the interpretation of weight vectors 
of linear models in multivariate neuroimaging." NeuroImage.
"""
import rootutils
path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import argparse
from pathlib import Path
import os
import sys

import h5py
import numpy as np
import logging
import time as _time

from mne_bids import BIDSPath
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from mne.decoding import Vectorizer
from ieeg.calc.oversample import MinimumNaNSplit
from src.decoder import sample_fold
from joblib import Parallel, delayed
from sklearn.base import clone

import gc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def load_roi_data_with_channels(
    bids_root, 
    ref,
    roi, 
    description, 
    phase,
    band, 
    datatype,
    tmin,
    tmax,
):
    root = BIDSPath(
        root=os.path.join(bids_root, 'derivatives', f'decoding({ref})'), 
        datatype=datatype, 
        description=description,
        suffix=band, 
        processing=phase,
        extension='.h5', 
        check=False
    )
    roi_path = root.copy().update(subject=roi)
    roi_files = roi_path.match()
    
    if not roi_files:
        raise FileNotFoundError(f"No files found for ROI {roi}")
        
    Xs, ys = [], []
    channels_list = []
    paths = []
    
    for roi_file in roi_files:
        data = h5py.File(roi_file, 'r')
        X = data['X'][:]
        y = data['y'][:]
        channels = [ch.decode('utf-8') for ch in data['channel'][:]]
        
        t_start = data.attrs['tmin']
        t_end = data.attrs['tmax']
        fs = data.attrs['fs']
        
        start_idx = int(fs * (tmin - t_start))  
        end_idx = int(fs * (tmax - t_start))    
        X = X[:, :, start_idx:end_idx]
        
        data.close()
        Xs.append(X)
        ys.append(y)
        channels_list.append(channels)
        paths.append(roi_file)
    
    return Xs, ys, channels_list, paths


def compute_haufe_pattern(X_raw, y_pred):
    """
    Computes the spatial pattern of a linear classifier according to Haufe et al. 2014.
    A = Cov(X, s_hat) / Var(s_hat)
    
    Parameters
    ----------
    X_raw : np.ndarray
        The raw input data of shape (n_trials, n_features).
        In our case, n_features = n_channels * n_times within the window.
    y_pred : np.ndarray
        The continuous decision function outputs (latent signal) of shape (n_trials,).
        
    Returns
    -------
    A : np.ndarray
        The activation pattern of shape (n_features,).
    """
    N = len(y_pred)
    # Center the variables
    X_centered = X_raw - np.mean(X_raw, axis=0)
    s_centered = y_pred - np.mean(y_pred)
    
    # Covariance between each feature and the latent signal s_hat
    cov_xs = np.dot(X_centered.T, s_centered) / (N - 1)
    
    # Variance of the latent signal
    var_s = np.var(y_pred, ddof=1)
    
    # To avoid division by zero if variance is extremely small
    if var_s < 1e-10:
        return np.zeros_like(cov_xs)
        
    return cov_xs / var_s


def compute_patterns_cv(X, y, cv, pipeline):
    """
    Computes the Haufe patterns across cross-validation splits.
    
    Parameters
    ----------
    X : np.ndarray
        The input data of shape (n_trials, n_channels, n_times_in_window)
    y : np.ndarray
        The true labels.
    cv : scikit-learn cross-validator
    pipeline : scikit-learn pipeline
    
    Returns
    -------
    mean_pattern : np.ndarray
        The averaged pattern across folds of shape (n_channels, n_times_in_window)
    """
    patterns = []
    
    for train, test in cv.split(X, y):
        # sample_fold cleanly removes any trials that contain NaNs in this specific window
        X_train, X_test, y_train, y_test = sample_fold(
            X,
            y,
            train,
            test,
        )
        
        # Fit the pipeline on the NaN-free training data
        pipeline.fit(X_train, y_train)
        
        # We need the latent decision function on the TRAINING data to estimate the true 
        # neurophysiological generators of the model we just trained
        y_pred_train = pipeline.decision_function(X_train)
        
        # Flatten X_train just for the Haufe pattern computation (trials, features)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        A_flat = compute_haufe_pattern(X_train_flat, y_pred_train)
        
        # Reshape back to (n_channels, n_times_in_window)
        A = A_flat.reshape(X.shape[1], X.shape[2])
        patterns.append(A)
        
    return np.mean(patterns, axis=0)


def main(
    bids_root,
    ref,
    roi,
    description,
    phase,
    band,
    datatype,
    variance,
    window,
    step,
    n_folds,
    tmin=-0.5,
    tmax=1.5,
):
    
    Xs, ys, channels_list, paths = load_roi_data_with_channels(
        bids_root,
        ref,
        roi,
        description,
        phase,
        band,
        datatype,
        tmin,
        tmax,
    )
    
    n_files = len(Xs)
    logger.info(f"Loaded {n_files} files to process for Pattern extraction")

    for i in range(n_files):
        X, y, channels, path = Xs[i], ys[i], channels_list[i], paths[i]
        file_t0 = _time.time()
        
        logger.info(f"Processing file: {i}")
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        logger.info(f"Channels: {len(channels)}")
        logger.info(f"File path: {path}")
        
        # We increase n_repeats purely to get a stable pattern estimation over multiple splits
        cv = MinimumNaNSplit(n_splits=n_folds, n_repeats=5)
        logger.info('Making pipeline with variance %f', variance)
        
        pipeline = make_pipeline(
            Vectorizer(),
            StandardScaler(),
            PCA(n_components=variance, random_state=42),
            LinearSVC(random_state=42, max_iter=10000)
        )
        
        fs = 128
        time_points = np.arange(tmin + window, tmax + step, step)
        window_samples = int(window * fs)
        
        # Data structure to hold patterns
        # Shape: (n_times, n_channels, window_samples)
        patterns_time_resolved = np.zeros((len(time_points), X.shape[1], window_samples))
        
        def _process_t_idx(t_idx, time_end):
            end_sample = int((time_end - tmin) * fs)
            start_sample = end_sample - window_samples
            
            if start_sample < 0 or end_sample > X.shape[-1]:
                logger.warning(f"Window out of bounds for time {time_end:.3f}s, skipping")
                return t_idx, None
            
            X_segment = X.copy()[..., start_sample:end_sample]
            
            # Clone pipeline to avoid shared state in parallel
            pipeline_clone = clone(pipeline)
            
            # Compute average Haufe pattern for this specific sliding time-window
            avg_pattern = compute_patterns_cv(X_segment, y, cv, pipeline_clone)
            return t_idx, avg_pattern

        logger.info(f"Extracting patterns across {len(time_points)} time windows (Parallel)...")
        results = Parallel(n_jobs=30)(
            delayed(_process_t_idx)(t_idx, time_end) 
            for t_idx, time_end in enumerate(tqdm(time_points, desc="Computing Patterns"))
        )
        
        for t_idx, avg_pattern in results:
            if avg_pattern is not None:
                patterns_time_resolved[t_idx] = avg_pattern
            
        save_path = BIDSPath(
            root=os.path.join('results', f'{path.task}(roi)({ref})'),
            datatype='(pattern)' + str(datatype),  # e.g., (pattern)lexicality
            subject=roi,
            suffix=band,
            processing=path.processing,
            description=path.description,
            recording=path.recording,
            extension='.h5',
            check=False
        )
        save_path.mkdir(exist_ok=True)

        logger.info('Saving pattern results to %s', save_path)
        with h5py.File(save_path, "w") as f:
            f.create_dataset(name="patterns", data=patterns_time_resolved)
            f.create_dataset(name="time_points", data=time_points)
            f.create_dataset(name="channels", data=[ch.encode('utf-8') for ch in channels])

            f.attrs["fs"] = fs
            f.attrs["tmin_overall"] = tmin
            f.attrs["tmax_overall"] = tmax
            f.attrs["window"] = window
            f.attrs["step"] = step
            f.attrs["variance"] = variance
            f.attrs["n_folds"] = n_folds

        logger.info(f"File {i} completed in {_time.time() - file_t0:.2f}s")
        
        Xs[i] = None
        ys[i] = None
        del X, y, pipeline, patterns_time_resolved, cv
        gc.collect()

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Decoder Patterns across time windows.")
    parser.add_argument("--bids_root", type=str, default="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS")
    parser.add_argument("--ref", type=str, default="bipolar")
    parser.add_argument("--roi", type=str, default='AICl')
    parser.add_argument("--description", type=str, default='Repeat')
    parser.add_argument("--phase", type=str, default='Response', help="Stimulus, Delay, Go, or Response")
    parser.add_argument("--band", type=str, default="highgamma")
    parser.add_argument("--datatype", type=str, default="lexicality")
    parser.add_argument("--variance", type=float, default=0.8)
    parser.add_argument("--window", type=float, default=0.2, help="Sliding window size in seconds")
    parser.add_argument("--step", type=float, default=0.1, help="Sliding window step size in seconds")
    parser.add_argument("--n_folds", type=int, default=10)
    args = parser.parse_args()

    # Time limits corresponding to phases based on the 1D plots
    match args.phase:
        case 'Stimulus':
            tmin, tmax = -0.2, 1.0
        case 'Delay':
            tmin, tmax = -0.2, 1.0
        case 'Go':
            tmin, tmax = -0.2, 1.0
        case 'Response':
            tmin, tmax = -0.2, 1.0
        case _:
            raise ValueError(f"Unknown phase: {args.phase}")

    main(
        bids_root=args.bids_root,
        ref=args.ref,
        roi=args.roi,
        description=args.description,
        phase=args.phase,
        band=args.band,
        datatype=args.datatype,
        variance=args.variance,
        window=args.window,
        step=args.step,
        n_folds=args.n_folds,
        tmin=tmin,
        tmax=tmax,
    )
