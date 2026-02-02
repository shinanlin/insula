#!/usr/bin/env python3
"""
Test script to diagnose the slowdown when processing multiple files.
Simulates the exact scenario in run_decoding_resolved.py
"""
import rootutils
path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import time
import gc
import psutil
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from mne.decoding import Vectorizer
from ieeg.calc.oversample import MinimumNaNSplit
from run_decoding import load_roi_data
from decoder import decode_permutation_scores

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def test_file_iteration():
    """
    Test processing multiple files in sequence.
    This mimics the exact loop in run_decoding_resolved.py
    """
    
    # Parameters matching the actual script
    bids_root = "/cwork/ns458/BIDS-1.3_PictureNaming/BIDS/"
    ref = "bipolar"
    subject = "AICl"
    description = "Repeat"
    phase = "Stimulus"
    band = "highgamma"
    datatype = "token"
    tmin = -0.5
    tmax = 1.5
    variance = 0.8
    n_perm = 10  # Small for testing
    n_folds = 5
    n_jobs = 4
    window = 0.2
    step = 0.2  # Larger step for faster testing
    
    logger.info("="*60)
    logger.info("Loading data...")
    Xs, ys, paths = load_roi_data(
        bids_root, ref, subject, description, phase, band, datatype, tmin, tmax
    )
    
    logger.info(f"Loaded {len(Xs)} files")
    for i, (X, y, p) in enumerate(zip(Xs, ys, paths)):
        logger.info(f"  File {i}: X shape = {X.shape}, y shape = {y.shape}")
        logger.info(f"    Path: {p}")
    
    fs = 128
    time_points = np.arange(tmin + window, tmax + step, step)
    window_samples = int(window * fs)
    
    logger.info(f"\nTime points to process: {len(time_points)}")
    logger.info(f"Processing only first 3 time windows per file for speed")
    
    # Process ALL files (not just [:1])
    n_files = len(Xs)
    
    for file_idx, (X, y, path) in enumerate(zip(Xs, ys, paths)):
        
        logger.info(f"\n{'='*60}")
        logger.info(f"FILE {file_idx}/{n_files}: {path}")
        logger.info(f"Memory: {get_memory_usage():.1f} MB")
        
        cv = MinimumNaNSplit(n_splits=n_folds, n_repeats=1)
        pipeline = make_pipeline(
            Vectorizer(),
            StandardScaler(),
            PCA(n_components=variance, random_state=42),
            LinearSVC(random_state=42)
        )
        
        file_start = time.time()
        
        # Process only first 3 time windows for speed
        for t_idx, time_end in enumerate(time_points[:3]):
            
            end_sample = int((time_end - tmin) * fs) + 1
            start_sample = end_sample - window_samples
            
            if start_sample < 0 or end_sample > X.shape[-1]:
                logger.warning(f"Window out of bounds for time {time_end:.3f}s, skipping")
                continue
            
            X_segment = X.copy()[..., start_sample:end_sample]
            
            t0 = time.time()
            score, permutation_scores, _ = decode_permutation_scores(
                X_segment, y, cv, pipeline,
                n_jobs=n_jobs,
                n_permutations=n_perm,
                random_state=42,
            )
            t1 = time.time()
            
            logger.info(f"  Time {t_idx} ({time_end:.2f}s): {t1-t0:.2f}s, score={np.mean(score):.3f}")
        
        file_elapsed = time.time() - file_start
        logger.info(f"File {file_idx} total: {file_elapsed:.2f}s")
        
        # Clean up
        del X_segment
        gc.collect()
        logger.info(f"Memory after gc: {get_memory_usage():.1f} MB")


if __name__ == "__main__":
    test_file_iteration()
