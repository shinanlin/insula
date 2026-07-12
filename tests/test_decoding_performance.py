#!/usr/bin/env python3
"""
Test script to diagnose the decoding performance issue.
This script tests whether the slowdown is caused by:
1. Memory accumulation
2. Joblib process pool issues
3. Data loading issues
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


def test_multiple_datasets():
    """Test decoding on multiple datasets to see if there's a slowdown."""
    
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
    n_perm = 3  # Small number for testing
    n_folds = 5  # Small number for testing
    n_jobs = 4   # Smaller for testing
    window = 0.2
    step = 0.5   # Larger step for faster testing
    
    logger.info("Loading data...")
    Xs, ys, paths = load_roi_data(
        bids_root, ref, subject, description, phase, band, datatype, tmin, tmax
    )
    
    logger.info(f"Loaded {len(Xs)} datasets")
    for i, (X, y) in enumerate(zip(Xs, ys)):
        logger.info(f"  Dataset {i}: X shape = {X.shape}, y shape = {y.shape}")
    
    # Test processing each dataset
    times_per_dataset = []
    memory_per_dataset = []
    
    cv = MinimumNaNSplit(n_splits=n_folds, n_repeats=1)
    pipeline = make_pipeline(
        Vectorizer(),
        StandardScaler(),
        PCA(n_components=variance, random_state=42),
        LinearSVC(random_state=42)
    )
    
    fs = 128
    time_points = np.arange(tmin + window, tmax + step, step)
    window_samples = int(window * fs)
    
    # Process only first 3 datasets or all if less
    n_datasets = min(3, len(Xs))
    
    for i in range(n_datasets):
        X, y, path = Xs[i], ys[i], paths[i]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing dataset {i}: {path}")
        logger.info(f"Memory before: {get_memory_usage():.1f} MB")
        
        start_time = time.time()
        
        # Process just 2 time windows for testing
        for t_idx, time_end in enumerate(time_points[:2]):
            end_sample = int((time_end - tmin) * fs) + 1
            start_sample = end_sample - window_samples
            
            if start_sample < 0 or end_sample > X.shape[-1]:
                continue
            
            X_segment = X.copy()[..., start_sample:end_sample]
            logger.info(f"  Time window {t_idx}: {time_end:.3f}s, X_segment shape: {X_segment.shape}")
            
            t0 = time.time()
            score, permutation_scores, _ = decode_permutation_scores(
                X_segment, y, cv, pipeline,
                n_jobs=n_jobs,
                n_permutations=n_perm,
                random_state=42,
            )
            t1 = time.time()
            logger.info(f"    Decoding took {t1-t0:.2f}s, score: {np.mean(score):.3f}")
        
        elapsed = time.time() - start_time
        memory_after = get_memory_usage()
        
        times_per_dataset.append(elapsed)
        memory_per_dataset.append(memory_after)
        
        logger.info(f"Dataset {i} completed in {elapsed:.2f}s")
        logger.info(f"Memory after: {memory_after:.1f} MB")
        
        # Force garbage collection between datasets
        gc.collect()
        logger.info(f"Memory after gc: {get_memory_usage():.1f} MB")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    for i, (t, m) in enumerate(zip(times_per_dataset, memory_per_dataset)):
        logger.info(f"Dataset {i}: {t:.2f}s, {m:.1f} MB")
    
    if len(times_per_dataset) > 1:
        ratio = times_per_dataset[1] / times_per_dataset[0] if times_per_dataset[0] > 0 else 0
        logger.info(f"\nTime ratio (dataset 1 / dataset 0): {ratio:.2f}x")
        if ratio > 2:
            logger.warning("SIGNIFICANT SLOWDOWN DETECTED!")
        else:
            logger.info("No significant slowdown detected.")


def test_joblib_isolation():
    """Test if joblib process pool causes issues across iterations."""
    from joblib import Parallel, delayed
    
    logger.info("\n" + "="*60)
    logger.info("Testing joblib isolation...")
    
    def dummy_work(x):
        return x ** 2
    
    for iteration in range(3):
        t0 = time.time()
        results = Parallel(n_jobs=4)(delayed(dummy_work)(i) for i in range(1000))
        t1 = time.time()
        logger.info(f"Iteration {iteration}: {t1-t0:.4f}s")
        gc.collect()


if __name__ == "__main__":
    logger.info("Starting performance diagnostic tests...")
    
    # Test 1: Joblib isolation
    test_joblib_isolation()
    
    # Test 2: Multiple datasets
    test_multiple_datasets()
