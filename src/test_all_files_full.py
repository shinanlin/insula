#!/usr/bin/env python3
"""
Test script to simulate the EXACT scenario when processing ALL files.
This mimics changing Xs[:1] to Xs[:] in run_decoding_resolved.py
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


def test_scenario(use_all_files=False):
    """
    Test processing files.
    use_all_files=False: mimics Xs[:1] (only first file)
    use_all_files=True: mimics Xs[:] (all files)
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
    n_perm = 20  # Realistic number
    n_folds = 10
    n_jobs = 8
    window = 0.2
    step = 0.1  # Realistic step
    
    logger.info("="*60)
    logger.info(f"SCENARIO: {'ALL FILES' if use_all_files else 'FIRST FILE ONLY'}")
    logger.info("="*60)
    
    logger.info("Loading data...")
    Xs, ys, paths = load_roi_data(
        bids_root, ref, subject, description, phase, band, datatype, tmin, tmax
    )
    
    logger.info(f"Loaded {len(Xs)} files")
    
    fs = 128
    time_points = np.arange(tmin + window, tmax + step, step)
    window_samples = int(window * fs)
    
    # Select files based on scenario
    if use_all_files:
        selected_Xs = Xs[:]
        selected_ys = ys[:]
        selected_paths = paths[:]
    else:
        selected_Xs = Xs[:1]
        selected_ys = ys[:1]
        selected_paths = paths[:1]
    
    logger.info(f"Processing {len(selected_Xs)} file(s)")
    logger.info(f"Time points: {len(time_points)} (only processing first 5 for speed)")
    
    total_start = time.time()
    
    for file_idx, (X, y, path) in enumerate(zip(selected_Xs, selected_ys, selected_paths)):
        
        logger.info(f"\n{'='*60}")
        logger.info(f"FILE {file_idx}: {path}")
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        logger.info(f"Memory: {get_memory_usage():.1f} MB")
        
        cv = MinimumNaNSplit(n_splits=n_folds, n_repeats=1)
        pipeline = make_pipeline(
            Vectorizer(),
            StandardScaler(),
            PCA(n_components=variance, random_state=42),
            LinearSVC(random_state=42)
        )
        
        file_start = time.time()
        
        # Process only first 5 time windows for speed
        for t_idx, time_end in enumerate(time_points[:5]):
            
            end_sample = int((time_end - tmin) * fs) + 1
            start_sample = end_sample - window_samples
            
            if start_sample < 0 or end_sample > X.shape[-1]:
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
        logger.info(f"File {file_idx} completed in {file_elapsed:.2f}s")
        
        gc.collect()
    
    total_elapsed = time.time() - total_start
    logger.info(f"\n{'='*60}")
    logger.info(f"TOTAL TIME: {total_elapsed:.2f}s")
    logger.info(f"Final memory: {get_memory_usage():.1f} MB")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        test_scenario(use_all_files=True)
    else:
        test_scenario(use_all_files=False)
