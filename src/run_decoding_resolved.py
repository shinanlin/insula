#!/usr/bin/env python3
"""
This script is used to generate time resolved decoding accuracy.
NOTE: not cross decoding, just the regular perception->perception, production->production
not in the generalization way, just train on t and test on t
"""
import rootutils
# add the root path to the python path for importing
path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import argparse
from pathlib import Path
from typing import Tuple, List, Optional

import h5py
import numpy as np
import sys
from mne_bids import BIDSPath
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from mne.decoding import Vectorizer, SlidingEstimator
from sklearn.model_selection import StratifiedKFold
from ieeg.decoding.decode import Decoder
from ieeg.calc.oversample import MinimumNaNSplit
from run_decoding import load_roi_data, decode_permutation_scores


import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)
RANDOM_SEED = 42


def cluster_correction(scores, baseline, p_thresh=0.05, tails=1):
    
    from scipy.stats import permutation_test
    from ieeg.calc.stats import time_cluster, proportion, tail_compare
    
    # scores: (n_time,)
    # baseline: (n_perm, n_time)
    
    pvals = np.zeros(scores.shape)
    n_perm = baseline.shape[0]

    diff = baseline - scores[None, :]  # shape (n_perm, n_time)
    p_act = (np.sum(diff >= 0, axis=0) + 1) / (diff.shape[0] + 1)
    # Calculate the p value of the permutation distribution
    p_perm = proportion(diff, tail=tails, axis=0)
    # Create binary clusters using the p value threshold
    b_act = tail_compare(1. - p_act, 1. - p_thresh, tails)
    b_perm = tail_compare(p_perm, 1. - p_thresh, tails)
    mask = time_cluster(b_act, b_perm, 1 - p_thresh, tails)
    
    return mask, p_act

def main(
    bids_root,
    ref,
    subject,
    description,
    phase,
    band,
    datatype,
    variance,
    window,
    step,
    n_perm,
    n_folds,
    n_jobs,
    tmin=-0.5,
    tmax=1.5,
):
    
    Xs, ys, paths = load_roi_data(
        bids_root,
        ref,
        subject,
        description,
        phase,
        band,
        datatype,
        tmin,
        tmax,
    )

    # process only the first phoneme file position
    for i, (X, y, path) in enumerate(zip(Xs[:1], ys[:1], paths[:1])):
        
        logger.info(f"Processing file: {i}")
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        logger.info(f"File path: {path}")
        
        cv = MinimumNaNSplit(n_splits=n_folds, n_repeats=1)
        logger.info('Making pipeline with variance %f', variance)
        # make pipeline
        pipeline = make_pipeline(
            Vectorizer(),
            StandardScaler(),
            PCA(n_components=variance, random_state=42),
            LinearSVC(random_state=42)
        )
        
        # window
        fs = 128
        time_points = np.arange(tmin + window,
                            tmax + step,
                            step)
        window_samples = int(window * fs)
        step_samples = int(step * fs)
        
        accuracies = np.zeros((len(time_points), n_folds))
        baseline_accuracies = np.zeros((len(time_points), n_folds, n_perm))
        
        for t_idx, time_end in enumerate(time_points):
            
            end_sample = int((time_end - tmin) * fs) + 1
            start_sample = end_sample - window_samples
            
            # Bounds checking
            if start_sample < 0 or end_sample > X.shape[-1]:
                logger.warning(f"Window out of bounds for time {time_end:.3f}s, skipping")
                continue
            
            # make a copy
            X_segment = X.copy()[..., start_sample:end_sample]
            logger.info(f"Processing time window: {time_end:.3f}s, samples {start_sample}:{end_sample}")
            
            score, permutation_scores, _ = decode_permutation_scores(
                X_segment,
                y,
                cv,
                pipeline,
                n_jobs=n_jobs,
                n_permutations=n_perm,
                random_state=42,
            )
            
            accuracies[t_idx] = score
            baseline_accuracies[t_idx] = permutation_scores
        
        # cluster correction for pval
        # accuracies: (n_time, n_folds) -> mean over folds -> (n_time,)
        # baseline_accuracies: (n_time, n_folds, n_perm) -> mean over folds (axis=1) -> (n_time, n_perm) -> .T -> (n_perm, n_time)
        mask, p_values = cluster_correction(accuracies.mean(axis=-1), baseline_accuracies.mean(axis=1).T)
        
        save_path = BIDSPath(
            root = f'results/{path.task}(roi)({ref})',
            datatype='(decode)(resolved)'+str(datatype),
            subject=subject,
            suffix=band,
            processing=path.processing,
            description=path.description,
            recording=path.recording,
            extension='.h5',
            check=False
        )
        save_path.mkdir(exist_ok=True)
        print(f"Saving results to: {save_path}")

        with h5py.File(save_path, "w") as f:
            # Create a group for each feature type
            f.create_dataset(name='accuracy', data=accuracies)
            f.create_dataset(name='baseline', data=baseline_accuracies)
            f.create_dataset(name='time', data=time_points)
            f.create_dataset(name='mask', data=mask)
            f.create_dataset(name='p_values', data=p_values)

            f.attrs["fs"] = fs
            f.attrs["tmin"] = tmin
            f.attrs["tmax"] = tmax

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--bids_root", type=str,
                        default="/cwork/ns458/BIDS-1.3_PictureNaming/BIDS/",
                        help="Root directory of the BIDS dataset")
    # parser.add_argument("--bids_root", type=str,
                        # default="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/",
                        # help="Root directory of the BIDS dataset")
    parser.add_argument("--subject", type=str, default='HGr',
                        help="Subject to process")
    parser.add_argument("--ref", type=str, default='bipolar',
                        choices=['car', 'bipolar'],
                        help="Reference scheme")
    parser.add_argument("--description", type=str, default='Repeat',
                        choices=['Repeat', 'Decision', 'Passive'],
                        help="perception or production")
    parser.add_argument("--phase", type=str, default='Stimulus',
                        choices=['Stimulus', 'Delay', 'Go', 'Response'],
                        help="the phase for decoding")
    parser.add_argument("--band", type=str, default='highgamma',
                        help="highgamma or other band of neural signal")
    parser.add_argument("--datatype", type=str, default='token',
                        choices=['phoneme','articulator','structure',
                                 'word', 'token','lexicality'])
    parser.add_argument("--variance", type=float, default=0.8,
                        help="number of variance")
    parser.add_argument("--window", type=float, default=0.6,
                        help="window length in seconds")
    parser.add_argument("--step", type=float, default=0.5,
                        help="step size in seconds")
    parser.add_argument("--n_perm", type=int, default=3,
                        help="number of permutations")
    parser.add_argument("--n_folds", type=int, default=10,
                        help="number of folds")
    parser.add_argument("--n_jobs", type=int, default=2,
                        help="number of jobs")
    parser.add_argument("--tmin", type=float, default=-0.5,
                        help="tmin")
    parser.add_argument("--tmax", type=float, default=1.5,
                        help="tmax")

    args = parser.parse_args()
    main(**vars(args))
