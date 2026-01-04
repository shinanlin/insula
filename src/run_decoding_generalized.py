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
from decoding import load_roi_data, decode_permutation_scores
from decoder import generalized_permutation_scores


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
    
    baseline = np.transpose(baseline, (-1, 0, 1))
    p_cluster = 0.05
    P = baseline.shape[0]
    # scores: (n_time,n_time)
    # baseline: (n_perm, n_time, n_time)
    
    pvals = np.zeros(scores.shape)
    n_perm = baseline.shape[0]
    
    pvals_pt = ((baseline >= scores[None,...]).sum(axis=0) + 1 ) / (P + 1)   
    b_act = tail_compare(1.0 - pvals_pt, 1.0 - p_thresh, tails).astype(bool)  # (T, T)
    p_perm_full = proportion(baseline, tail=tails, axis=0)
    b_perm = tail_compare(p_perm_full, 1.0 - p_thresh, tails).astype(bool)  # (n_perm, T, T)

    mask = time_cluster(b_act, b_perm, p_val=1.0 - p_cluster, tails=tails)  # (T, T)

    return mask, p_perm_full



def main(
    bids_root,
    subject,
    description,
    band,
    datatype,
    variance,
    window,
    step,
    n_perm,
    n_folds,
    n_jobs,
    tmin=-1,
    tmax=1.5,
):
    Xs, ys = load_roi_data(
        bids_root,
        subject,
        description,
        band,
        datatype,
        tmin,
        tmax,
    )

    for i, (X, y) in enumerate(zip(Xs, ys)):
        
        logger.info(f"Processing file: {i}")
        
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
        accuracies, baseline_accuracies, _ = generalized_permutation_scores(
            X,
            y,
            cv,
            pipeline,
            n_jobs=n_jobs,
            n_permutations=n_perm,
            random_state=42,
            window=window,
            step=step,
            fs=fs,
            tmin=tmin,
            tmax=tmax,
        )
        
        # cluster correction for pval
        mask, p_values = cluster_correction(accuracies.mean(axis=-1), baseline_accuracies.mean(axis=-1))
        
        processing=None if len(Xs)==1 else str(i+1)
        save_path = BIDSPath(
            root = f'results/PhonemeSequence',
            datatype='(decode)(generalized)'+str(datatype),
            subject=subject,
            suffix=band,
            processing=processing,
            description=description,
            extension='.h5',
            check=False
        )
        save_path.mkdir(exist_ok=True)
        print(f"Saving results to: {save_path}")

        time_points = np.arange(tmin + window,tmax,step)
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
                        default="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/derivatives/decoding(ROI)",
                        help="Root directory of the BIDS dataset")
    parser.add_argument("--subject", type=str, default='Amygl',
                        help="Subject to process")
    parser.add_argument("--description", type=str, default='production',
                        help="perception or production")
    parser.add_argument("--band", type=str, default='highgamma',
                        help="highgamma or other band of neural signal")
    parser.add_argument("--datatype", type=str, default='phoneme',
                        help="what to classify? can be phoenem")
    parser.add_argument("--variance", type=float, default=0.8,
                        help="number of variance")
    parser.add_argument("--window", type=float, default=0.5,
                        help="window")
    parser.add_argument("--step", type=float, default=0.3,
                        help="step")
    parser.add_argument("--n_perm", type=int, default=3,
                        help="number of permutations")
    parser.add_argument("--n_folds", type=int, default=10,
                        help="number of folds")
    parser.add_argument("--n_jobs", type=int, default=2,
                        help="number of jobs")
    parser.add_argument("--tmin", type=float, default=-1,
                        help="tmin")
    parser.add_argument("--tmax", type=float, default=1.5,
                        help="tmax")

    args = parser.parse_args()
    main(**vars(args))
