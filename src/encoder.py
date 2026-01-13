#!/usr/bin/env python3
"""
Unified mTRF encoding analysis with clean separation of trial-level vs concatenated approaches.
"""
import rootutils
path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import h5py
import numpy as np
import sys
import copy
from mne_bids import BIDSPath
import pandas as pd
import mne
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from ieeg.calc.stats import time_perm_cluster
from einops import rearrange
import logging
from statsmodels.stats.multitest import fdrcorrection
from mtrf import TRF
import os
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def interpolate(epochs, min_trials_per_class=2):
    """
    Fill NaNs per channel and class using class/global mean & std.

    - For each channel and class:
      * If class has >= min_trials_per_class valid samples -> fill with class mean (or mean+noise*std)
      * Else -> fill with global mean (or mean+noise*std)
    - If global has no donors at all (extremely rare), fill 0.0

    Parameters
    ----------
    epochs : mne.Epochs
    min_trials_per_class : int
        Threshold for using class-specific statistics
    """
    data = epochs._data  # shape (n_epochs, n_channels, n_times)
    if not np.any(np.isnan(data)):
        print("No NaN values found, skipping interpolation")
        return

    n_epochs, n_channels, n_times = data.shape
    cond_labels = epochs.events[:, 2]
    unique_classes = np.unique(cond_labels)

    total_nans = int(np.isnan(data).sum())
    print(f"Interpolating {total_nans} NaN values ({total_nans/(n_epochs*n_channels*n_times)*100:.2f}% of data)")

    for ch in range(n_channels):
        channel = data[:, ch, :]  # view (epochs, times)

        # get nonnan trials
        nan_trials = np.any(np.isnan(channel), axis=1)
        global_valid = channel[~nan_trials]
        g_mean = np.mean(global_valid, axis=0)
        g_std  = np.std(global_valid, axis=0)
        
        for c in unique_classes:
            rows = np.where(cond_labels == c)[0]
            sub = channel[rows, :]                 # (n_rows, times)
            sub_nan_trials = np.any(np.isnan(sub), axis=-1)

            # if no nan trials continue
            if np.sum(sub_nan_trials)==0:
                continue
            
            # Class donors for this channel (all non-NaN samples in this class)
            class_valid = sub[~sub_nan_trials]
            n_class_valid = class_valid.shape[0]

            if n_class_valid >= min_trials_per_class:
                c_mean = np.mean(class_valid, axis=0)
                c_std  = np.std(class_valid, axis=0)
                mean_to_use, std_to_use = c_mean, c_std
            else:
                mean_to_use, std_to_use = g_mean, g_std

            # Prepare replacement values for all NaNs in this class
            N = class_valid.shape[-1]
            
            for k, nan_trial in enumerate(sub_nan_trials):
                if nan_trial:
                    channel[rows[k], :] = mean_to_use + np.random.randn(N) * 1e-2 * std_to_use

        # Persist filled channel back
        data[:, ch, :] = channel
    epochs._data = data
    print(f"Interpolation complete. Remaining NaNs: {int(np.isnan(data).sum())}")
    
    return

def load_pair_data(
    bids_root: str,
    feature_type: List[str],
    description: str,
    phase: str,
    subject: str,
    band: str='highgamma',
    concat: bool=False,
    n_folds: int=10,
):
    
    from scipy.stats import zscore
    
    # load neural data first
    bids_path = BIDSPath(
        root=bids_root + 'derivatives/epoch(bipolar)',
        subject=subject,
        suffix=band,
        description=description,
        processing=phase,
        datatype='epoch(band)(sig)(effective)',
        extension='.h5',
        check=False
    )

    # read epochs
    neural_epochs = mne.read_epochs(bids_path.match()[0], verbose='error')
    # neural.crop(tmin=0)
    fs = neural_epochs.info['sfreq']
    chn_names = neural_epochs.ch_names
    
    # load stimulus data
    bids_path = BIDSPath(
        root=bids_root + 'derivatives/features',
        subject=subject,
        extension='.h5',
        datatype='acoustic',
        check=False
    )
    stimulus_epochs = []

    # track the min N of each feature
    for ft in feature_type:
        ft_path = bids_path.update(suffix=ft).match()[0]
        s = mne.read_epochs(ft_path, verbose='error')
        # s.crop(tmin=0)
        stimulus_epochs.append(s)
    
    # crop the same time window
    tmin, tmax = -0.2, 0.8
    neural_epochs.crop(tmin=tmin, tmax=tmax)
    for s in stimulus_epochs:
        s.crop(tmin=tmin, tmax=tmax)
    
    interpolate(neural_epochs)
    
    # get data
    neural = neural_epochs.get_data()
    y = neural_epochs.events[:, 2]
    # concat along the feature dim 
    stimulus = np.concatenate([s.get_data() for s in stimulus_epochs], axis=1)
    
    # zscore to the HGA and features
    # neural = zscore(neural, axis=1)
    mask = (stimulus == 0)
    if mask.any():
        stimulus[mask] = np.random.uniform(1e-9, 1e-8, size=mask.sum()).astype(stimulus.dtype)
    # stimulus = zscore(stimulus, axis=-1)
    # set float32
    neural = neural.astype(np.float32)
    stimulus = stimulus.astype(np.float32)

    # assert the trial and time dim
    assert neural.shape[0] == stimulus.shape[0], "The number of trials does not match"
    assert neural.shape[-1] == stimulus.shape[-1], "The number of time points does not match"
    
    # to print the OOM error, concat small epoch to big chunks of trials of n_trials = n_folds
    if concat:
        # randomly assign the trials to n_folds
        rng = np.random.default_rng(42)
        idx = rng.permutation(neural.shape[0])
        neural = neural[idx]
        stimulus = stimulus[idx]
        
        folds = np.array_split(idx, n_folds)
        neural = [np.concatenate(neural[f], axis=-1).T for f in folds]
        stimulus = [np.concatenate(stimulus[f], axis=-1).T for f in folds]
    else:
        neural = [n.T for n in neural]
        stimulus = [s.T for s in stimulus]
    
    return neural, stimulus, fs, chn_names


def optimize_regularization(
    neural: List[np.ndarray],
    stimulus: List[np.ndarray],
    regularizations: np.ndarray,
    fs: int,
    tmin: float,
    tmax: float,
    n_folds: int = 10,
):
    
    model = TRF(direction=1)
    model.train(
        stimulus=stimulus,
        response=neural,
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        regularization=regularizations,
        k=10,
    )
    
    best_regularization = model.best_regularization
    times = model.times
    return best_regularization, times


def cross_validation_encoding(
    neural: List[np.ndarray],
    stimulus: List[np.ndarray],
    cv,
    encoder: TRF,
    fs: int,
    tmin: float,
    tmax: float,
    regularization: float = 1,
    n_jobs: int = 1,
):
        
    def _one_fold(tr, te):
        
        train_neural = [neural[i] for i in tr]
        test_neural = [neural[i] for i in te]
        train_stimulus = [stimulus[i] for i in tr]
        test_stimulus = [stimulus[i] for i in te]
        
        encoder.train(
            stimulus=train_stimulus,
            response=train_neural,
            fs=fs,
            tmin=tmin,
            tmax=tmax,
            regularization=regularization,
        )
        
        if encoder.direction == 1:
            _, pearsonr = encoder.predict(test_stimulus, test_neural, None, average=False)
            w = rearrange(encoder.weights, 'n_feature n_time n_channel -> n_channel n_feature n_time')
            
        elif encoder.direction == -1:
            _, pearsonr = encoder.predict(test_neural, test_stimulus, None, average=False)
            w = rearrange(encoder.weights, 'n_channel n_time n_feature -> n_channel n_feature n_time')
            
        return w, pearsonr
    
    results = Parallel(n_jobs=n_jobs)(delayed(_one_fold)(tr, te) for tr, te in cv.split(neural))
    
    # this is the kernel in shape (n_folds, n_channel, n_feature, n_lag)
    weights = np.stack([r[0] for r in results], axis=0)
    # this is the metric in shape (n_folds, n_channel) if direction == 1, or (n_folds, n_feature) if direction == -1
    pearsonr = np.stack([r[1] for r in results], axis=0)
    
    return weights, pearsonr
    
def permutation_test(
    neural: List[np.ndarray],
    stimulus: List[np.ndarray],
    cv,
    encoder: TRF,
    fs: int,
    tmin: float,
    tmax: float,
    regularization: float = 1,
    n_permutations: int = 10,
    n_jobs: int = 1,
    random_state: int = 42,
):
    
    weights_permuted = []
    pearsonr_permuted = []
    rng_fold = np.random.RandomState(random_state)
    seeds_fold = rng_fold.randint(0, 2**31 - 1, size=n_permutations)
    
    for tr, te in cv.split(neural):
        
        def _one_perm(seed):
            # permutate the pairs and rerun the cross-validation
            rng_perm = np.random.RandomState(seed)

            train_neural, test_neural = [neural[i] for i in tr], [neural[i] for i in te]
            train_stimulus, test_stimulus = [stimulus[i] for i in tr], [stimulus[i] for i in te]

            train_neural_p = train_neural.copy()
            rng_perm.shuffle(train_neural_p)
            
            # trim the neural/stimulus data to the min length of each element
            train_neural_p, train_stimulus = map(
                list,
                zip(*(
                    (n[:min(n.shape[0], s.shape[0])], s[:min(n.shape[0], s.shape[0])])
                    for n, s in zip(train_neural_p, train_stimulus)
                ))
            )
            encoder = TRF()
            encoder.train(
                stimulus=train_stimulus,
                response=train_neural_p,
                fs=fs,
                tmin=tmin,
                tmax=tmax,
                regularization=regularization,
            )
        
            if encoder.direction == 1:
                _, pearsonr = encoder.predict(test_stimulus, test_neural, None, average=False)
                w = rearrange(encoder.weights, 'n_feature n_time n_channel -> n_channel n_feature n_time')
                
            elif encoder.direction == -1:
                _, pearsonr = encoder.predict(test_neural, test_stimulus, None, average=False)
                w = rearrange(encoder.weights, 'n_channel n_time n_feature -> n_channel n_feature n_time')

            return w, pearsonr
        
        results = Parallel(n_jobs=n_jobs, batch_size=1)(delayed(_one_perm)(seed) for seed in tqdm(seeds_fold, desc='Permutation test'))
        weights_this_fold = np.stack([r[0] for r in results], axis=0)
        pearsonr_this_fold = np.stack([r[1] for r in results], axis=0)
    
        weights_permuted.append(weights_this_fold)
        pearsonr_permuted.append(pearsonr_this_fold)
    
    weights_permuted = np.stack(weights_permuted, axis=0)
    pearsonr_permuted = np.stack(pearsonr_permuted, axis=0)
    
    # collapse the fold dimension, 
    weights_permuted = np.mean(weights_permuted, axis=0)
    pearsonr_permuted = np.mean(pearsonr_permuted, axis=0)
    return weights_permuted, pearsonr_permuted

 
def main(
    bids_root: str,
    subject: str,
    band: str,
    feature_type: List[str],
    description: str,
    phase: str,
    direction: int,
    regularization: float,
    tmin: float,
    tmax: float,
    n_folds: int,
    n_perm: int,
    n_jobs: int,
    concat: bool,
):
    
    # load files: neural and behavioral/stimulus
    neural, stimulus, fs, chn_names = load_pair_data(
        bids_root,
        feature_type,
        description,
        phase,
        subject,
        band,
        concat=concat,
        n_folds=n_folds,
    )
    logger.info(f"Loaded data for subject {subject}, feature type {feature_type}, direction {direction}")
    # find the best regularization parameter
    optimal_regularization, times = optimize_regularization(
        neural,
        stimulus,
        np.logspace(-1, 10, 10),
        fs,
        tmin,
        tmax,
        n_folds
    )
    
    if regularization is None:
        optimal_regularization = optimal_regularization
    else:
        optimal_regularization = regularization
        
    logger.info(f"Setting regularization: {optimal_regularization}")

    # split the data into n_folds using sklearn KFold
    kfold = KFold(n_folds, shuffle=True, random_state=42)
    encoder = TRF(direction=direction)
    
    weights, pearsonr = cross_validation_encoding(
        neural=neural,
        stimulus=stimulus,
        cv=kfold,
        encoder=encoder,
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        regularization=optimal_regularization,
    )
    
    weights_permuted, pearsonr_permuted = permutation_test(
        neural=neural,
        stimulus=stimulus,
        cv=kfold,
        encoder=encoder,
        fs=fs,
        tmin=tmin,
        tmax=tmax,
        regularization=optimal_regularization,
        n_permutations=n_perm,
        n_jobs=n_jobs,
        random_state=42,
    )

    # perm test
    # weights_permuted in shape (n_perm, n_channel, n_feature, n_lag)
    # pearsonr_permuted in shape (n_perm, n_channel)
    
    # time perm cluster for weights, ignore channel adjacency
    weights = weights - np.mean(weights, axis=-1, keepdims=True)
    weights_permuted = weights_permuted - np.mean(weights_permuted, axis=-1, keepdims=True)
    
    mask, p_val = time_perm_cluster(
        weights,
        weights_permuted,
        p_thresh=0.01,
        ignore_adjacency=1,
        tails=2,
        n_perm=2000,
    )
    
    # obs_r: (n_channel,) - average across folds
    obs_r = np.mean(pearsonr, axis=0)

    # Two-sided test (recommended for correlation)
    p_one_sided = (np.sum(pearsonr_permuted >= obs_r, axis=0) + 1) / (n_perm + 1)
    
    # FDR correction across channels
    reject_fdr, p_fdr = fdrcorrection(p_one_sided, alpha=0.05, method='indep')
    
    feature_tag = f"({')('.join(feature_type)})"


    temp_pt = BIDSPath(
        root=os.path.join(bids_root, 'derivatives/epoch(bipolar)'),
        subject=subject,
        datatype='epoch(band)(sig)',
        processing='Stimulus',
        extension='.h5',
        check=False
    ).match()[0]
    
    task = temp_pt.task
    save_path = BIDSPath(
        root=f'results/{task}(bipolar)',
        subject=subject,
        task=task,
        description=temp_pt.description,
        processing=temp_pt.processing,
        datatype=f'mtrf',
        suffix=''.join(w[0] for w in feature_type if w),
        extension='.h5',
        check=False
    )
    save_path.mkdir(exist_ok=True)
    
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('times', data=times)
        f.create_dataset('weights', data=weights)
        f.create_dataset('weights_pval', data=p_val)
        f.create_dataset('weights_permuted', data=weights_permuted)
        f.create_dataset('pearsonr', data=pearsonr)
        f.create_dataset('pearsonr_permuted', data=pearsonr_permuted)
        f.create_dataset('mask', data=mask)
        f.create_dataset('pearsonr_pval', data=p_fdr)
        f.create_dataset('fdr_mask', data=reject_fdr)
        f.create_dataset('chn_names', data=chn_names)
        
        f.attrs['regularization'] = optimal_regularization
        f.attrs['direction'] = direction
        f.attrs['tmin'] = tmin
        f.attrs['tmax'] = tmax
        f.attrs['fs'] = fs
        f.attrs['n_folds'] = n_folds
        f.attrs['n_perm'] = n_perm
        f.attrs['n_jobs'] = n_jobs
        
    logger.info(f"Encoding analysis completed, saved to: {save_path}")

    return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified mTRF encoding analysis")
    
    parser.add_argument("--bids_root", type=str,
                        default="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/",
                        help="Root directory of the BIDS dataset")
    parser.add_argument("--subject", type=str, default='D0040',
                        help="Subject to process")
    parser.add_argument("--description", type=str, default='Repeat',
                        help="Description of the neural signal")
    parser.add_argument("--phase", type=str, default='Stimulus',
                        choices=['Stimulus', 'Response'],
                        help="Phase of the neural signal")
    parser.add_argument("--band", type=str, default='highgamma',
                        help="Neural signal band")
    parser.add_argument("--feature_type",nargs="+",
                        type=str,default=["envelope"],help="Feature type(s)")
    parser.add_argument("--n_folds", type=int, default=10,
                        help="Number of CV folds")
    parser.add_argument("--n_jobs", type=int, default=12,
                        help="Number of parallel jobs")
    parser.add_argument("--direction", type=int, default=1,
                        help="TRF direction")
    parser.add_argument("--tmin", type=float, default=-0.3,
                        help="TRF start time (seconds)")
    parser.add_argument("--tmax", type=float, default=0.7,
                        help="TRF end time (seconds)")
    parser.add_argument("--n_perm", type=int, default=20,
                        help="Number of permutations")
    parser.add_argument("--regularization", type=float, default=None,
                        help="Regularization parameter")
    parser.add_argument("--concat", default=False, type=bool,
                        help="Concatenate the epochs")
    
    args = parser.parse_args()
    
    main(**vars(args))