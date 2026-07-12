#!/usr/bin/env python3
"""Decoding script for phoneme classification using PCA-SVM."""

import rootutils
# add the root path to the python path for importing
path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import argparse
from pathlib import Path
from typing import Tuple, List, Optional

import h5py
import numpy as np
from mne_bids import BIDSPath
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, permutation_test_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score, accuracy_score
)
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from src.decoding.decoder import decode_permutation_scores
from ieeg.calc.oversample import MinimumNaNSplit
from mne.decoding import Vectorizer
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import sys
import logging
import os
import gc
import time as _time

# Simple logging: everything INFO and above to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def load_roi_data(
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
    """Load and preprocess neural data from a specific brain region (ROI).
    
    Loads neural time series data from HDF5 files following BIDS structure,
    then crops the temporal window to -0.5 to 0.5 seconds relative to stimulus.
    
    Parameters
    ----------
    bids_root : str or Path
        Root directory of BIDS dataset containing neural data
    ref : str
        Reference scheme (e.g., 'car', 'bipolar')
    roi : str
        Region of interest identifier (e.g., 'PrGl', 'STGl')
    description : str
        Task description (e.g., 'production', 'perception')
    band : str
        Frequency band (e.g., 'highgamma', 'beta')
    datatype : str
        Data type (e.g., 'phoneme', 'word')
    tmin : float
        Start time of the temporal window in seconds
    tmax : float
        End time of the temporal window in seconds
        
    Returns
    -------
    X : ndarray, shape (n_epochs, n_channels, n_times)
        Neural time series data cropped to 1-second window
    y : ndarray, shape (n_epochs,)
        Class labels for each epoch
        
    Raises
    ------
    FileNotFoundError
        If no matching files found for the specified ROI
    """
    # Construct BIDS path for the ROI data file
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
    paths = []
    
    for roi_file in roi_files:
        
        data = h5py.File(roi_file, 'r')
        X = data['X'][:]
        y = data['y'][:]
        
        t_start = data.attrs['tmin']
        t_end = data.attrs['tmax']
        fs = data.attrs['fs']
        
        start_idx = int(fs * (tmin - t_start))  # Start at tmin seconds (originally -0.5s relative)
        end_idx = int(fs * (tmax - t_start))    # End at tmax seconds (originally +0.5s relative)
        X = X[:, :, start_idx:end_idx]
        
        data.close()
        Xs.append(X)
        ys.append(y)
        paths.append(roi_file)
    
    
    return Xs, ys, paths

def main(
    bids_root,
    subject,
    ref,
    description,
    phase,
    band,
    datatype,
    variance,
    n_perm,
    n_folds,
    n_jobs,
):

    match phase:
        case 'Stimulus':
            tmin, tmax = 0, 0.75
        case 'Delay':
            tmin, tmax = 0, 0.75
        case 'Go':
            tmin, tmax = 0, 0.5
        case 'Response':
            tmin, tmax = -0.5, 0
        case _:
            raise ValueError(f"Unknown phase: {phase}")
    
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
    
    # now just doing the first phoneme
    for i, (X, y, path) in enumerate(zip(Xs[:2], ys[:2], paths[:2])):
        _t0 = _time.time()
        logger.info(f"Processing file {i}/{len(Xs)}: {path}")
        
        cv = MinimumNaNSplit(n_splits=n_folds, n_repeats=1)
        
        logger.info('Making pipeline with variance %f', variance)
        
        decoder = make_pipeline(
            Vectorizer(),
            StandardScaler(),
            PCA(n_components=variance, random_state=42),
            SVC(kernel='linear', random_state=42)
        )
        
        obs_scores, perm_scores, p_value = decode_permutation_scores(
            X,
            y,
            cv,
            decoder,
            n_jobs=n_jobs,
            n_permutations=n_perm,
        )
        
        save_path = BIDSPath(
            root = os.path.join('results', f'{path.task}(roi)({ref})'),
            datatype='(decode)'+str(datatype),
            subject=subject,
            suffix=band,
            processing=path.processing,
            description=path.description,
            recording=path.recording,
            extension='.h5',
            check=False
        )
        save_path.mkdir(exist_ok=True)

        logger.info('Saving results to %s', save_path)
        with h5py.File(save_path, "w") as f:
            # Create a group for each feature type
            f.create_dataset(name="accuracy", data=obs_scores)
            f.create_dataset(name='perm_scores', data=perm_scores)
            f.create_dataset(name='p_value', data=p_value)

            f.attrs["fs"] = 128
            f.attrs["tmin"] = tmin
            f.attrs["tmax"] = tmax
            f.attrs["variance"] = variance
            f.attrs["n_perm"] = n_perm
            f.attrs["n_folds"] = n_folds
            f.attrs["n_jobs"] = n_jobs

        logger.info(f"File {i} completed in {_time.time() - _t0:.2f}s")
        # Explicit memory cleanup between files
        Xs[i] = None
        ys[i] = None
        del X, y, decoder, obs_scores, perm_scores, p_value, cv
        gc.collect()

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--bids_root", type=str,
                        default="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/",
                        help="Root directory of the BIDS dataset")
    parser.add_argument("--subject", type=str, default='AICl',
                        help="Subject to process")
    parser.add_argument("--ref", type=str, default='bipolar',
                        choices=['car', 'bipolar'],
                        help="ROI to process")
    parser.add_argument("--description", type=str, default='Repeat',
                        choices=['Repeat', 'Passive','Decision'],
                        help="Repeat, Passive, or Decision")
    parser.add_argument("--phase", type=str, default='Stimulus',
                        choices=['Stimulus', 'Delay', 'Go', 'Response'],
                        help="Stimulus, Delay, Go, or Response")
    parser.add_argument("--band", type=str, default='highgamma',
                        help="highgamma or other band of neural signal")
    parser.add_argument("--datatype", type=str, default='phoneme',
                        choices=['phoneme','articulator','token','lexicality'],
                        help="what to classify? can be phoneme, articulator, token, or lexicality")
    parser.add_argument("--variance", type=float, default=0.85,
                        help="number of variance")
    parser.add_argument("--n_perm", type=int, default=2,
                        help="number of permutations")
    parser.add_argument("--n_folds", type=int, default=10,
                        help="number of folds")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="number of jobs")

    args = parser.parse_args()
    main(**vars(args))