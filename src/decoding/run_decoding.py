#!/usr/bin/env python3
"""Windowed ROI decoding with PCA-LinearSVC and permutation testing."""

import rootutils
# add the root path to the python path for importing
path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import argparse
import gc
import logging
import os
import sys
import time as _time

import h5py
import numpy as np
from ieeg.calc.oversample import MinimumNaNSplit
from mne.decoding import Vectorizer
from mne_bids import BIDSPath
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.decoding.decoder import (
    decode_cv_scores,
    decode_permutation_scores,
    get_cv_predict,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42

# Fixed decoding windows per phase (seconds, relative to phase onset).
PHASE_WINDOWS = {
    'Stimulus': (0.0, 0.5),
    'Delay': (0.0, 0.7),
    'Go': (0.0, 0.5),
    'Response': (-0.5, 0.5),
}


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
    then crops the temporal window to the requested interval.

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
    Xs : list of ndarray
        Neural time series per matched file, cropped to the window
    ys : list of ndarray
        Class labels per file
    paths : list
        Matched BIDS paths

    Raises
    ------
    FileNotFoundError
        If no matching files found for the specified ROI
    """
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

    Xs, ys, paths = [], [], []

    for roi_file in roi_files:
        data = h5py.File(roi_file, 'r')
        X = data['X'][:]
        y = data['y'][:]

        t_start = data.attrs['tmin']
        fs = data.attrs['fs']

        start_idx = int(fs * (tmin - t_start))
        end_idx = int(fs * (tmax - t_start))
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
    n_repeats,
    n_jobs,
):
    try:
        tmin, tmax = PHASE_WINDOWS[phase]
    except KeyError as exc:
        raise ValueError(f"Unknown phase: {phase}") from exc

    logger.info('Phase window: %s [%.2f, %.2f] s', phase, tmin, tmax)

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

    n_files = len(Xs)
    logger.info(f"Loaded {n_files} files to process")

    for i in range(n_files):
        X, y, path = Xs[i], ys[i], paths[i]
        _t0 = _time.time()
        logger.info(f"Processing file {i}/{n_files - 1}: {path}")

        logger.info('Making pipeline with variance %f', variance)
        decoder = make_pipeline(
            Vectorizer(),
            StandardScaler(),
            PCA(n_components=variance, random_state=42),
            LinearSVC(random_state=42),
        )

        accuracy_repeats = np.zeros((n_repeats, n_folds))
        confusion = None
        confusion_norm = None
        classes = None
        obs_scores = None
        perm_scores = None
        p_value = None

        for r in range(n_repeats):
            cv_seed = RANDOM_SEED + r
            cv_r = MinimumNaNSplit(
                n_splits=n_folds,
                n_repeats=1,
                random_state=cv_seed,
            )
            if r == 0:
                obs_scores, perm_scores, p_value = decode_permutation_scores(
                    X,
                    y,
                    cv_r,
                    decoder,
                    n_jobs=n_jobs,
                    n_permutations=n_perm,
                    scoring="balanced_accuracy",
                    random_state=RANDOM_SEED,
                )
                accuracy_repeats[r] = obs_scores

                y_pred = get_cv_predict(
                    X,
                    y,
                    cv_r,
                    decoder,
                    n_jobs=n_jobs,
                    random_state=RANDOM_SEED,
                )
                classes = np.unique(y)
                confusion = confusion_matrix(y, y_pred, labels=classes)
                confusion_norm = confusion_matrix(
                    y, y_pred, labels=classes, normalize="true"
                )
            else:
                accuracy_repeats[r] = decode_cv_scores(
                    X,
                    y,
                    cv_r,
                    decoder,
                    n_jobs=n_jobs,
                    scoring="balanced_accuracy",
                    random_state=cv_seed,
                )

        accuracy_stable = accuracy_repeats.mean()

        save_path = BIDSPath(
            root=os.path.join('results', f'{path.task}(roi)({ref})'),
            datatype='(decode)' + str(datatype),
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
            f.create_dataset(name="accuracy", data=obs_scores)
            f.create_dataset(name='perm_scores', data=perm_scores)
            f.create_dataset(name='p_value', data=p_value)
            f.create_dataset(name='accuracy_repeats', data=accuracy_repeats)
            f.create_dataset(name='accuracy_stable', data=accuracy_stable)
            f.create_dataset(name='confusion', data=confusion)
            f.create_dataset(name='confusion_norm', data=confusion_norm)
            f.create_dataset(name='classes', data=classes)

            f.attrs["fs"] = 128
            f.attrs["tmin"] = tmin
            f.attrs["tmax"] = tmax
            f.attrs["variance"] = variance
            f.attrs["n_perm"] = n_perm
            f.attrs["n_folds"] = n_folds
            f.attrs["n_repeats"] = n_repeats
            f.attrs["n_jobs"] = n_jobs
            f.attrs["cv_random_state"] = RANDOM_SEED

        logger.info(f"File {i} completed in {_time.time() - _t0:.2f}s")
        Xs[i] = None
        ys[i] = None
        del (
            X, y, decoder, obs_scores, perm_scores, p_value,
            accuracy_repeats, accuracy_stable, confusion, confusion_norm,
            classes, y_pred,
        )
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
                        help="Reference scheme")
    parser.add_argument("--description", type=str, default='Repeat',
                        choices=['Repeat', 'Passive', 'Decision'],
                        help="Repeat, Passive, or Decision")
    parser.add_argument("--phase", type=str, default='Stimulus',
                        choices=['Stimulus', 'Delay', 'Go', 'Response'],
                        help="Stimulus, Delay, Go, or Response")
    parser.add_argument("--band", type=str, default='highgamma',
                        help="highgamma or other band of neural signal")
    parser.add_argument("--datatype", type=str, default='phoneme',
                        choices=['phoneme', 'articulator', 'token', 'lexicality'],
                        help="what to classify? can be phoneme, articulator, token, or lexicality")
    parser.add_argument("--variance", type=float, default=0.85,
                        help="number of variance")
    parser.add_argument("--n_perm", type=int, default=2,
                        help="number of permutations")
    parser.add_argument("--n_folds", type=int, default=10,
                        help="number of folds")
    parser.add_argument("--n_repeats", type=int, default=1,
                        help="number of CV-seed repeats (only r=0 runs permutations)")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="number of jobs")

    args = parser.parse_args()
    main(**vars(args))
