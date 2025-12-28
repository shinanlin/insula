"""ROI decoding with cross-validated permutation testing.

This module performs single-ROI neural decoding using scikit-learn pipelines.
It provides utilities to:

- Load BIDS-organized HDF5 data for a given ROI and crop a temporal window
  via ``load_roi_data`` (returns X: epochs×channels×times, y: labels).
- Train/evaluate a classifier pipeline with cross-validation and obtain
  out-of-fold predictions and scores via ``get_cv_predict`` and ``get_cv_score``.
- Compute permutation-based null distributions and p-values across CV folds via
  ``decode_permutation_scores``.
- Prepare train/test splits per fold with basic data hygiene and optional
  class-wise mixup augmentation via ``sample_fold``.

The default example pipeline uses ``Vectorizer -> StandardScaler -> PCA -> SVC``.
For CV splitting, ``ieeg.calc.oversample.MinimumNaNSplit`` can be used to avoid
NaN-heavy folds.

Typical workflow:
1. Load neural data for one ROI and a desired temporal window (``tmin``, ``tmax``).
2. Build an sklearn estimator pipeline (e.g., PCA+SVC) over vectorized epochs.
3. Choose a CV splitter (e.g., ``MinimumNaNSplit`` or ``StratifiedKFold``).
4. Get cross-validated accuracy and run permutation testing for significance.

Example:
    >>> from mne.decoding import Vectorizer
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.svm import SVC
    >>> from ieeg.calc.oversample import MinimumNaNSplit
    >>> 
    >>> # Load data from one brain region (ROI) and crop to [-0.5, 0.5] s
    >>> X, y = load_roi_data(bids_root, 'STGl', 'perception', 'highgamma', 'phoneme',
    ...                      tmin=-0.5, tmax=0.5)
    >>> 
    >>> # Build estimator pipeline
    >>> estimator = make_pipeline(Vectorizer(), StandardScaler(), PCA(0.85), SVC(kernel='linear'))
    >>> 
    >>> # Cross-validation splitter
    >>> cv = MinimumNaNSplit(n_splits=3, n_repeats=1)
    >>> 
    >>> # Permutation testing across folds
    >>> obs_scores, perm_scores, p_value = decode_permutation_scores(
    ...     X, y, cv, estimator, n_jobs=-1, n_permutations=100
    ... )
    >>> 
    >>> # Aggregate cross-validated score
    >>> cv_score = get_cv_score(X, y, cv, estimator, n_jobs=-1)
"""

import rootutils
path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import argparse
from pathlib import Path
import h5py
import numpy as np
from mne_bids import BIDSPath
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import get_scorer
from sklearn.base import clone
from mne.decoding import Vectorizer
from joblib import Parallel, delayed
import logging
import sys
from sklearn.base import BaseEstimator, ClassifierMixin
from ieeg.calc.oversample import MinimumNaNSplit
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def load_roi_data(
    bids_root, 
    roi, 
    description, 
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
        root=bids_root, datatype=datatype, description=description,
        suffix=band, extension='.h5', check=False
    )
    roi_path = root.copy().update(subject=roi)
    roi_files = roi_path.match()
    
    if not roi_files:
        raise FileNotFoundError(f"No files found for ROI {roi}")
        
    # Load neural data and metadata from HDF5 file
    with h5py.File(roi_files[0], 'r') as data:
        X = data['X'][()]  # Neural time series: (epochs, channels, times)
        y = data['y'][()]  # Class labels: (epochs,)
        fs = data.attrs['fs']  # Sampling frequency
        
    data.close()
    
    # Crop temporal window to tmin to tmax seconds relative to stimulus onset
    # Assumes original data spans -1 to +1.5 seconds, so we take (tmin, tmax)
    t_start = -1.0          # data starts at -1 s
    t_end = 1.5
    start_idx = int(fs * (tmin - t_start))  # Start at tmin seconds (originally -0.5s relative)
    end_idx = int(fs * (tmax - t_start))    # End at tmax seconds (originally +0.5s relative)
    X = X[:, :, start_idx:end_idx]
    
    return X, y
    

def get_cv_predict(
    X,
    y,
    cv,
    decoder,
    n_jobs: int = -1,
    predict_method: str = "predict",
):
    """
    Cross-domain OOF-style predictions on XB: fit on A_train, predict on B_test.

    Parameters are analogous to cross_domain_cv_score, with:
    - predict_method: 'predict' | 'predict_proba' | 'decision_function'

    Returns
    - y_pred: array aligned to y (index-wise), containing predictions for XB at test folds.
              For 'predict', shape (n_samples,);
              For 'predict_proba' or 'decision_function', shape (n_samples, n_outputs).
    """
    # Only 'predict' is currently supported by CrossDecoder
    if predict_method != "predict":
        raise ValueError("Only 'predict' is supported by CrossDecoder in CV mode.")

    splits = list(cv.split(X, y))
    if len(splits) == 0:
        raise ValueError("CV splitter produced no splits")

    # Pre-allocate output (labels)
    y_pred = np.empty_like(y)

    def one_fold(train_idx, test_idx):
        # Create a fresh CrossDecoder per fold to avoid shared state
        dec = clone(decoder)
        X_train, X_test, y_train, y_test = sample_fold(
            X,
            y,
            train_idx,
            test_idx,
        )
        
        dec.fit(
            X_train,
            y_train,
        )
        pred = dec.predict(X_test)
        return test_idx, pred

    # Run cross-validation with progress bar
    results = Parallel(n_jobs=n_jobs)(
        delayed(one_fold)(tr, te) for tr, te in tqdm(splits, desc="CV folds")
    )

    for te, pred in results:
        y_pred[te] = pred

    return y_pred

def get_cv_score(
    X,
    y,
    cv,
    decoder,
    n_jobs: int = -1,
    predict_method: str = "predict",
):
    """Cross-domain OOF-style predictions on XB: fit on A_train, predict on B_test."""
    
    from sklearn.metrics import accuracy_score
    
    y_pred = get_cv_predict(
        X,
        y,
        cv,
        decoder,
        n_jobs=n_jobs,
        predict_method=predict_method,
    )
    
    return accuracy_score(y, y_pred)


def decode_permutation_scores(
    X,
    y,
    cv,
    decoder,
    n_jobs: int = -1,
    n_permutations: int = 10,
    scoring: str = "accuracy",
    random_state: int = 42,
):
    
    scorer = get_scorer(scoring)
    splits = list(cv.split(X, y))
    if len(splits) == 0:
        raise ValueError("CV splitter produced no splits")
    
    # Observed
    obs_scores = []
    perm_scores = []
    for tr, te in tqdm(splits, desc="Cross-validation"):
        dec = clone(decoder)
        X_train, X_test, y_train, y_test = sample_fold(
            X,
            y,
            tr,
            te,
        )
        
        dec.fit(X_train, y_train)
        observed_score = scorer(dec, X_test, y_test)
        obs_scores.append(observed_score)
        
        rng_fold = np.random.RandomState(random_state)
        seeds_fold = rng_fold.randint(0, 2**31 - 1, size=n_permutations)

        def one_perm(seed):
            r = np.random.RandomState(seed)
            y_train_perm = y_train.copy()
            r.shuffle(y_train_perm)
            dec_p = clone(dec)
            dec_p.fit(X_train, y_train_perm)
            return scorer(dec_p, X_test, y_test)
        
        perm_score = np.asarray(Parallel(n_jobs=n_jobs)(delayed(one_perm)(s) for s in tqdm(seeds_fold, desc="Permutations")))
        perm_scores.append(perm_score)
        
    score = np.mean(obs_scores)
    perm_scores = np.stack(perm_scores)
    
    # p-value (greater is better metric)
    p_value = (np.sum(perm_scores.mean(axis=0) >= score) + 1.0) / (n_permutations + 1.0)

    return obs_scores, perm_scores, p_value


def sample_fold(
    X,
    y,
    train_idx,
    test_idx,
):
    from ieeg.calc.oversample import mixup
    """Sample a fold of data for cross-validation."""
    X_, y_ = X.copy(), y.copy()
    X_train, X_test = X_[train_idx], X_[test_idx]
    y_train, y_test = y_[train_idx], y_[test_idx]
    
    unique_classes = np.unique(y_train)
    for cls in unique_classes:
        idx = (y_train == cls)
        # observer axis is the epoch axis
        x_cls = X_train[idx]
        mixup(x_cls, obs_axis=0, rng=42)
        X_train[idx] = x_cls
    
    is_nan_test = np.isnan(X_test)
    if is_nan_test.any():
        X_test[is_nan_test] = np.random.normal(0, 1, int(np.sum(is_nan_test)))
    
    return X_train, X_test, y_train, y_test

def generalized_permutation_scores(
    X,
    y,
    cv,
    decoder,
    scoring: str = "accuracy",
    n_permutations: int = 10,
    n_jobs: int = -1,
    random_state: int = 42,
    window: float = 0.2,
    step: float = 0.1,
    fs: int = 128,
    tmin: float = 0,
    tmax: float = 0.5,
):
    """Temporal generalization (train-time × test-time) decoding with permutations and FDR.

    For each CV fold, this function:
    - Fits a fresh ``decoder`` (sklearn pipeline) on the training split.
    - Constructs two sliding-window grids: a train-time grid within [train_tmin, train_tmax] and
      a test-time grid within [test_tmin, test_tmax], each with window length ``window`` and step ``step``.
    - For every cell (t_train, t_test), trains the estimator on X (training split) using the
      train window and evaluates on X (test split) using the test window.
    - Builds a permutation baseline at each cell by shuffling y in the training split ``n_permutations`` times.

    After all folds finish, scores are averaged across folds to obtain a 2D observed map and a corresponding
    permutation distribution per cell. One-sided per-cell p-values are computed from the permutation null and
    then corrected across the 2D field using FDR-BH.

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_times)
        ROI data.
    y : ndarray, shape (n_epochs,)
        Class labels for X. Upstream balancing should align label distributions.
    cv : CV splitter
        Any sklearn-compatible splitter yielding (train_idx, test_idx) on (X, y).
    decoder : sklearn Pipeline
        Configured sklearn pipeline.
    scoring : str, default='accuracy'
        Scorer name accepted by sklearn's ``get_scorer``.
    n_permutations : int, default=10
        Number of label permutations per fold per (t_train, t_test) cell.
    n_jobs : int, default=-1
        Parallel jobs for permutation evaluations within each cell.
    random_state : int, default=42
        Base seed for reproducibility (expanded per fold/cell internally as needed).
    window : float, default=0.2
        Window length in seconds for temporal slicing.
    step : float, default=0.1
        Step size in seconds between adjacent windows.
    fs : int, default=128
        Sampling rate to convert seconds to sample indices.
    tmin, tmax : float, default=(0, 0.5)
        Temporal range (seconds) for train-time windows (applied to X on the training split).

    Returns
    -------
    obs_scores : ndarray, shape (T_train, T_test, n_folds)
        Observed score (``scoring``) per (train-time, test-time) cell and per fold.
    perm_scores : ndarray, shape (T_train, T_test, n_permutations, n_folds)
        Permutation baseline scores per cell, permutation, and fold.
    pvals_fdr : ndarray, shape (T_train, T_test)
        FDR-BH corrected per-cell p-values (one-sided; greater-is-better).

    Notes
    -----
    - Indexing: indices are derived with rounding to samples (``round(seconds * fs)``) and slicing
      uses half-open ranges [start:end], ensuring a constant window length of ``window * fs`` samples.
    - Parallelization occurs at the permutation level inside each cell; consider adjusting the
      granularity (e.g., over folds) if overhead becomes significant for large grids.
    - To implement FWER control, you can add a 2D max-field correction or a 2D cluster-based
      permutation procedure on the fold-averaged statistic map.
    """
    from statsmodels.stats.multitest import multipletests
    
    scorer = get_scorer(scoring)
    splits = list(cv.split(X, y))
    if len(splits) == 0:
        raise ValueError("CV splitter produced no splits")
    
    # format to .2f 
    tmin, tmax = [round(t, 2) for t in [tmin, tmax]]
    
    train_time_points = np.arange(tmin + window,
                            tmax,
                            step)
    test_time_points = train_time_points
    window_samples = int(window * fs)
    step_samples = int(step * fs)
    
    # Observed
    n_folds = len(splits)
    obs_scores = np.empty((len(train_time_points), len(test_time_points), n_folds))
    perm_scores = np.empty((len(train_time_points), len(test_time_points), n_permutations, n_folds))
    
    for fold_idx, (tr, te) in enumerate(tqdm(splits, desc="Cross-validation")):
        
        dec = clone(decoder)
        
        X_train, X_test, y_train, y_test = sample_fold(
            X,
            y,
            tr,
            te,
        )

        # time resolved decoding
        for train_t_idx, train_time_end in enumerate(train_time_points):
            
            end_train = int(round((train_time_end - tmin) * fs))
            start_train = end_train - window_samples
            
            if start_train < 0 or end_train > X_train.shape[-1]:
                logger.warning(f"Window out of bounds for time {train_time_end:.3f}s, skipping")
                continue
            
            x_train_s = X_train[..., start_train:end_train]
            dec.fit(x_train_s, y_train)
            
            for test_t_idx, test_time_end in enumerate(test_time_points):
                
                end_test = int(round((test_time_end - tmin) * fs))
                start_test = end_test - window_samples
            
                if start_test < 0 or end_test > X_test.shape[-1]:
                    logger.warning(f"Window out of bounds for time {test_time_end:.3f}s, skipping")
                    continue
                
                x_test_s = X_test[..., start_test:end_test]
                    
                pred = dec.predict(x_test_s)
                observed_score = scorer(dec, x_test_s, y_test)
                
                rng_fold = np.random.RandomState(random_state)
                seeds_fold = rng_fold.randint(0, 2**31 - 1, size=n_permutations)

                def one_perm(seed):
                    r = np.random.RandomState(seed)
                    y_train_perm = y_train.copy()
                    r.shuffle(y_train_perm)
                    dec.fit(x_train_s, y_train_perm)
                    return scorer(dec, x_test_s, y_test)

                perm_score = np.asarray(Parallel(n_jobs=n_jobs, batch_size=10)(delayed(one_perm)(s) for s in seeds_fold))
                
                obs_scores[train_t_idx, test_t_idx, fold_idx] = observed_score
                perm_scores[train_t_idx, test_t_idx, :, fold_idx] = perm_score
            
    observed_mean = obs_scores.mean(axis=-1)        # (Ttr, Tte)
    perm_mean = perm_scores.mean(axis=-1)           # (Ttr, Tte, n_perm)
    
    P = n_permutations
    pvals_pt = ( (perm_mean >= observed_mean[..., None]).sum(axis=2) + 1 ) / (P + 1)

    p_flat = pvals_pt.ravel()
    _, pvals_corrected, _, _ = multipletests(p_flat, alpha=0.05, method='fdr_bh')
    pvals_corrected = pvals_corrected.reshape(pvals_pt.shape)

    return obs_scores, perm_scores, pvals_corrected

def main(
    bids_root,
    train_roi,
    test_roi,
    description,
    band,
    datatype,
    variance,
    n_permutations,
    n_jobs,
    n_folds,
):
    # Create estimator pipeline
    estimator = make_pipeline(
        Vectorizer(),
        StandardScaler(),
        PCA(n_components=variance, random_state=42),
        SVC(kernel='linear', random_state=42)
    )
    
    # load X and y
    X, y = load_roi_data(bids_root, train_roi, 'perception', band, datatype, tmin=-0.5, tmax=0.5)
    
    msn = MinimumNaNSplit(n_splits=n_folds, n_repeats=1)
    
    obs_scores, perm_scores, p_value = decode_permutation_scores(
        X = X,
        y = y,
        cv = msn,
        decoder = estimator,
        n_jobs=n_jobs,
        n_permutations=n_permutations,
    )
    
    cv_score = get_cv_score(X, y, msn, estimator, n_jobs=n_jobs)
    
    print('obs_scores:', np.mean(obs_scores))
    print('perm_scores:', np.mean(perm_scores))
    print('p_value:', p_value)
    
    
    return
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", type=str,
                        default="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/derivatives/decoding(ROI)")
    parser.add_argument("--train_roi", type=str, default="STGl")
    parser.add_argument("--test_roi", type=str, default="PrGl")
    parser.add_argument("--description", type=str, default='production')
    parser.add_argument("--band", type=str, default='highgamma')
    parser.add_argument("--datatype", type=str, default='phoneme')
    parser.add_argument("--variance", type=float, default=0.85)
    parser.add_argument("--n_permutations", type=int, default=4)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--n_folds", type=int, default=3)
    args = parser.parse_args()
    main(**vars(args))
