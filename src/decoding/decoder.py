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
from sklearn.metrics import get_scorer, roc_auc_score
from sklearn.base import clone
from mne.decoding import Vectorizer
from joblib import Parallel, delayed, effective_n_jobs
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
    random_state: int = 42,
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

    def one_fold(fold_idx, train_idx, test_idx):
        # Create a fresh CrossDecoder per fold to avoid shared state
        dec = clone(decoder)
        X_train, X_test, y_train, y_test = sample_fold(
            X,
            y,
            train_idx,
            test_idx,
            seed=random_state + fold_idx,
        )
        
        dec.fit(
            X_train,
            y_train,
        )
        pred = dec.predict(X_test)
        return test_idx, pred

    # Run cross-validation with progress bar
    results = Parallel(n_jobs=n_jobs)(
        delayed(one_fold)(i, tr, te)
        for i, (tr, te) in enumerate(tqdm(splits, desc="CV folds"))
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
    random_state: int = 42,
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
        random_state=random_state,
    )
    
    return accuracy_score(y, y_pred)


def decode_cv_scores(
    X,
    y,
    cv,
    decoder,
    n_jobs: int = -1,
    scoring: str = "accuracy",
    random_state: int = 42,
):
    """Cross-validated observed scores without permutation testing.

    Returns per-fold scores for use in cheap outer-loop CV-seed repeats.
    """
    scorer = get_scorer(scoring)
    splits = list(cv.split(X, y))
    if len(splits) == 0:
        raise ValueError("CV splitter produced no splits")

    def one_fold(fold_idx, train_idx, test_idx):
        dec = clone(decoder)
        X_train, X_test, y_train, y_test = sample_fold(
            X,
            y,
            train_idx,
            test_idx,
            seed=random_state + fold_idx,
        )
        dec.fit(X_train, y_train)
        return scorer(dec, X_test, y_test)

    obs_scores = np.asarray(
        Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(one_fold)(i, tr, te)
            for i, (tr, te) in enumerate(splits)
        )
    )
    return obs_scores


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
    import time as _time
    
    scorer = get_scorer(scoring)
    splits = list(cv.split(X, y))
    if len(splits) == 0:
        raise ValueError("CV splitter produced no splits")
    
    # Observed
    obs_scores = []
    perm_scores = []
    for fold_idx, (tr, te) in enumerate(tqdm(splits, desc="Cross-validation")):
        fold_t0 = _time.time()
        dec = clone(decoder)
        X_train, X_test, y_train, y_test = sample_fold(
            X,
            y,
            tr,
            te,
            seed=random_state + fold_idx,
        )
        
        dec.fit(X_train, y_train)
        observed_score = scorer(dec, X_test, y_test)
        obs_scores.append(observed_score)
        
        rng_fold = np.random.RandomState(random_state)
        seeds_fold = rng_fold.randint(0, 2**31 - 1, size=n_permutations)

        # Clone the original unfitted decoder template, not the fitted dec.
        # This avoids serializing fitted model parameters to worker processes.
        def one_perm(seed):
            r = np.random.RandomState(seed)
            y_train_perm = y_train.copy()
            r.shuffle(y_train_perm)
            dec_p = clone(decoder)
            dec_p.fit(X_train, y_train_perm)
            return scorer(dec_p, X_test, y_test)
        
        # Use prefer="threads" to avoid expensive process forking.
        # SVC/LinearSVC release the GIL in their C/Fortran backends.
        perm_score = np.asarray(
            Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(one_perm)(s)
                for s in tqdm(seeds_fold, desc="Permutations")
            )
        )
        perm_scores.append(perm_score)
        logger.info(f"  Fold {fold_idx} done in {_time.time() - fold_t0:.2f}s")
        
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
    seed: int = 0,
):
    from ieeg.calc.oversample import mixup
    """Sample a fold of data for cross-validation.

    NaN fill uses a local RandomState so results are reproducible under
    multithreaded joblib and do not depend on the global np.random state.
    """
    # Avoid copying the entire X array - only copy the slices we need
    X_train = X[train_idx].copy()
    X_test = X[test_idx].copy()
    y_train = y[train_idx].copy()
    y_test = y[test_idx].copy()
    _fill_rng = np.random.RandomState(seed)
    
    unique_classes = np.unique(y_train)
    for cls in unique_classes:
        idx = (y_train == cls)
        # observer axis is the epoch axis
        x_cls = X_train[idx]
        # Fill NaN before mixup to avoid Floating Point Exception in C backend
        is_nan_cls = np.isnan(x_cls)
        if is_nan_cls.any():
            x_cls[is_nan_cls] = 0.0
        mixup(x_cls, obs_axis=0, rng=42)
        X_train[idx] = x_cls
    
    is_nan_train = np.isnan(X_train)
    if is_nan_train.any():
        X_train[is_nan_train] = _fill_rng.normal(0, 1, int(np.sum(is_nan_train)))
    
    is_nan_test = np.isnan(X_test)
    if is_nan_test.any():
        X_test[is_nan_test] = _fill_rng.normal(0, 1, int(np.sum(is_nan_test)))
    
    return X_train, X_test, y_train, y_test


def _decision_values(estimator, X):
    """Per-class continuous scores, shape ``(n_samples, n_classes)``.

    A binary ``decision_function`` yields a single column; it is mirrored into
    ``[-v, v]`` so callers can treat every case as one-vs-rest. AUC is unchanged
    when both the label and the sign of the score are flipped, so the two
    columns score identically and a macro average over them reproduces the
    plain binary AUC exactly.
    """
    if hasattr(estimator, "decision_function"):
        values = estimator.decision_function(X)
    else:
        values = estimator.predict_proba(X)
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = np.column_stack([-values, values])
    return values


def _group_label_table(y, groups):
    """Map each group to its single label, for group-level permutation."""
    uniq, inverse = np.unique(groups, return_inverse=True)
    group_label = np.empty(uniq.shape[0], dtype=y.dtype)
    for gi in range(uniq.shape[0]):
        members = np.unique(y[inverse == gi])
        if members.size != 1:
            raise ValueError(
                f"Group {uniq[gi]!r} carries {members.size} labels; group-level "
                "permutation is only defined when the label is a function of the group"
            )
        group_label[gi] = members[0]
    return inverse, group_label


def decode_permutation_auc_pooled(
    X,
    y,
    cv,
    classifier,
    transformer=None,
    groups=None,
    permute_groups: bool = True,
    n_jobs: int = -1,
    n_permutations: int = 5000,
    random_state: int = 42,
    batch_size: int = None,
):
    """Pooled out-of-fold AUC with a group-level permutation null.

    Differs from :func:`decode_permutation_scores` in three ways that matter for
    low-channel-count pools, where per-fold accuracy is too noisy to interpret:

    - The metric is AUC over the *pooled* out-of-fold decision values rather
      than a mean of per-fold scores, so every trial contributes to a single
      ranking instead of to one small, separately-thresholded estimate. With
      more than two classes this becomes a macro one-vs-rest average, which
      keeps chance at 0.5 whatever the class proportions are.
    - ``groups`` is forwarded to ``cv.split``, so a group-aware splitter such as
      ``StratifiedGroupKFold`` can keep a stimulus out of both train and test.
    - With ``permute_groups=True`` the null permutes labels at the group level,
      which is the right exchangeability unit when a label is a deterministic
      function of the stimulus. Trial-level shuffling breaks the tie between a
      stimulus's repeats and yields an over-narrow null.

    Decision values are standardised column-wise within each fold before
    pooling. That is monotone within a fold and within a class, so it cannot
    manufacture or destroy within-fold ranking, but it stops a fold with an
    offset decision function from dominating the pooled ordering.

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_times)
    y : ndarray, shape (n_epochs,)
        Target with two or more classes.
    cv : CV splitter
        If ``groups`` is given, must accept ``split(X, y, groups)``.
    classifier : sklearn estimator
        Refit on every permutation. Must expose ``decision_function`` or
        ``predict_proba``.
    transformer : sklearn transformer, optional
        Unsupervised feature preparation, fit on each training split *once*
        with ``fit(X_train)`` and reused across permutations. This is exact
        rather than an approximation only because the transformer never sees
        the labels, so permuting them cannot change what it would have learnt.
        Passing supervised preprocessing here would leak.
    groups : array-like, shape (n_epochs,), optional
        Grouping vector, e.g. the stimulus word.
    permute_groups : bool, default=True
        Permute labels group-wise. Requires ``groups``.
    n_jobs : int, default=-1
    n_permutations : int, default=5000
    random_state : int, default=42
    batch_size : int, optional
        Permutations evaluated per parallel task. Individual permutations are
        far too short to amortise handing the fold arrays to a worker, so they
        are dispatched in batches. Defaults to roughly four batches per worker.

    Returns
    -------
    obs_auc : float
    perm_aucs : ndarray, shape (n_permutations,)
        Null AUCs. Degenerate permutations (a training split that is missing a
        class) are returned as NaN and excluded from the p-value.
    p_value : float
        One-sided, ``(#{null >= observed} + 1) / (n_valid + 1)``.
    """
    y = np.asarray(y)
    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError(f"Pooled AUC needs at least two classes, got {classes.size}")
    n_classes = classes.size

    groups_arr = None if groups is None else np.asarray(groups)
    if permute_groups and groups_arr is None:
        raise ValueError("permute_groups=True requires a groups vector")

    if groups_arr is None:
        splits = list(cv.split(X, y))
    else:
        splits = list(cv.split(X, y, groups_arr))
    if len(splits) == 0:
        raise ValueError("CV splitter produced no splits")

    # sample_fold's NaN handling is label-independent: every training trial
    # belongs to some class, so the per-class loop zero-fills all training NaNs
    # before the global fill can see them, and the test fill is driven by a
    # seeded RNG alone. The imputed fold arrays are therefore identical under
    # any relabelling and are built once here instead of per permutation.
    # tests/test_decoding_lda_resolved.py pins that invariant.
    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test, _, _ = sample_fold(
            X, y, train_idx, test_idx, seed=random_state + fold_idx
        )
        if transformer is not None:
            prep = clone(transformer)
            X_train = prep.fit_transform(X_train)
            X_test = prep.transform(X_test)
        folds.append((train_idx, test_idx, X_train, X_test))

    def pooled_auc(labels):
        oof = np.empty((labels.shape[0], n_classes), dtype=float)
        for train_idx, test_idx, X_train, X_test in folds:
            y_train = labels[train_idx]
            # A class absent from a training split leaves the estimator with
            # fewer score columns than `classes`, so pooling them would mix up
            # which column belongs to which class. Drop the whole permutation.
            if np.unique(y_train).size < n_classes:
                return np.nan
            dec = clone(classifier)
            dec.fit(X_train, y_train)
            values = _decision_values(dec, X_test)
            spread = values.std(axis=0)
            usable = spread > 0
            oof[test_idx] = np.where(
                usable, (values - values.mean(axis=0)) / np.where(usable, spread, 1.0), 0.0
            )
        return float(np.mean([
            roc_auc_score(labels == cls, oof[:, index])
            for index, cls in enumerate(classes)
        ]))

    obs_auc = pooled_auc(y)
    if not np.isfinite(obs_auc):
        raise ValueError("Observed fit degenerated; check the CV splits")

    if permute_groups:
        inverse, group_label = _group_label_table(y, groups_arr)

        def draw(rng):
            permuted = group_label.copy()
            rng.shuffle(permuted)
            return permuted[inverse]
    else:
        def draw(rng):
            permuted = y.copy()
            rng.shuffle(permuted)
            return permuted

    seeds = np.random.RandomState(random_state).randint(
        0, 2**31 - 1, size=n_permutations
    )

    if batch_size is None:
        n_workers = effective_n_jobs(n_jobs)
        batch_size = max(1, int(np.ceil(n_permutations / (n_workers * 4))))
    batches = [
        seeds[i:i + batch_size] for i in range(0, n_permutations, batch_size)
    ]

    def one_batch(batch):
        return [pooled_auc(draw(np.random.RandomState(s))) for s in batch]

    results = Parallel(n_jobs=n_jobs)(
        delayed(one_batch)(b) for b in tqdm(batches, desc="Permutations", leave=False)
    )
    perm_aucs = np.asarray([auc for batch in results for auc in batch])

    valid = np.isfinite(perm_aucs)
    n_valid = int(valid.sum())
    if n_valid < n_permutations:
        logger.warning(
            "%d/%d permutations degenerated and were dropped",
            n_permutations - n_valid,
            n_permutations,
        )
    if n_valid == 0:
        raise ValueError("All permutations degenerated; check the CV splits")
    p_value = (np.sum(perm_aucs[valid] >= obs_auc) + 1.0) / (n_valid + 1.0)

    return obs_auc, perm_aucs, float(p_value)


def decode_cross_permutation_auc_pooled(
    X_src,
    X_tgt,
    y,
    cv,
    classifier,
    transformer=None,
    groups=None,
    permute_groups: bool = True,
    tgt_slices=None,
    n_jobs: int = -1,
    n_permutations: int = 5000,
    random_state: int = 42,
    batch_size: int = None,
):
    """Cross-condition pooled OOF AUC, scoring one source window on many targets.

    Same metric, grouping, and word-level null as
    :func:`decode_permutation_auc_pooled`, but the classifier is fit on
    ``X_src`` (already cropped to one train window) and scored on slices of
    ``X_tgt``. Fold indices are shared, so a held-out word is held out of both
    domains.

    The transformer is fit once per training split on the source window and
    reused across target times and permutations. That is exact only because
    the transformer is unsupervised.

    Parameters
    ----------
    X_src : ndarray, shape (n_epochs, n_channels, n_times_window)
        Source-domain train window.
    X_tgt : ndarray, shape (n_epochs, n_channels, n_times)
        Target-domain series. When ``tgt_slices`` is omitted the whole array
        is one test window and must have the same duration as ``X_src``.
    y : ndarray, shape (n_epochs,)
        Shared labels after trial pairing.
    cv, classifier, transformer, groups, permute_groups, n_jobs,
    n_permutations, random_state, batch_size
        As in :func:`decode_permutation_auc_pooled`.
    tgt_slices : sequence of (start, end), optional
        Half-open sample slices into the last axis of ``X_tgt``.

    Returns
    -------
    obs_aucs : ndarray, shape (n_test,)
    perm_aucs : ndarray, shape (n_permutations, n_test)
    p_values : ndarray, shape (n_test,)
        Uncorrected one-sided permutation p-values. Degenerate null draws
        are excluded per cell.
    """
    y = np.asarray(y)
    X_src = np.asarray(X_src)
    X_tgt = np.asarray(X_tgt)
    if X_src.shape[0] != X_tgt.shape[0] or y.shape[0] != X_src.shape[0]:
        raise ValueError(
            f"X_src, X_tgt, and y must share n_epochs; got "
            f"{X_src.shape[0]}, {X_tgt.shape[0]}, {y.shape[0]}"
        )
    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError(f"Pooled AUC needs at least two classes, got {classes.size}")
    n_classes = classes.size

    if tgt_slices is None:
        tgt_slices = ((0, X_tgt.shape[-1]),)
    tgt_slices = tuple((int(start), int(end)) for start, end in tgt_slices)
    n_test = len(tgt_slices)

    groups_arr = None if groups is None else np.asarray(groups)
    if permute_groups and groups_arr is None:
        raise ValueError("permute_groups=True requires a groups vector")

    if groups_arr is None:
        splits = list(cv.split(X_src, y))
    else:
        splits = list(cv.split(X_src, y, groups_arr))
    if len(splits) == 0:
        raise ValueError("CV splitter produced no splits")

    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_src_train, _, _, _ = sample_fold(
            X_src, y, train_idx, test_idx, seed=random_state + fold_idx
        )
        _, X_tgt_test, _, _ = sample_fold(
            X_tgt, y, train_idx, test_idx, seed=random_state + fold_idx
        )
        if transformer is not None:
            prep = clone(transformer)
            X_src_train = prep.fit_transform(X_src_train)
            tgt_tests = [
                prep.transform(X_tgt_test[..., start:end])
                for start, end in tgt_slices
            ]
        else:
            tgt_tests = [
                X_tgt_test[..., start:end] for start, end in tgt_slices
            ]
        folds.append((train_idx, test_idx, X_src_train, tgt_tests))

    def pooled_auc_map(labels):
        oof = np.empty((n_test, labels.shape[0], n_classes), dtype=float)
        for train_idx, test_idx, X_src_train, tgt_tests in folds:
            y_train = labels[train_idx]
            if np.unique(y_train).size < n_classes:
                return np.full(n_test, np.nan)
            dec = clone(classifier)
            dec.fit(X_src_train, y_train)
            for test_i, X_tgt_fold in enumerate(tgt_tests):
                values = _decision_values(dec, X_tgt_fold)
                spread = values.std(axis=0)
                usable = spread > 0
                oof[test_i, test_idx] = np.where(
                    usable,
                    (values - values.mean(axis=0)) / np.where(usable, spread, 1.0),
                    0.0,
                )
        return np.asarray([
            float(np.mean([
                roc_auc_score(labels == cls, oof[test_i, :, index])
                for index, cls in enumerate(classes)
            ]))
            for test_i in range(n_test)
        ])

    obs_aucs = pooled_auc_map(y)
    if not np.isfinite(obs_aucs).all():
        raise ValueError("Observed fit degenerated; check the CV splits")

    if permute_groups:
        inverse, group_label = _group_label_table(y, groups_arr)

        def draw(rng):
            permuted = group_label.copy()
            rng.shuffle(permuted)
            return permuted[inverse]
    else:
        def draw(rng):
            permuted = y.copy()
            rng.shuffle(permuted)
            return permuted

    seeds = np.random.RandomState(random_state).randint(
        0, 2**31 - 1, size=n_permutations
    )

    if batch_size is None:
        n_workers = effective_n_jobs(n_jobs)
        batch_size = max(1, int(np.ceil(n_permutations / (n_workers * 4))))
    batches = [
        seeds[i:i + batch_size] for i in range(0, n_permutations, batch_size)
    ]

    def one_batch(batch):
        return [pooled_auc_map(draw(np.random.RandomState(s))) for s in batch]

    results = Parallel(n_jobs=n_jobs)(
        delayed(one_batch)(b) for b in tqdm(batches, desc="Permutations", leave=False)
    )
    perm_aucs = np.asarray([auc for batch in results for auc in batch])
    if perm_aucs.ndim == 1:
        perm_aucs = perm_aucs.reshape(n_permutations, n_test)

    p_values = np.empty(n_test, dtype=float)
    for test_i in range(n_test):
        column = perm_aucs[:, test_i]
        valid = np.isfinite(column)
        n_valid = int(valid.sum())
        if n_valid < n_permutations:
            logger.warning(
                "test window %d: %d/%d permutations degenerated and were dropped",
                test_i,
                n_permutations - n_valid,
                n_permutations,
            )
        if n_valid == 0:
            raise ValueError("All permutations degenerated; check the CV splits")
        p_values[test_i] = (
            np.sum(column[valid] >= obs_aucs[test_i]) + 1.0
        ) / (n_valid + 1.0)

    return obs_aucs, perm_aucs, p_values


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
            seed=random_state + fold_idx,
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
