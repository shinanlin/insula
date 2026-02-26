"""Decoding with optional pattern extraction via cross-validated permutation testing.

Extends the ``decode_permutation_scores`` interface from ``deocder.py`` with an
optional ``use_pattern`` flag.  When ``use_pattern=False`` the behaviour is
identical to the original function (returns accuracy-based obs_scores,
perm_scores, p_value).  When ``use_pattern=True`` the function additionally
extracts spatial patterns (via ``mne.decoding.get_coef``) on the same CV folds
and permutations, performs cluster-corrected significance testing on the
patterns, and returns both accuracy *and* pattern results.

Requires the pipeline to wrap a linear model with ``LinearModel`` so that
``get_coef(..., 'patterns_', inverse_transform=True)`` works.
"""

import gc

import numpy as np
from sklearn.metrics import get_scorer
from sklearn.base import clone
from joblib import Parallel, delayed
from mne.decoding import get_coef
from tqdm import tqdm
from ieeg.calc.stats import time_cluster
from ieeg.calc.fast import mixup
import logging

logger = logging.getLogger(__name__)


def feature_mixup(x_cls, alpha=1.0, rng=None):
    """Per-feature NaN interpolation using mixup.

    For each feature position (channel, time), NaN values across samples are
    replaced by a convex combination of two randomly chosen non-NaN values at
    that same feature position.  This is much more suitable than row-level
    mixup when every sample has *some* NaN but no single feature is entirely
    NaN across all samples.

    Parameters
    ----------
    x_cls : ndarray, shape (n_samples, ...)
        Data for one class.  First axis is the sample/epoch axis.
    alpha : float
        Beta distribution parameter (default 1.0 = uniform on [0, 1]).
    rng : int | np.random.RandomState | None
        Random state for reproducibility.

    Returns
    -------
    None – ``x_cls`` is modified **in-place**.
    """
    if rng is None:
        rng = np.random.RandomState()
    elif isinstance(rng, (int, np.integer)):
        rng = np.random.RandomState(rng)

    n_samples = x_cls.shape[0]
    # flatten features: (n_samples, n_features)
    x_2d = x_cls.reshape(n_samples, -1)
    n_features = x_2d.shape[1]

    nan_mask = np.isnan(x_2d)
    if not nan_mask.any():
        return

    for f in range(n_features):
        col = x_2d[:, f]
        nan_idx = np.where(nan_mask[:, f])[0]
        if len(nan_idx) == 0:
            continue
        valid_idx = np.where(~nan_mask[:, f])[0]

        if len(valid_idx) == 0:
            # entire feature is NaN across all samples – fill with noise
            x_2d[nan_idx, f] = rng.normal(0, 1, len(nan_idx))
            continue
        if len(valid_idx) == 1:
            # only one valid value – copy it
            x_2d[nan_idx, f] = col[valid_idx[0]]
            continue

        # mixup: convex combination of two random non-NaN values
        n_nan = len(nan_idx)
        idx1 = rng.choice(valid_idx, size=n_nan, replace=True)
        idx2 = rng.choice(valid_idx, size=n_nan, replace=True)
        lam = rng.beta(alpha, alpha, size=n_nan)
        lam = np.maximum(lam, 1.0 - lam)  # ensure coefficient >= 0.5
        x_2d[nan_idx, f] = lam * col[idx1] + (1.0 - lam) * col[idx2]


# reuse sample_fold from the existing decoder module
def sample_fold(
    X,
    y,
    train_idx,
    test_idx,
):
    """Sample a fold of data for cross-validation."""
    X_train, X_test = X[train_idx].copy(), X[test_idx].copy()
    y_train, y_test = y[train_idx].copy(), y[test_idx].copy()

    unique_classes = np.unique(y_train)
    for cls in unique_classes:
        idx = (y_train == cls)
        x_cls = X_train[idx]
        feature_mixup(x_cls, alpha=1.0, rng=42)
        X_train[idx] = x_cls

    # fill remaining test NaN with noise
    is_nan_test = np.isnan(X_test)
    if is_nan_test.any():
        X_test[is_nan_test] = np.random.normal(0, 1, int(np.sum(is_nan_test)))

    return X_train, X_test, y_train, y_test

# ---------------------------------------------------------------------------
# Cluster correction (ported from decoding_weights.py)
# ---------------------------------------------------------------------------

def cluster_correction(scores, baseline, p_thresh=0.05, tails=2, ignore=None):
    """Cluster correction with permutation-derived threshold masks.

    Parameters
    ----------
    scores : ndarray, shape (..., T)
        Observed scores (e.g. mean pattern across folds).
    baseline : ndarray, shape (n_perm, ..., T)
        Permutation scores.
    p_thresh : float
        Significance threshold.
    tails : {1, 2}
        One- or two-tailed test.
    ignore : None | int | tuple[int]
        Axes to ignore when clustering (passed to ``time_cluster``).

    Returns
    -------
    mask : ndarray[bool], same shape as *scores*
        Cluster-corrected significance mask.
    p_act : ndarray[float], same shape as *scores*
        Pointwise p-values.
    """
    scores = np.asarray(scores)
    baseline = np.asarray(baseline)
    if baseline.ndim < 2:
        raise ValueError("baseline must be (n_perm, ..., time)")
    n_perm = baseline.shape[0]

    mu = baseline.mean(axis=0)

    if tails == 2:
        obs_dev = np.abs(scores - mu)
        perm_dev = np.abs(baseline - mu)
    else:
        sign = np.sign(scores - mu)
        obs_dev = (scores - mu) * sign
        perm_dev = (baseline - mu) * sign

    # pointwise p-values
    p_act = (np.sum(perm_dev >= obs_dev, axis=0) + 1.0) / (n_perm + 1.0)

    # perm p-values via ranking
    order = np.argsort(perm_dev, axis=0)
    ranks = np.empty_like(order, dtype=np.int64)
    np.put_along_axis(
        ranks,
        order,
        np.arange(n_perm, dtype=np.int64)[:, *((None,) * (perm_dev.ndim - 1))],
        axis=0,
    )
    one_minus_p_perm = (n_perm - ranks + 1) / (n_perm + 1)

    b_act = (1.0 - p_act) >= (1.0 - p_thresh)
    b_perm = one_minus_p_perm >= (1.0 - p_thresh)

    mask = time_cluster(b_act, b_perm, p_val=p_thresh, tails=tails, ignore=ignore)
    return mask, p_act


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def decode_permutation_scores(
    X,
    y,
    cv,
    decoder,
    n_jobs: int = -1,
    n_permutations: int = 10,
    scoring: str = "accuracy",
    random_state: int = 42,
    use_pattern: bool = False,
):
    """Cross-validated permutation decoding with optional pattern extraction.

    When ``use_pattern=False`` (default), this function behaves identically to
    ``deocder.decode_permutation_scores``: it returns observed accuracy scores,
    permutation accuracy scores, and a p-value.

    When ``use_pattern=True``, for each fold and each permutation, spatial
    patterns are also extracted via ``get_coef(pipeline, 'patterns_',
    inverse_transform=True)``.  The mean observed pattern is compared against
    the permutation distribution using cluster-corrected significance testing.

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_times)
        Neural time series data.
    y : ndarray, shape (n_epochs,)
        Class labels.
    cv : cross-validation splitter
        E.g. ``MinimumNaNSplit`` or ``StratifiedKFold``.
    decoder : sklearn estimator / pipeline
        The classification pipeline.  When ``use_pattern=True``, the pipeline
        must include ``LinearModel`` so that ``get_coef`` can extract patterns.
    n_jobs : int
        Number of parallel jobs (default ``-1`` = all cores).
    n_permutations : int
        Number of label permutations per fold (default 10).
    scoring : str
        Scoring metric name understood by ``sklearn.metrics.get_scorer``
        (default ``"accuracy"``).
    random_state : int
        Seed for reproducibility (default 42).
    use_pattern : bool
        If ``True``, also compute and return pattern-based results
        (default ``False``).

    Returns
    -------
    If ``use_pattern=False``:
        obs_scores : list[float]
            Observed score per CV fold.
        perm_scores : ndarray, shape (n_folds, n_permutations)
            Permutation scores per fold.
        p_value : float
            Proportion of mean-permutation scores >= mean observed score.

    If ``use_pattern=True``:
        obs_scores : list[float]
            Observed accuracy score per CV fold.
        perm_scores : ndarray, shape (n_folds, n_permutations)
            Permutation accuracy scores per fold.
        p_value : float
            Accuracy-based p-value.
        pattern_obs_scores : ndarray, shape (n_folds, n_channels, n_times)
            Observed patterns per CV fold.
        pattern_perm_scores : ndarray, shape (n_folds, n_permutations, n_channels, n_times)
            Permutation patterns per fold.
        pattern_p_values : ndarray, shape (n_folds, n_channels, n_times)
            Per-fold pointwise p-values from cluster correction.
        pattern_masks : ndarray[bool], shape (n_folds, n_channels, n_times)
            Per-fold cluster-corrected significance masks.
    """
    scorer = get_scorer(scoring)

    # --- NaN diagnostics ---
    nan_per_epoch = np.isnan(X).any(axis=tuple(range(1, X.ndim))).sum()
    nan_per_channel = np.isnan(X).any(axis=(0, 2)).sum() if X.ndim == 3 else None
    total_nans = np.isnan(X).sum()
    logger.info("X shape: %s, y shape: %s", X.shape, y.shape)
    logger.info("Total NaN elements: %d / %d (%.2f%%)",
                total_nans, X.size, 100 * total_nans / X.size)
    logger.info("Epochs with any NaN: %d / %d", nan_per_epoch, X.shape[0])
    if nan_per_channel is not None:
        logger.info("Channels with any NaN: %d / %d", nan_per_channel, X.shape[1])
    for cls in np.unique(y):
        cls_mask = y == cls
        cls_nan = np.isnan(X[cls_mask]).any(axis=tuple(range(1, X.ndim))).sum()
        logger.info("  Class %s: %d total, %d with NaN, %d clean",
                     cls, cls_mask.sum(), cls_nan, cls_mask.sum() - cls_nan)

    splits = list(cv.split(X, y))
    if len(splits) == 0:
        raise ValueError("CV splitter produced no splits")

    obs_scores = []
    perm_scores = []

    # pattern containers (only used when use_pattern=True)
    pattern_obs_list = [] if use_pattern else None
    pattern_perm_list = [] if use_pattern else None

    for tr, te in tqdm(splits, desc="Cross-validation"):
        dec = clone(decoder)
        X_train, X_test, y_train, y_test = sample_fold(X, y, tr, te)

        # ---- observed ----
        dec.fit(X_train, y_train)
        observed_score = scorer(dec, X_test, y_test)
        obs_scores.append(observed_score)

        if use_pattern:
            observed_pattern = get_coef(dec, "patterns_", inverse_transform=True)
            pattern_obs_list.append(observed_pattern)

        # ---- permutations ----
        rng_fold = np.random.RandomState(random_state)
        seeds_fold = rng_fold.randint(0, 2**31 - 1, size=n_permutations)

        def one_perm(seed):
            r = np.random.RandomState(seed)
            y_train_perm = y_train.copy()
            r.shuffle(y_train_perm)
            dec_p = clone(dec)
            dec_p.fit(X_train, y_train_perm)
            acc = scorer(dec_p, X_test, y_test)
            if use_pattern:
                pat = get_coef(dec_p, "patterns_", inverse_transform=True)
                return acc, pat
            return acc

        results_perm = Parallel(n_jobs=n_jobs)(
            delayed(one_perm)(s) for s in tqdm(seeds_fold, desc="Permutations")
        )

        if use_pattern:
            fold_perm_acc = np.asarray([r[0] for r in results_perm])
            fold_perm_pat = np.asarray([r[1] for r in results_perm])
            perm_scores.append(fold_perm_acc)
            pattern_perm_list.append(fold_perm_pat)
        else:
            perm_scores.append(np.asarray(results_perm))

        # Free per-fold intermediates
        del dec, X_train, X_test, y_train, y_test, results_perm
        gc.collect()

    # ---- aggregate accuracy ----
    score = np.mean(obs_scores)
    perm_scores = np.stack(perm_scores)  # (n_folds, n_permutations)

    # p-value (greater-is-better metric)
    p_value = (np.sum(perm_scores.mean(axis=0) >= score) + 1.0) / (n_permutations + 1.0)

    if not use_pattern:
        return obs_scores, perm_scores, p_value

    # ---- per-fold pattern cluster correction ----
    pattern_obs = np.stack(pattern_obs_list)        # (n_folds, n_chn, n_times)
    pattern_perm = np.stack(pattern_perm_list)      # (n_folds, n_perm, n_chn, n_times)

    n_folds = pattern_obs.shape[0]
    pattern_masks = []      # per-fold boolean masks
    pattern_p_values = []   # per-fold pointwise p-values

    for fi in range(n_folds):
        fold_obs = pattern_obs[fi]          # (n_chn, n_times)
        fold_perm = pattern_perm[fi]        # (n_perm, n_chn, n_times)
        mask_fi, pval_fi = cluster_correction(
            fold_obs, fold_perm, p_thresh=0.05, tails=2, ignore=(0,)
        )
        pattern_masks.append(mask_fi)
        pattern_p_values.append(pval_fi)

    pattern_masks = np.stack(pattern_masks)         # (n_folds, n_chn, n_times)
    pattern_p_values = np.stack(pattern_p_values)   # (n_folds, n_chn, n_times)

    return (
        obs_scores,
        perm_scores,
        p_value,
        pattern_obs,
        pattern_perm,
        pattern_p_values,
        pattern_masks,
    )
