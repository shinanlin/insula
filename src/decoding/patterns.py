"""Whole-window Haufe (2014) activation patterns for ROI decoding."""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone

from src.decoding.decoder import sample_fold


def make_decoding_pipeline(
    variance: float = 0.85,
    random_state: int = 42,
    class_weight=None,
    max_iter: int = 1000,
):
    """Standard decoding pipeline: Vectorizer → StandardScaler → PCA → LinearSVC."""
    from mne.decoding import Vectorizer
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    return make_pipeline(
        Vectorizer(),
        StandardScaler(),
        PCA(n_components=variance, random_state=random_state),
        LinearSVC(
            random_state=random_state,
            class_weight=class_weight,
            max_iter=max_iter,
        ),
    )


def fold_haufe_pattern_ct(
    X_train: np.ndarray,
    y_train: np.ndarray,
    pipeline,
    n_channels: int,
    n_times: int,
) -> np.ndarray:
    """Matrix-free Haufe pattern reshaped to (n_channel, n_time)."""
    vectorizer = clone(pipeline.named_steps["vectorizer"])
    scaler = clone(pipeline.named_steps["standardscaler"])
    pca = clone(pipeline.named_steps["pca"])
    svc = clone(pipeline.named_steps["linearsvc"])

    X_vec = vectorizer.fit_transform(X_train)
    X_scaled = scaler.fit_transform(X_vec)
    X_pca = pca.fit_transform(X_scaled)
    svc.fit(X_pca, y_train)

    w_pca = svc.coef_.ravel()
    s_train = X_pca @ w_pca
    pattern_scaled = X_scaled.T @ s_train / max(len(y_train) - 1, 1)
    pattern = pattern_scaled * scaler.scale_
    expected = n_channels * n_times
    if pattern.size != expected:
        raise ValueError(f"Expected {expected} features, got {pattern.size}")
    return pattern.reshape(n_channels, n_times)


def cv_mean_pattern(
    X: np.ndarray,
    y: np.ndarray,
    cv,
    pipeline,
    random_state: int = 42,
) -> np.ndarray:
    """Fold-mean whole-epoch Haufe pattern, shape (n_channel, n_time)."""
    n_channels, n_times = X.shape[1], X.shape[2]
    patterns = []
    for fold_idx, (tr, te) in enumerate(cv.split(X, y)):
        X_train, _, y_train, _ = sample_fold(
            X,
            y,
            tr,
            te,
            seed=random_state + fold_idx,
        )
        patterns.append(
            fold_haufe_pattern_ct(X_train, y_train, pipeline, n_channels, n_times)
        )
    return np.mean(np.stack(patterns), axis=0)


def _wholewindow_one_perm(
    seed: int,
    X: np.ndarray,
    y: np.ndarray,
    splits,
    pipeline,
    n_channels: int,
    n_times: int,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    y_perm = y.copy()
    rng.shuffle(y_perm)
    patterns = []
    for fold_idx, (tr, te) in enumerate(splits):
        X_train, _, y_train, _ = sample_fold(
            X,
            y_perm,
            tr,
            te,
            seed=seed + fold_idx,
        )
        patterns.append(
            fold_haufe_pattern_ct(X_train, y_train, pipeline, n_channels, n_times)
        )
    return np.mean(np.stack(patterns), axis=0)


def wholewindow_pattern_null(
    X: np.ndarray,
    y: np.ndarray,
    cv,
    pipeline,
    n_permutations: int = 200,
    n_jobs: int = -1,
    random_state: int = 42,
) -> np.ndarray:
    """Global label-shuffle permutation null, shape (n_perm, n_channel, n_time)."""
    n_channels, n_times = X.shape[1], X.shape[2]
    splits = list(cv.split(X, y))
    rng = np.random.RandomState(random_state)
    seeds = rng.randint(0, 2**31 - 1, size=n_permutations)
    perm_args = [
        (s, X, y, splits, pipeline, n_channels, n_times) for s in seeds
    ]
    perm_patterns = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_wholewindow_one_perm)(*args) for args in perm_args
    )
    return np.stack(perm_patterns)


def permutation_rank_proportions(
    values: np.ndarray,
    *,
    tails: int,
    axis: int = 0,
) -> np.ndarray:
    """Return permutation ranks in [0, 1] without NumPy 2-only APIs."""
    ranked_values = np.asarray(values)
    if tails == 2:
        ranked_values = np.abs(ranked_values)
    elif tails == -1:
        ranked_values = -ranked_values
    elif tails != 1:
        raise ValueError("tails must be 1, 2, or -1")

    denominator = ranked_values.shape[axis] - 1
    if denominator < 1:
        raise ValueError("Need at least two permutations for rank proportions")
    order = np.argsort(ranked_values, axis=axis, kind="quicksort")
    ranks = np.argsort(order, axis=axis, kind="quicksort")
    return ranks / denominator


def pattern_ct_cluster_correction(
    pattern: np.ndarray,
    perm_pattern: np.ndarray,
    cluster_forming_p: float = 0.10,
    cluster_alpha: float = 0.05,
    tails: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel temporal cluster correction on (channel, time) pattern."""
    from ieeg.calc.stats import tail_compare, time_cluster

    pattern = np.asarray(pattern)
    perm_pattern = np.asarray(perm_pattern)
    if pattern.ndim != 2:
        raise ValueError(f"pattern must be (channel, time), got {pattern.shape}")
    if perm_pattern.ndim != 3:
        raise ValueError(
            f"perm_pattern must be (perm, channel, time), got {perm_pattern.shape}"
        )
    if perm_pattern.shape[1:] != pattern.shape:
        raise ValueError(
            f"perm_pattern channel/time must match pattern: "
            f"{perm_pattern.shape[1:]} != {pattern.shape}"
        )
    if not 0 < cluster_forming_p < 1:
        raise ValueError("cluster_forming_p must be between 0 and 1")
    if not 0 < cluster_alpha < 1:
        raise ValueError("cluster_alpha must be between 0 and 1")

    n_channels, _ = pattern.shape
    n_perm = perm_pattern.shape[0]
    mask = np.zeros((n_channels, pattern.shape[1]), dtype=bool)
    p_values = np.ones((n_channels, pattern.shape[1]), dtype=float)

    for ch_idx in range(n_channels):
        observed = pattern[ch_idx]
        baseline = perm_pattern[:, ch_idx, :]
        if tails == 2:
            p_act = (
                np.sum(np.abs(baseline) >= np.abs(observed)[None, :], axis=0) + 1
            ) / (n_perm + 1)
        elif tails == 1:
            p_act = (
                np.sum(baseline >= observed[None, :], axis=0) + 1
            ) / (n_perm + 1)
        elif tails == -1:
            p_act = (
                np.sum(baseline <= observed[None, :], axis=0) + 1
            ) / (n_perm + 1)
        else:
            raise ValueError("tails must be 1, 2, or -1")
        p_perm = permutation_rank_proportions(baseline, tails=tails, axis=0)
        b_act = p_act <= cluster_forming_p
        b_perm = tail_compare(
            p_perm, 1.0 - cluster_forming_p, tails=tails
        ).astype(bool)
        mask[ch_idx] = time_cluster(
            b_act, b_perm, p_val=1.0 - cluster_alpha, tails=tails
        )
        p_values[ch_idx] = p_act

    return mask, p_values
