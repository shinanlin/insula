"""Single-electrode, time-resolved insula reaction-time prediction.

Outer cross-validation is grouped by stimulus item.  Permutation targets use
the legacy null requested for this analysis: within each outer training fold,
all training RT values are shuffled without item or block restrictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from himalaya.ridge import RidgeCV
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, PredefinedSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_ALPHAS = np.logspace(-3, 3, 10)


@dataclass(frozen=True)
class WindowScores:
    """Cross-validated scores for every channel in one time window."""

    score_r: np.ndarray
    score_r2: np.ndarray
    score_mae: np.ndarray
    perm_score_r: np.ndarray
    oof_prediction: np.ndarray


@dataclass(frozen=True)
class ClusterResult:
    """Pointwise and joint phase/electrode/time cluster inference."""

    point_p: np.ndarray
    cluster_p_fwer: np.ndarray
    sig_mask_fwer: np.ndarray


def parse_item_id(task: str, event_name: str) -> str:
    """Extract a task-aware stimulus item from an epoch event path."""

    parts = str(event_name).split("/")
    if task == "PictureNaming":
        if len(parts) < 2:
            raise ValueError(f"Cannot parse PictureNaming item: {event_name!r}")
        return Path(parts[1]).stem
    if task in {"LexicalDelay", "PhonemeSequence"}:
        if len(parts) < 4:
            raise ValueError(f"Cannot parse {task} item: {event_name!r}")
        return "/".join(parts[2:4])
    raise ValueError(f"Unsupported RT task: {task!r}")


def make_group_splits(
    item_ids: Iterable[str],
    *,
    n_splits: int = 10,
    random_state: int = 42,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Create shuffled GroupKFold splits and trial-level test-fold ids."""

    groups = np.asarray(list(item_ids), dtype=str)
    actual_splits = min(int(n_splits), len(np.unique(groups)))
    if actual_splits < 2:
        raise ValueError("Need at least two unique items for grouped CV")
    cv = GroupKFold(
        n_splits=actual_splits,
        shuffle=True,
        random_state=int(random_state),
    )
    dummy = np.zeros((len(groups), 1), dtype=float)
    splits = [(tr, te) for tr, te in cv.split(dummy, groups=groups)]
    fold_id = np.full(len(groups), -1, dtype=np.int16)
    for fold, (train, test) in enumerate(splits):
        overlap = np.intersect1d(groups[train], groups[test])
        if overlap.size:
            raise RuntimeError(f"Item leakage in fold {fold}: {overlap.tolist()}")
        fold_id[test] = fold
    if np.any(fold_id < 0):
        raise RuntimeError("At least one trial has no outer-CV test fold")
    return splits, fold_id


def make_permutation_seeds(
    n_folds: int,
    n_perm: int,
    *,
    random_state: int = 42,
) -> np.ndarray:
    """Create a fold x permutation seed schedule shared by all models."""

    rng = np.random.RandomState(int(random_state))
    return rng.randint(
        0,
        np.iinfo(np.int32).max,
        size=(int(n_folds), int(n_perm)),
        dtype=np.int64,
    )


def make_shuffled_targets(y_train: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Shuffle all training RT values for each seed, without any grouping."""

    y_train = np.asarray(y_train, dtype=float)
    out = np.empty((len(y_train), len(seeds)), dtype=float)
    for column, seed in enumerate(seeds):
        order = np.random.RandomState(int(seed)).permutation(len(y_train))
        out[:, column] = y_train[order]
    return out


def _pearson_columns(y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(predictions, dtype=float)
    out = np.full(pred.shape[1], np.nan, dtype=float)
    for column in range(pred.shape[1]):
        keep = np.isfinite(y) & np.isfinite(pred[:, column])
        if keep.sum() < 2:
            continue
        dy = y[keep] - np.mean(y[keep])
        dp = pred[keep, column] - np.mean(pred[keep, column])
        denominator = np.linalg.norm(dy) * np.linalg.norm(dp)
        if denominator > 0:
            out[column] = float(np.dot(dy, dp) / denominator)
    return out


def _inner_cv(groups: np.ndarray, n_splits: int, random_state: int):
    _, fold_id = make_group_splits(
        groups,
        n_splits=min(int(n_splits), len(np.unique(groups))),
        random_state=int(random_state),
    )
    return PredefinedSplit(test_fold=fold_id)


def _fit_one_channel(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    permutation_seeds: np.ndarray,
    alphas: np.ndarray,
    inner_splits: int,
    random_state: int,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    n_perm = permutation_seeds.shape[1]
    predictions = np.full((len(y), 1 + n_perm), np.nan, dtype=float)

    for fold, (train, test) in enumerate(splits):
        y_train = np.asarray(y[train], dtype=float)
        targets = np.column_stack(
            [y_train, make_shuffled_targets(y_train, permutation_seeds[fold])]
        )
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            RidgeCV(
                alphas=np.asarray(alphas, dtype=float),
                fit_intercept=True,
                cv=_inner_cv(
                    groups[train],
                    n_splits=inner_splits,
                    random_state=int(random_state) + fold + 1,
                ),
            ),
        )
        try:
            model.fit(x[train], targets)
            predictions[test] = np.asarray(model.predict(x[test]), dtype=float)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return (
                np.nan,
                np.nan,
                np.nan,
                np.full(n_perm, np.nan),
                predictions[:, 0],
            )

    r_all = _pearson_columns(y, predictions)
    observed = predictions[:, 0]
    keep = np.isfinite(y) & np.isfinite(observed)
    if keep.sum() < 2:
        return np.nan, np.nan, np.nan, r_all[1:], observed
    residual = y[keep] - observed[keep]
    denominator = np.sum((y[keep] - np.mean(y[keep])) ** 2)
    score_r2 = (
        1.0 - np.sum(residual**2) / denominator if denominator > 0 else np.nan
    )
    return (
        float(r_all[0]),
        float(score_r2),
        float(np.mean(np.abs(residual))),
        r_all[1:],
        observed,
    )


def fit_window_scores(
    X: np.ndarray,
    y: np.ndarray,
    groups: Iterable[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    permutation_seeds: np.ndarray,
    *,
    alphas: np.ndarray = DEFAULT_ALPHAS,
    inner_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = 1,
) -> WindowScores:
    """Fit all single-channel models for one sliding window.

    ``X`` has shape trial x channel x samples-within-window.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(list(groups), dtype=str)
    if X.ndim != 3 or X.shape[0] != len(y) or len(groups) != len(y):
        raise ValueError(f"Incompatible X/y/groups shapes: {X.shape}, {y.shape}")
    if permutation_seeds.shape[0] != len(splits):
        raise ValueError("permutation_seeds must have one row per outer fold")
    if int(n_jobs) == 0:
        raise ValueError("n_jobs must be non-zero")
    effective_jobs = int(n_jobs)
    if effective_jobs > 0:
        effective_jobs = min(effective_jobs, X.shape[1])

    fitted = Parallel(n_jobs=effective_jobs, batch_size=1)(
        delayed(_fit_one_channel)(
            X[:, channel, :],
            y,
            groups,
            splits,
            permutation_seeds,
            np.asarray(alphas, dtype=float),
            int(inner_splits),
            int(random_state),
        )
        for channel in range(X.shape[1])
    )
    return WindowScores(
        score_r=np.asarray([row[0] for row in fitted]),
        score_r2=np.asarray([row[1] for row in fitted]),
        score_mae=np.asarray([row[2] for row in fitted]),
        perm_score_r=np.stack([row[3] for row in fitted]),
        oof_prediction=np.stack([row[4] for row in fitted]),
    )


def iter_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open intervals for consecutive True samples."""

    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    edges = np.diff(padded)
    return list(
        zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist())
    )


def _longest_runs_by_column(mask: np.ndarray) -> np.ndarray:
    """Longest consecutive True run for each column of time x permutation."""

    mask = np.asarray(mask, dtype=bool)
    current = np.zeros(mask.shape[1], dtype=np.int32)
    longest = np.zeros(mask.shape[1], dtype=np.int32)
    for row in mask:
        current = np.where(row, current + 1, 0)
        longest = np.maximum(longest, current)
    return longest


def joint_cluster_correction(
    phase_scores: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    point_alpha: float = 0.05,
    cluster_alpha: float = 0.05,
) -> dict[str, ClusterResult]:
    """Max-cluster correction jointly across phase, electrode, and time.

    Temporal adjacency exists only within an electrode.  Each permutation's
    maximum run length is taken across every electrode and supplied phase.
    """

    if not phase_scores:
        return {}
    n_perm_values = {permutations.shape[2] for _, permutations in phase_scores.values()}
    if len(n_perm_values) != 1:
        raise ValueError("All phases must have the same number of permutations")
    n_perm = n_perm_values.pop()

    point_p: dict[str, np.ndarray] = {}
    observed_supra: dict[str, np.ndarray] = {}
    permutation_supra: dict[str, np.ndarray] = {}
    for phase, (observed, permutations) in phase_scores.items():
        observed = np.asarray(observed, dtype=float)
        permutations = np.asarray(permutations, dtype=float)
        if permutations.shape[:2] != observed.shape:
            raise ValueError(f"Observed/permutation shape mismatch for {phase}")
        valid = np.isfinite(permutations)
        exceed = np.sum(valid & (permutations >= observed[..., None]), axis=2)
        denominator = np.sum(valid, axis=2)
        p = (exceed + 1.0) / (denominator + 1.0)
        point_p[phase] = np.where(
            np.isfinite(observed) & (denominator > 0), p, 1.0
        )
        threshold = np.nanquantile(
            permutations, 1.0 - float(point_alpha), axis=2
        )
        observed_supra[phase] = point_p[phase] <= float(point_alpha)
        permutation_supra[phase] = permutations > threshold[..., None]

    null_max = np.zeros(n_perm, dtype=np.int32)
    for phase in phase_scores:
        supra = permutation_supra[phase]
        for channel in range(supra.shape[0]):
            null_max = np.maximum(
                null_max, _longest_runs_by_column(supra[channel])
            )

    output: dict[str, ClusterResult] = {}
    for phase, (observed, _) in phase_scores.items():
        cluster_p = np.ones_like(observed, dtype=float)
        significant = np.zeros_like(observed, dtype=bool)
        for channel in range(observed.shape[0]):
            for start, stop in iter_true_runs(observed_supra[phase][channel]):
                length = stop - start
                p_value = (1.0 + np.sum(null_max >= length)) / (n_perm + 1.0)
                cluster_p[channel, start:stop] = p_value
                significant[channel, start:stop] = p_value <= float(cluster_alpha)
        output[phase] = ClusterResult(
            point_p=point_p[phase],
            cluster_p_fwer=cluster_p,
            sig_mask_fwer=significant,
        )
    return output
