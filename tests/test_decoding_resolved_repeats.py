"""Smoke tests for cheap outer-loop CV-seed repeats in time-resolved decoding."""
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import numpy as np
import pytest
from ieeg.calc.oversample import MinimumNaNSplit
from mne.decoding import Vectorizer
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.decoding.decoder import decode_cv_scores, decode_permutation_scores

RANDOM_SEED = 42


def _make_synthetic_data(n_trials=40, n_channels=4, n_times=32):
    rng = np.random.RandomState(0)
    X = rng.randn(n_trials, n_channels, n_times)
    signal = 2.0 * X[:, 0, n_times // 2]
    y = (signal + rng.randn(n_trials) * 0.2 > 0).astype(int)
    return X, y


def _make_pipeline():
    return make_pipeline(
        Vectorizer(),
        StandardScaler(),
        PCA(n_components=0.95, random_state=42),
        LinearSVC(random_state=42),
    )


def _run_repeat_loop(X_segment, y, n_folds, n_perm, n_repeats, n_jobs=1):
    """Mirror runner aggregation without BIDS I/O."""
    pipeline = _make_pipeline()
    n_time = 1
    accuracies = np.zeros((n_time, n_folds))
    baseline_accuracies = np.zeros((n_time, n_folds, n_perm))
    accuracy_repeats = np.zeros((n_time, n_repeats, n_folds))

    for r in range(n_repeats):
        cv_r = MinimumNaNSplit(
            n_splits=n_folds,
            n_repeats=1,
            random_state=RANDOM_SEED + r,
        )
        if r == 0:
            score, permutation_scores, _ = decode_permutation_scores(
                X_segment,
                y,
                cv_r,
                pipeline,
                n_jobs=n_jobs,
                n_permutations=n_perm,
                scoring="balanced_accuracy",
                random_state=RANDOM_SEED,
            )
            accuracies[0] = score
            baseline_accuracies[0] = permutation_scores
            accuracy_repeats[0, r] = score
        else:
            accuracy_repeats[0, r] = decode_cv_scores(
                X_segment,
                y,
                cv_r,
                pipeline,
                n_jobs=n_jobs,
                scoring="balanced_accuracy",
                random_state=RANDOM_SEED + r,
            )

    accuracy_stable = accuracy_repeats.mean(axis=(1, 2))
    return accuracies, baseline_accuracies, accuracy_repeats, accuracy_stable


def test_decode_cv_scores_shape():
    X, y = _make_synthetic_data()
    cv = MinimumNaNSplit(n_splits=2, n_repeats=1, random_state=RANDOM_SEED)
    scores = decode_cv_scores(
        X,
        y,
        cv,
        _make_pipeline(),
        n_jobs=1,
        scoring="balanced_accuracy",
        random_state=RANDOM_SEED,
    )
    assert scores.shape == (2,)


def test_repeat_loop_shapes_and_baseline_independent_of_repeats():
    X, y = _make_synthetic_data()
    n_folds, n_perm, n_repeats = 2, 2, 2

    accuracies, baseline, accuracy_repeats, accuracy_stable = _run_repeat_loop(
        X, y, n_folds, n_perm, n_repeats
    )

    assert accuracies.shape == (1, n_folds)
    assert baseline.shape == (1, n_folds, n_perm)
    assert accuracy_repeats.shape == (1, n_repeats, n_folds)
    assert accuracy_stable.shape == (1,)

    # baseline only from official repeat (r=0); shape must not depend on n_repeats
    acc1, base1, _, _ = _run_repeat_loop(X, y, n_folds, n_perm, n_repeats=1)
    assert base1.shape == baseline.shape


def test_n_repeats_one_matches_stable_mean():
    X, y = _make_synthetic_data()
    n_folds, n_perm = 2, 2

    accuracies, _, accuracy_repeats, accuracy_stable = _run_repeat_loop(
        X, y, n_folds, n_perm, n_repeats=1
    )

    np.testing.assert_allclose(
        accuracy_stable[0],
        accuracy_repeats[0, 0].mean(),
    )
    np.testing.assert_allclose(
        accuracy_stable[0],
        accuracies[0].mean(),
    )


def test_nan_fill_rerun_bit_identical_with_threads():
    """NaN fill must be fold-seeded so threaded runs are bit-identical."""
    X, y = _make_synthetic_data()
    # Inject sparse NaNs on a subset of trials only. MinimumNaNSplit treats any
    # NaN in a trial as marking that trial; keep enough fully-clean trials.
    nan_rng = np.random.RandomState(1)
    X = X.copy()
    n_trials, n_channels, n_times = X.shape
    nan_trials = nan_rng.choice(n_trials, size=max(1, n_trials // 4), replace=False)
    for i in nan_trials:
        X[i, nan_rng.randint(0, n_channels), nan_rng.randint(0, n_times)] = np.nan

    cv = MinimumNaNSplit(n_splits=2, n_repeats=1, random_state=RANDOM_SEED)
    pipeline = _make_pipeline()
    kwargs = dict(
        X=X,
        y=y,
        cv=cv,
        decoder=pipeline,
        n_jobs=8,
        n_permutations=2,
        scoring="balanced_accuracy",
        random_state=RANDOM_SEED,
    )

    obs1, perm1, _ = decode_permutation_scores(**kwargs)
    obs2, perm2, _ = decode_permutation_scores(**kwargs)

    assert np.array_equal(np.asarray(obs1), np.asarray(obs2))
    assert np.array_equal(perm1, perm2)

    cv_scores_1 = decode_cv_scores(
        X, y, cv, pipeline, n_jobs=8, scoring="balanced_accuracy", random_state=RANDOM_SEED
    )
    cv_scores_2 = decode_cv_scores(
        X, y, cv, pipeline, n_jobs=8, scoring="balanced_accuracy", random_state=RANDOM_SEED
    )
    assert np.array_equal(cv_scores_1, cv_scores_2)
