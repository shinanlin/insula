"""Smoke tests for outer-loop CV-seed repeats in windowed decoding."""
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import numpy as np
from ieeg.calc.oversample import MinimumNaNSplit
from mne.decoding import Vectorizer
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


def _run_windowed_repeat_loop(X, y, n_folds, n_perm, n_repeats, n_jobs=1):
    """Mirror run_decoding.py aggregation without BIDS I/O."""
    pipeline = _make_pipeline()
    accuracy_repeats = np.zeros((n_repeats, n_folds))
    perm_scores = None
    confusion = None

    for r in range(n_repeats):
        cv_r = MinimumNaNSplit(
            n_splits=n_folds,
            n_repeats=1,
            random_state=RANDOM_SEED + r,
        )
        if r == 0:
            score, perm_scores, _ = decode_permutation_scores(
                X,
                y,
                cv_r,
                pipeline,
                n_jobs=n_jobs,
                n_permutations=n_perm,
                scoring="balanced_accuracy",
                random_state=RANDOM_SEED,
            )
            accuracy_repeats[r] = score

            y_pred = get_cv_predict(
                X,
                y,
                cv_r,
                pipeline,
                n_jobs=n_jobs,
                random_state=RANDOM_SEED,
            )
            classes = np.unique(y)
            confusion = confusion_matrix(y, y_pred, labels=classes)
        else:
            accuracy_repeats[r] = decode_cv_scores(
                X,
                y,
                cv_r,
                pipeline,
                n_jobs=n_jobs,
                scoring="balanced_accuracy",
                random_state=RANDOM_SEED + r,
            )

    accuracy_stable = accuracy_repeats.mean()
    return score, perm_scores, accuracy_repeats, accuracy_stable, confusion


def test_windowed_repeat_loop_shapes():
    X, y = _make_synthetic_data()
    n_folds, n_perm, n_repeats = 2, 2, 2

    score, perm_scores, accuracy_repeats, accuracy_stable, confusion = (
        _run_windowed_repeat_loop(X, y, n_folds, n_perm, n_repeats)
    )

    score = np.asarray(score)
    assert score.shape == (n_folds,)
    assert perm_scores.shape == (n_folds, n_perm)
    assert accuracy_repeats.shape == (n_repeats, n_folds)
    assert accuracy_stable.shape == ()

    _, perm1, _, _, _ = _run_windowed_repeat_loop(
        X, y, n_folds, n_perm, n_repeats=1
    )
    assert perm1.shape == perm_scores.shape


def test_windowed_confusion_shape():
    X, y = _make_synthetic_data()
    n_classes = len(np.unique(y))

    _, _, _, _, confusion = _run_windowed_repeat_loop(
        X, y, n_folds=2, n_perm=2, n_repeats=1
    )

    assert confusion.shape == (n_classes, n_classes)


def test_windowed_n_repeats_one_matches_stable_mean():
    X, y = _make_synthetic_data()
    n_folds, n_perm = 2, 2

    score, _, accuracy_repeats, accuracy_stable, _ = _run_windowed_repeat_loop(
        X, y, n_folds, n_perm, n_repeats=1
    )

    score = np.asarray(score)
    np.testing.assert_allclose(accuracy_stable, accuracy_repeats[0].mean())
    np.testing.assert_allclose(accuracy_stable, score.mean())


def test_get_cv_predict_random_state_reproducible():
    X, y = _make_synthetic_data()
    cv = MinimumNaNSplit(n_splits=2, n_repeats=1, random_state=RANDOM_SEED)
    pipeline = _make_pipeline()

    pred1 = get_cv_predict(
        X, y, cv, pipeline, n_jobs=1, random_state=RANDOM_SEED
    )
    pred2 = get_cv_predict(
        X, y, cv, pipeline, n_jobs=1, random_state=RANDOM_SEED
    )

    assert np.array_equal(pred1, pred2)
