"""Tests for whole-window Haufe pattern extraction."""

from __future__ import annotations

import numpy as np
from mne_bids import BIDSPath
from sklearn.model_selection import StratifiedKFold

from src.decoding.patterns import (
    cv_mean_pattern,
    fold_haufe_pattern_ct,
    make_decoding_pipeline,
    permutation_rank_proportions,
)
from src.decoding.run_decoding_patterns import pattern_datatype
from src.paths import decoding_task_dir


def test_fold_haufe_recovers_generative_feature():
    np.random.seed(42)
    n_trials, n_channels, n_times = 100, 10, 5
    X = np.random.randn(n_trials, n_channels, n_times)
    scores = 2.0 * X[:, 0, 2] - 1.5 * X[:, 1, 1] + np.random.randn(n_trials) * 0.1
    y = (scores > 0).astype(int)

    pipeline = make_decoding_pipeline(variance=0.99, random_state=42)
    pattern = fold_haufe_pattern_ct(X, y, pipeline, n_channels, n_times)

    assert pattern.shape == (n_channels, n_times)
    assert pattern[0, 2] > max(pattern[3:].ravel())
    assert pattern[1, 1] < min(pattern[3:].ravel())


def test_cv_mean_pattern_shape():
    np.random.seed(0)
    n_trials, n_channels, n_times = 24, 3, 4
    X = np.random.randn(n_trials, n_channels, n_times)
    signal = 3.0 * X[:, 0, 2] + np.random.randn(n_trials) * 0.1
    y = (signal > 0).astype(int)

    pipeline = make_decoding_pipeline(variance=0.99, random_state=42)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    avg_pattern = cv_mean_pattern(X, y, cv, pipeline, random_state=42)

    assert avg_pattern.shape == (n_channels, n_times)
    assert avg_pattern[0, 2] > 0
    assert avg_pattern[0, 2] == np.max(avg_pattern)


def test_multiclass_ovr_stack_shape():
    np.random.seed(1)
    n_trials, n_channels, n_times = 60, 4, 3
    X = np.random.randn(n_trials, n_channels, n_times)
    y = np.repeat(np.arange(3), 20)

    patterns = []
    for class_id in range(3):
        y_binary = (y == class_id).astype(np.int8)
        pipeline = make_decoding_pipeline(variance=0.99, random_state=42)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        patterns.append(cv_mean_pattern(X, y_binary, cv, pipeline, random_state=42))

    stacked = np.stack(patterns)
    assert stacked.shape == (3, n_channels, n_times)


def test_permutation_rank_proportions_supports_numpy_1():
    values = np.array([[-2.0, 3.0], [1.0, -1.0], [3.0, 2.0]])

    proportions = permutation_rank_proportions(values, tails=2)

    np.testing.assert_allclose(
        proportions,
        np.array([[0.5, 1.0], [0.0, 0.0], [1.0, 0.5]]),
    )


def test_pattern_datatype_and_output_path():
    assert pattern_datatype("lexicality") == "(decode)(pattern)lexicality"
    path = BIDSPath(
        root=str(decoding_task_dir("LexicalDelay")),
        subject="INSl",
        task="LexicalDelay",
        processing="Delay",
        description="Repeat",
        datatype="(decode)(pattern)lexicality",
        suffix="highgamma",
        extension=".h5",
        check=False,
    )
    assert "(decode)(pattern)lexicality" in str(path)
    assert "sub-INSl" in str(path)
    assert "results/decoding/LexicalDelay" in str(path)
