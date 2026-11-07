"""Repeated-CV balanced accuracy with a coherent repeated-stimulus null.

The statistic is the mean over every configured repeat and fold. Each null
draw is one global relabeling, reused in every fold. Stimulus repetitions are
permuted as complete equal-size blocks, preserving both repeated labels and
trial-level class totals. The caller supplies a label-free training transform.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
import warnings

import h5py
from ieeg.calc.oversample import MinimumNaNSplit
from joblib import Parallel, delayed
import numpy as np
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from src.decoding.decoder import sample_fold

LOGGER = logging.getLogger(__name__)
METHOD = 'repeated_cv_mean_whole_stimulus'
SCHEMA_VERSION = 2
SE_DESCRIPTION = 'SE across CV-repeat means; split variability conditional on the fixed dataset, not participant/population uncertainty'


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def whole_stimulus_permutations(y, groups, n_permutations, random_state=42):
    """Draw whole-stimulus label permutations within repeat-count strata.

    This deliberately differs from sklearn's within-group label shuffling:
    a stimulus's label is constant, so the entire stimulus block is exchanged
    with another block of the same size rather than shuffled internally.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    if y.ndim != 1 or groups.shape != y.shape:
        raise ValueError('y and stimulus groups must be aligned one-dimensional arrays')
    if n_permutations < 1:
        raise ValueError('At least one permutation is required')
    names, inverse, sizes = np.unique(groups, return_inverse=True, return_counts=True)
    labels = []
    for index in range(len(names)):
        values = np.unique(y[inverse == index])
        if len(values) != 1:
            raise ValueError(f'Stimulus {names[index]!r} does not have one constant label')
        labels.append(values[0])
    labels = np.asarray(labels, dtype=y.dtype)
    seeds = np.random.RandomState(random_state).randint(0, 2**31-1, size=n_permutations)
    draws, row_permutations = [], []
    for seed in seeds:
        rng = np.random.RandomState(int(seed))
        block_indices = np.arange(len(names))
        for size in sorted(set(sizes)):
            indices = np.flatnonzero(sizes == size)
            block_indices[indices] = rng.permutation(indices)
        draw = labels[block_indices][inverse]
        row_indices = np.empty(len(y), dtype=int)
        for destination, source in enumerate(block_indices):
            row_indices[np.flatnonzero(inverse == destination)] = np.flatnonzero(inverse == source)
        assert np.array_equal(np.sort(row_indices), np.arange(len(y)))
        assert np.array_equal(draw, y[row_indices])
        draws.append(draw)
        row_permutations.append(row_indices)
    return dict(labels=np.asarray(draws), row_permutations=np.asarray(row_permutations),
                seeds=seeds, group_names=names, group_inverse=inverse,
                group_sizes=sizes, group_labels=labels)


@dataclass
class FrozenFold:
    repeat: int
    fold: int
    train: np.ndarray
    test: np.ndarray
    X_train: np.ndarray
    X_test: np.ndarray
    full_train_digest: str
    full_test_digest: str
    pca_components: int


def repeated_cv_permutation(
    X, y, groups, transformer, classifier, *, keep_channels=None,
    n_splits=5, n_repeats=30, n_permutations=200, random_state=42,
    n_jobs=1, observed_reference: str | Path | None = None,
):
    """Fit observed and permuted labels using identical frozen full-input folds.

    Analysis channel exclusions are applied after splitting/filling the full
    reference input. This preserves the requested trial axis, CV and numeric
    imputations when an electrode is removed. Standardisation/PCA is fitted
    inside each training fold and reused across labels because it is unsupervised.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 3 or len(X) != len(y):
        raise ValueError('Expected trials x channels x time aligned with y')
    if n_repeats < 2:
        raise ValueError('At least two repeats are required to report repeat SE')
    keep = np.arange(X.shape[1]) if keep_channels is None else np.asarray(keep_channels, dtype=int)
    if keep.ndim != 1 or not len(keep) or len(np.unique(keep)) != len(keep):
        raise ValueError('At least one unique retained channel is required')
    if keep.min() < 0 or keep.max() >= X.shape[1]:
        raise ValueError('Retained channel indices are outside the input')
    plan = whole_stimulus_permutations(y, groups, n_permutations, random_state)
    scores = np.empty((n_repeats, n_splits))
    first_predictions = np.empty_like(y)
    observed_iterations = np.empty((n_repeats, n_splits), dtype=int)
    folds = []
    for repeat in range(n_repeats):
        cv = MinimumNaNSplit(n_splits=n_splits, n_repeats=1, random_state=random_state+repeat)
        splits = list(cv.split(X, y))
        assert len(splits) == n_splits
        assert np.array_equal(np.sort(np.concatenate([test for _, test in splits])), np.arange(len(y)))
        for fold, (train, test) in enumerate(splits):
            seed = random_state+repeat+fold
            full_train, full_test, y_train, y_test = sample_fold(X, y, train, test, seed=seed)
            # The existing sample_fold is numerically label-independent. Verify
            # that invariant before freezing it for all permutations.
            relabeled_train, relabeled_test, _, _ = sample_fold(X, plan['labels'][0], train, test, seed=seed)
            if not (np.array_equal(full_train, relabeled_train) and np.array_equal(full_test, relabeled_test)):
                raise ValueError('Imputation depends on target labels; cannot freeze it for this permutation test')
            prep = clone(transformer)
            a = prep.fit_transform(full_train[:, keep, :])
            b = prep.transform(full_test[:, keep, :])
            model = clone(classifier)
            model.fit(a, y_train)
            predicted = model.predict(b)
            scores[repeat, fold] = balanced_accuracy_score(y_test, predicted)
            observed_iterations[repeat, fold] = int(model.n_iter_)
            if repeat == 0:
                first_predictions[test] = predicted
            folds.append(FrozenFold(repeat, fold, train, test, a, b,
                array_digest(full_train), array_digest(full_test),
                int(prep.named_steps['pca'].n_components_)))
        LOGGER.info('Prepared/observed repeat %d/%d: %.8f', repeat+1, n_repeats, scores[repeat].mean())
    reference_check = None
    if observed_reference is not None:
        with h5py.File(observed_reference, 'r') as reference:
            expected = reference['accuracy_repeats'][:]
        if not np.array_equal(scores, expected):
            raise ValueError(f'Observed reference changed: {np.max(np.abs(scores-expected))}')
        reference_check = dict(path=str(observed_reference), all_fold_scores_identical=True,
            maximum_absolute_difference=0.0)

    def one_permutation(index, labels):
        null_scores = np.empty_like(scores)
        iterations = np.empty_like(observed_iterations)
        missing_test_class_folds = 0
        for item in folds:
            fit = clone(classifier)
            fit.fit(item.X_train, labels[item.train])
            predictions = fit.predict(item.X_test)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='y_pred contains classes not in y_true')
                null_scores[item.repeat, item.fold] = balanced_accuracy_score(labels[item.test], predictions)
            iterations[item.repeat, item.fold] = int(fit.n_iter_)
            missing_test_class_folds += np.unique(labels[item.test]).size < np.unique(y).size
        LOGGER.info('Permutation %d/%d: %.8f', index+1, n_permutations, null_scores.mean())
        return null_scores, iterations, missing_test_class_folds

    permuted = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(one_permutation)(index, labels) for index, labels in enumerate(plan['labels']))
    null_scores = np.stack([item[0] for item in permuted])
    null_iterations = np.stack([item[1] for item in permuted])
    repeat_scores = scores.mean(axis=1)
    statistic = float(scores.mean())
    null_means = null_scores.mean(axis=(1, 2))
    exceedances = int(np.count_nonzero(null_means >= statistic))
    classes = np.unique(y)
    return dict(accuracy=repeat_scores, accuracy_repeats=scores,
        accuracy_stable=statistic,
        accuracy_se=float(repeat_scores.std(ddof=1)/np.sqrt(n_repeats)),
        perm_scores=null_scores, permutation_means=null_means,
        p_value=(exceedances+1.)/(n_permutations+1.), null_exceedances=exceedances,
        confusion=confusion_matrix(y, first_predictions, labels=classes),
        confusion_norm=confusion_matrix(y, first_predictions, labels=classes, normalize='true'),
        classes=classes, first_repeat_predictions=first_predictions,
        observed_iterations=observed_iterations, permutation_iterations=null_iterations,
        pca_component_counts=np.asarray([f.pca_components for f in folds]).reshape(n_repeats, n_splits),
        folds=folds, permutation_plan=plan, reference_check=reference_check,
        null_test_folds_missing_any_class=int(sum(item[2] for item in permuted)),
        keep_channels=keep)
