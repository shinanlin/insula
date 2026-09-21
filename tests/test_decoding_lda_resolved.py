"""Tests for the shrinkage-LDA / pooled-AUC time-resolved decoding path."""
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from src.decoding.decoder import (
    _group_label_table,
    decode_permutation_auc_pooled,
    sample_fold,
)
from src.decoding.run_decoding_resolved_lda import (
    TimeBin,
    build_classifier,
    build_cv,
    build_pipeline,
    build_transformer,
    drop_label_trials,
)

RANDOM_SEED = 42


def _split_estimator():
    return dict(classifier=build_classifier(), transformer=build_transformer(5))


def _grouped_cv():
    return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)


def _make_grouped_data(n_items=40, n_channels=6, n_times=38, effect=1.5, seed=0):
    """Two presentations per stimulus, label a deterministic function of the item."""
    rng = np.random.RandomState(seed)
    item_labels = np.tile([0, 1], n_items // 2)
    conditions, y, trials = [], [], []
    for item in range(n_items):
        label = item_labels[item]
        item_offset = rng.randn() * 0.5
        for _ in range(2):
            epoch = rng.randn(n_channels, n_times)
            epoch[0] += effect * label
            # Item identity leaves its own trace, shared by both presentations.
            epoch[1] += item_offset
            trials.append(epoch)
            conditions.append(f"item{item:03d}")
            y.append(label)
    return np.stack(trials), np.asarray(y), np.asarray(conditions)


def _make_multiclass_data(n_items=48, n_classes=4, n_channels=6, n_times=38,
                          effect=1.5, seed=0):
    """Articulator-like target: several classes, one channel carrying each."""
    rng = np.random.RandomState(seed)
    trials, y, conditions = [], [], []
    for item in range(n_items):
        label = item % n_classes
        item_offset = rng.randn() * 0.5
        for _ in range(2):
            epoch = rng.randn(n_channels, n_times)
            epoch[label] += effect
            epoch[n_classes] += item_offset
            trials.append(epoch)
            conditions.append(f"item{item:03d}")
            y.append(label)
    return np.stack(trials), np.asarray(y), np.asarray(conditions)


def _legacy_binary_pooled_auc(X, y, cv, groups, transformer, classifier,
                              random_state=RANDOM_SEED):
    """The pre-multiclass scoring path, kept verbatim as a regression reference.

    Macro one-vs-rest over the mirrored ``[-v, v]`` columns must reproduce this
    exactly, otherwise generalising the metric silently moved the published
    lexicality numbers.
    """
    oof = np.empty(y.shape[0], dtype=float)
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        X_train, X_test, _, _ = sample_fold(
            X, y, train_idx, test_idx, seed=random_state + fold_idx
        )
        prep = clone(transformer)
        X_train = prep.fit_transform(X_train)
        X_test = prep.transform(X_test)
        dec = clone(classifier)
        dec.fit(X_train, y[train_idx])
        values = np.asarray(dec.decision_function(X_test), dtype=float).ravel()
        spread = values.std()
        oof[test_idx] = (values - values.mean()) / spread if spread > 0 else 0.0
    return float(roc_auc_score(y, oof))


class _ClassHoldoutSplitter:
    """Two folds, one of which trains on data missing an entire class."""

    def __init__(self, holdout_class):
        self.holdout_class = holdout_class

    def split(self, X, y, groups=None):
        y = np.asarray(y)
        held = np.flatnonzero(y == self.holdout_class)
        rest = np.flatnonzero(y != self.holdout_class)
        first, second = rest[: rest.size // 2], rest[rest.size // 2:]
        yield np.concatenate([second, held]), first
        yield first, np.concatenate([second, held])


class TestTimeBin:
    def test_collapses_time_axis_to_n_bins(self):
        X = np.random.RandomState(0).randn(12, 5, 38)
        assert TimeBin(n_bins=5).transform(X).shape == (12, 5, 5)

    def test_bins_average_their_samples(self):
        X = np.arange(2 * 1 * 6, dtype=float).reshape(2, 1, 6)
        binned = TimeBin(n_bins=3).transform(X)
        np.testing.assert_allclose(binned[0, 0], [0.5, 2.5, 4.5])

    def test_uneven_split_is_allowed(self):
        X = np.random.RandomState(0).randn(4, 3, 38)
        assert TimeBin(n_bins=5).transform(X).shape == (4, 3, 5)

    def test_rejects_window_shorter_than_n_bins(self):
        X = np.random.RandomState(0).randn(4, 3, 3)
        with pytest.raises(ValueError, match="fewer than n_bins"):
            TimeBin(n_bins=5).transform(X)


class TestGroupedSplits:
    def test_no_stimulus_spans_train_and_test(self):
        X, y, conditions = _make_grouped_data()
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        for train_idx, test_idx in cv.split(X, y, conditions):
            assert not set(conditions[train_idx]) & set(conditions[test_idx])


class TestGroupLabelTable:
    def test_maps_each_group_to_its_single_label(self):
        _, y, conditions = _make_grouped_data(n_items=6)
        inverse, group_label = _group_label_table(y, conditions)
        assert group_label.shape == (6,)
        np.testing.assert_array_equal(group_label[inverse], y)

    def test_rejects_group_carrying_two_labels(self):
        y = np.array([0, 1, 0, 1])
        conditions = np.array(["a", "a", "b", "b"])
        with pytest.raises(ValueError, match="carries 2 labels"):
            _group_label_table(y, conditions)


class TestGroupPermutation:
    def test_permuted_labels_stay_constant_within_a_stimulus(self):
        _, y, conditions = _make_grouped_data(n_items=20)
        inverse, group_label = _group_label_table(y, conditions)
        rng = np.random.RandomState(7)
        for _ in range(20):
            permuted = group_label.copy()
            rng.shuffle(permuted)
            drawn = permuted[inverse]
            for condition in np.unique(conditions):
                assert np.unique(drawn[conditions == condition]).size == 1

    def test_permutation_preserves_class_balance(self):
        _, y, conditions = _make_grouped_data(n_items=20)
        inverse, group_label = _group_label_table(y, conditions)
        rng = np.random.RandomState(7)
        permuted = group_label.copy()
        rng.shuffle(permuted)
        assert np.bincount(permuted[inverse]).tolist() == np.bincount(y).tolist()


class TestSampleFoldLabelIndependence:
    """decode_permutation_auc_pooled caches imputed folds across permutations.

    That is only valid while sample_fold's imputation ignores the labels. If
    sample_fold ever changes, this test fails and the caching must be revisited.
    """

    def test_imputed_arrays_do_not_depend_on_labels(self):
        X, y, _ = _make_grouped_data(n_items=20)
        X[3, 0, :5] = np.nan
        X[11, 2, 7:9] = np.nan
        train_idx = np.arange(0, 30)
        test_idx = np.arange(30, 40)

        y_shuffled = y.copy()
        np.random.RandomState(1).shuffle(y_shuffled)

        train_a, test_a, _, _ = sample_fold(X, y, train_idx, test_idx, seed=RANDOM_SEED)
        train_b, test_b, _, _ = sample_fold(
            X, y_shuffled, train_idx, test_idx, seed=RANDOM_SEED
        )
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(test_a, test_b)


class TestPooledAuc:
    def test_recovers_a_planted_effect(self):
        X, y, conditions = _make_grouped_data(effect=1.5)
        obs_auc, perm_aucs, p_value = decode_permutation_auc_pooled(
            X, y, _grouped_cv(), **_split_estimator(),
            groups=conditions, n_jobs=1, n_permutations=60,
            random_state=RANDOM_SEED,
        )
        assert obs_auc > 0.7
        assert perm_aucs.shape == (60,)
        assert p_value < 0.05

    def test_null_data_is_not_significant(self):
        X, y, conditions = _make_grouped_data(effect=0.0)
        obs_auc, _, p_value = decode_permutation_auc_pooled(
            X, y, _grouped_cv(), **_split_estimator(),
            groups=conditions, n_jobs=1, n_permutations=60,
            random_state=RANDOM_SEED,
        )
        assert 0.25 < obs_auc < 0.75
        assert p_value > 0.05

    def test_null_distribution_centres_on_chance(self):
        X, y, conditions = _make_grouped_data(effect=1.5)
        _, perm_aucs, _ = decode_permutation_auc_pooled(
            X, y, _grouped_cv(), **_split_estimator(),
            groups=conditions, n_jobs=1, n_permutations=120,
            random_state=RANDOM_SEED,
        )
        assert abs(np.nanmean(perm_aucs) - 0.5) < 0.06

    def test_hoisted_transformer_matches_a_single_pipeline(self):
        """Caching the unsupervised transform must not change the answer."""
        X, y, conditions = _make_grouped_data(effect=1.5)
        kwargs = dict(
            groups=conditions, n_jobs=1, n_permutations=25, random_state=RANDOM_SEED
        )
        split = decode_permutation_auc_pooled(
            X, y, _grouped_cv(), **_split_estimator(), **kwargs
        )
        whole = decode_permutation_auc_pooled(
            X, y, _grouped_cv(), build_pipeline(5), **kwargs
        )
        assert split[0] == pytest.approx(whole[0])
        np.testing.assert_allclose(split[1], whole[1])

    def test_is_reproducible_across_calls(self):
        X, y, conditions = _make_grouped_data()
        kwargs = dict(
            groups=conditions, n_jobs=1, n_permutations=30, random_state=RANDOM_SEED
        )
        first = decode_permutation_auc_pooled(
            X, y, _grouped_cv(), **_split_estimator(), **kwargs
        )
        second = decode_permutation_auc_pooled(
            X, y, _grouped_cv(), **_split_estimator(), **kwargs
        )
        assert first[0] == second[0]
        np.testing.assert_array_equal(first[1], second[1])

    def test_batch_size_does_not_change_results(self):
        X, y, conditions = _make_grouped_data()
        kwargs = dict(
            groups=conditions, n_jobs=1, n_permutations=24, random_state=RANDOM_SEED
        )
        one = decode_permutation_auc_pooled(
            X, y, _grouped_cv(), **_split_estimator(), batch_size=1, **kwargs
        )
        many = decode_permutation_auc_pooled(
            X, y, _grouped_cv(), **_split_estimator(), batch_size=7, **kwargs
        )
        np.testing.assert_array_equal(one[1], many[1])
        assert one[2] == many[2]

    def test_requires_groups_for_group_permutation(self):
        X, y, _ = _make_grouped_data(n_items=10)
        with pytest.raises(ValueError, match="requires a groups vector"):
            decode_permutation_auc_pooled(
                X, y, StratifiedGroupKFold(n_splits=2), **_split_estimator(),
                groups=None, permute_groups=True, n_permutations=2,
            )

    def test_rejects_single_class_target(self):
        X, _, conditions = _make_grouped_data(n_items=12)
        y = np.zeros(X.shape[0], dtype=int)
        with pytest.raises(ValueError, match="at least two classes"):
            decode_permutation_auc_pooled(
                X, y, StratifiedGroupKFold(n_splits=2), **_split_estimator(),
                groups=conditions, n_permutations=2,
            )

    def test_tolerates_nan_channels(self):
        X, y, conditions = _make_grouped_data(effect=1.5)
        X[:20, 4, :] = np.nan
        obs_auc, perm_aucs, _ = decode_permutation_auc_pooled(
            X, y, _grouped_cv(), **_split_estimator(),
            groups=conditions, n_jobs=1, n_permutations=20,
            random_state=RANDOM_SEED,
        )
        assert np.isfinite(obs_auc)
        assert np.isfinite(perm_aucs).all()


class TestBinaryRegression:
    """Generalising to macro OVR must leave the binary numbers untouched."""

    def test_matches_the_pre_multiclass_scoring(self):
        X, y, conditions = _make_grouped_data(effect=1.5)
        estimator = _split_estimator()
        obs_auc, _, _ = decode_permutation_auc_pooled(
            X, y, _grouped_cv(), **estimator,
            groups=conditions, n_jobs=1, n_permutations=2,
            random_state=RANDOM_SEED,
        )
        legacy = _legacy_binary_pooled_auc(
            X, y, _grouped_cv(), conditions,
            estimator["transformer"], estimator["classifier"],
        )
        assert obs_auc == pytest.approx(legacy, abs=1e-12)

    def test_matches_on_null_data_too(self):
        X, y, conditions = _make_grouped_data(effect=0.0, seed=3)
        estimator = _split_estimator()
        obs_auc, _, _ = decode_permutation_auc_pooled(
            X, y, _grouped_cv(), **estimator,
            groups=conditions, n_jobs=1, n_permutations=2,
            random_state=RANDOM_SEED,
        )
        legacy = _legacy_binary_pooled_auc(
            X, y, _grouped_cv(), conditions,
            estimator["transformer"], estimator["classifier"],
        )
        assert obs_auc == pytest.approx(legacy, abs=1e-12)


class TestMulticlassAuc:
    def test_recovers_a_planted_four_class_effect(self):
        X, y, conditions = _make_multiclass_data(effect=1.5)
        obs_auc, perm_aucs, p_value = decode_permutation_auc_pooled(
            X, y, _grouped_cv(), **_split_estimator(),
            groups=conditions, n_jobs=1, n_permutations=60,
            random_state=RANDOM_SEED,
        )
        assert obs_auc > 0.7
        assert perm_aucs.shape == (60,)
        assert p_value < 0.05

    def test_chance_stays_at_half_without_an_effect(self):
        X, y, conditions = _make_multiclass_data(effect=0.0, seed=5)
        obs_auc, perm_aucs, p_value = decode_permutation_auc_pooled(
            X, y, _grouped_cv(), **_split_estimator(),
            groups=conditions, n_jobs=1, n_permutations=120,
            random_state=RANDOM_SEED,
        )
        assert 0.25 < obs_auc < 0.75
        assert abs(np.nanmean(perm_aucs) - 0.5) < 0.06
        assert p_value > 0.05

    def test_chance_is_unmoved_by_class_imbalance(self):
        """Macro OVR, unlike accuracy, does not drift with the class prior."""
        X, y, conditions = _make_multiclass_data(n_items=60, effect=0.0, seed=8)
        # Thin one class at the item level, so no stimulus is split in half.
        items = np.array([int(name[4:]) for name in conditions])
        keep = (y != 3) | (items % 3 == 0)
        counts = np.bincount(y[keep])
        assert counts.min() * 3 == counts.max()
        _, perm_aucs, _ = decode_permutation_auc_pooled(
            X[keep], y[keep], _grouped_cv(), **_split_estimator(),
            groups=conditions[keep], n_jobs=1, n_permutations=120,
            random_state=RANDOM_SEED,
        )
        assert abs(np.nanmean(perm_aucs) - 0.5) < 0.06

    def test_training_split_missing_a_class_is_degenerate(self):
        X, y, conditions = _make_multiclass_data(n_items=24, effect=1.0)
        with pytest.raises(ValueError, match="Observed fit degenerated"):
            decode_permutation_auc_pooled(
                X, y, _ClassHoldoutSplitter(holdout_class=2),
                **_split_estimator(), groups=conditions,
                n_jobs=1, n_permutations=2, random_state=RANDOM_SEED,
            )


class TestCvScheme:
    def test_group_scheme_keeps_a_stimulus_on_one_side(self):
        X, y, conditions = _make_grouped_data(n_items=20)
        for train_idx, test_idx in build_cv("group", 5).split(X, y, conditions):
            assert not set(conditions[train_idx]) & set(conditions[test_idx])

    def test_stratified_scheme_ignores_the_grouping_vector(self):
        """Articulator runs split ungrouped while the null stays word-level."""
        X, y, conditions = _make_grouped_data(n_items=20)
        cv = build_cv("stratified", 5)
        assert isinstance(cv, StratifiedKFold)
        spans = [
            bool(set(conditions[train_idx]) & set(conditions[test_idx]))
            for train_idx, test_idx in cv.split(X, y, conditions)
        ]
        assert any(spans)

    def test_rejects_an_unknown_scheme(self):
        with pytest.raises(ValueError, match="Unknown cv_scheme"):
            build_cv("grouped", 5)


class TestDropLabelTrials:
    def _labelled(self, n_items=24):
        X, y, conditions = _make_multiclass_data(n_items=n_items)
        labels = np.array(["other" if value == 3 else f"c{value}" for value in y])
        return X, y, conditions, labels

    def test_removes_the_named_label_from_every_array(self):
        X, y, conditions, labels = self._labelled()
        expected = int((labels == "other").sum())
        X_kept, y_kept, cond_kept, lab_kept, dropped = drop_label_trials(
            X, y, conditions, labels, ("other",)
        )
        assert dropped == expected
        assert "other" not in set(lab_kept)
        remaining = X.shape[0] - expected
        assert X_kept.shape[0] == remaining
        assert y_kept.shape[0] == cond_kept.shape[0] == lab_kept.shape[0] == remaining

    def test_is_a_noop_without_labels_to_drop(self):
        X, y, conditions, labels = self._labelled()
        X_kept, _, _, lab_kept, dropped = drop_label_trials(
            X, y, conditions, labels, ()
        )
        assert dropped == 0
        assert X_kept.shape == X.shape
        np.testing.assert_array_equal(lab_kept, labels)

    def test_rejects_dropping_every_trial(self):
        X, y, conditions, labels = self._labelled()
        with pytest.raises(ValueError, match="removed every trial"):
            drop_label_trials(X, y, conditions, labels, tuple(set(labels)))
