"""Tests for shrinkage-LDA 2D cross-condition decoding."""
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import numpy as np
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from src.decoding.decoder import decode_cross_permutation_auc_pooled, decode_permutation_auc_pooled
from src.decoding.pooled_io import pair_keys, pair_pooled_conditions
from src.decoding.run_cross_condition_resolved_lda import cluster_correction_2d, window_slices
from src.decoding.run_decoding_resolved_lda import (
    build_classifier,
    build_pipeline,
    build_transformer,
)

RANDOM_SEED = 42


def _split_estimator():
    return dict(classifier=build_classifier(), transformer=build_transformer(5))


def _grouped_cv():
    return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)


def _make_grouped_data(n_items=40, n_channels=6, n_times=38, effect=1.5, seed=0):
    rng = np.random.RandomState(seed)
    item_labels = np.tile([0, 1], n_items // 2)
    conditions, y, trials, epochs = [], [], [], []
    for item in range(n_items):
        label = item_labels[item]
        item_offset = rng.randn() * 0.5
        for presentation in range(1, 3):
            epoch = rng.randn(n_channels, n_times)
            epoch[0] += effect * label
            epoch[1] += item_offset
            epochs.append(epoch)
            conditions.append(f"item{item:03d}")
            y.append(label)
            trials.append(f"item{item:03d}_{presentation}")
    labels = np.array(["Word" if value == 1 else "Nonword" for value in y])
    channels = np.array([f"ch{i}" for i in range(n_channels)])
    return (
        np.stack(epochs),
        np.asarray(y),
        np.asarray(conditions),
        labels,
        channels,
        np.asarray(trials),
    )


class TestPairKeys:
    def test_uses_trial_ids_when_present(self):
        trials = np.array(["banic_1", "banic_2"])
        conditions = np.array(["banic", "banic"])
        np.testing.assert_array_equal(pair_keys(trials, conditions), trials)

    def test_synthesises_keys_from_condition_occurrence(self):
        conditions = np.array(["banic", "banic", "baron"])
        np.testing.assert_array_equal(
            pair_keys(None, conditions),
            ["banic_1", "banic_2", "baron_1"],
        )


class TestPairPooledConditions:
    def _two_domains(self, shuffle_b=False, extra_trial=False, relabel_b=False):
        X, y, conditions, labels, channels, trials = _make_grouped_data()
        X_b = X + 0.01
        trials_b = trials.copy()
        y_b = y.copy()
        labels_b = labels.copy()
        conditions_b = conditions.copy()
        if shuffle_b:
            order = np.arange(len(y))[::-1]
            X_b, y_b, conditions_b, labels_b, trials_b = (
                X_b[order], y_b[order], conditions_b[order],
                labels_b[order], trials_b[order],
            )
        if extra_trial:
            trials_b = np.append(trials_b, "missing_1")
            X_b = np.concatenate([X_b, X_b[:1]], axis=0)
            y_b = np.append(y_b, y_b[:1])
            conditions_b = np.append(conditions_b, "missing")
            labels_b = np.append(labels_b, labels_b[:1])
        if relabel_b:
            labels_b = labels_b.copy()
            labels_b[0] = "Word" if labels_b[0] == "Nonword" else "Nonword"
        return (
            (X, y, conditions, labels, channels, trials),
            (X_b, y_b, conditions_b, labels_b, channels, trials_b),
        )

    def test_keeps_shared_trials_in_source_order(self):
        src, tgt = self._two_domains(shuffle_b=True, extra_trial=True)
        X_a, y, conditions, labels, channels, keys, X_b = pair_pooled_conditions(
            *src, *tgt
        )
        assert X_a.shape[0] == src[0].shape[0]
        assert X_b.shape[0] == src[0].shape[0]
        np.testing.assert_array_equal(keys, src[5])
        np.testing.assert_array_equal(conditions, src[2])

    def test_rejects_label_mismatch(self):
        src, tgt = self._two_domains(relabel_b=True)
        with pytest.raises(ValueError, match="labels differ"):
            pair_pooled_conditions(*src, *tgt)

    def test_word_groups_stay_aligned(self):
        src, tgt = self._two_domains(shuffle_b=True)
        _, y, conditions, labels, _, keys, _ = pair_pooled_conditions(*src, *tgt)
        for key, condition, label in zip(keys, conditions, labels):
            assert key.startswith(condition + "_")
            expected = "Word" if y[conditions == condition][0] == 1 else "Nonword"
            assert label == expected

    def test_intersects_channels(self):
        src, tgt = self._two_domains()
        X_b, y_b, cond_b, lab_b, ch_b, trials_b = tgt
        ch_b = np.append(ch_b[1:], "extra")
        X_b = np.concatenate([X_b[:, 1:], X_b[:, :1]], axis=1)
        X_a, _, _, _, channels, _, X_b_out = pair_pooled_conditions(
            *src, X_b, y_b, cond_b, lab_b, ch_b, trials_b
        )
        np.testing.assert_array_equal(channels, src[4][1:])
        assert X_a.shape[1] == X_b_out.shape[1] == len(channels)

    def test_pairs_without_trial_ids(self):
        src, tgt = self._two_domains()
        src = src[:5] + (None,)
        tgt = tgt[:5] + (None,)
        _, _, conditions, _, _, keys, _ = pair_pooled_conditions(*src, *tgt)
        assert keys[0] == "item000_1"
        np.testing.assert_array_equal(conditions[:2], ["item000", "item000"])


class TestCrossGroupCv:
    def test_held_out_words_are_held_out_of_both_domains(self):
        X, y, conditions, _, _, _ = _make_grouped_data()
        cv = _grouped_cv()
        for train_idx, test_idx in cv.split(X, y, conditions):
            assert not set(conditions[train_idx]) & set(conditions[test_idx])


class TestCrossPooledAuc:
    def test_matches_within_condition_auc_on_the_same_window(self):
        X, y, conditions, _, _, _ = _make_grouped_data(effect=1.5)
        estimator = _split_estimator()
        kwargs = dict(
            groups=conditions, n_jobs=1, n_permutations=12, random_state=RANDOM_SEED
        )
        within = decode_permutation_auc_pooled(
            X, y, _grouped_cv(), **estimator, **kwargs
        )
        cross = decode_cross_permutation_auc_pooled(
            X, X, y, _grouped_cv(), **estimator, **kwargs
        )
        assert cross[0].shape == (1,)
        assert within[0] == pytest.approx(cross[0][0], abs=1e-12)
        np.testing.assert_allclose(within[1], cross[1][:, 0], atol=1e-12)

    def test_recovers_a_planted_effect(self):
        X, y, conditions, _, _, _ = _make_grouped_data(effect=1.5)
        rng = np.random.RandomState(1)
        X_tgt = X + rng.randn(*X.shape) * 0.05
        obs_aucs, perm_aucs, p_values = decode_cross_permutation_auc_pooled(
            X, X_tgt, y, _grouped_cv(), **_split_estimator(),
            groups=conditions, n_jobs=1, n_permutations=60,
            random_state=RANDOM_SEED,
        )
        assert obs_aucs[0] > 0.7
        assert perm_aucs.shape == (60, 1)
        assert p_values[0] < 0.05

    def test_hoisted_transformer_matches_a_single_pipeline(self):
        X, y, conditions, _, _, _ = _make_grouped_data(effect=1.5)
        kwargs = dict(
            groups=conditions, n_jobs=1, n_permutations=20, random_state=RANDOM_SEED
        )
        split = decode_cross_permutation_auc_pooled(
            X, X, y, _grouped_cv(), **_split_estimator(), **kwargs
        )
        whole = decode_cross_permutation_auc_pooled(
            X, X, y, _grouped_cv(), build_pipeline(5), **kwargs
        )
        np.testing.assert_allclose(split[0], whole[0])
        np.testing.assert_allclose(split[1], whole[1])

    def test_scores_each_target_window(self):
        X, y, conditions, _, _, _ = _make_grouped_data(n_times=76, effect=1.5)
        obs_aucs, perm_aucs, p_values = decode_cross_permutation_auc_pooled(
            X[..., :38],
            X,
            y,
            _grouped_cv(),
            **_split_estimator(),
            groups=conditions,
            tgt_slices=((0, 38), (38, 76)),
            n_jobs=1,
            n_permutations=8,
            random_state=RANDOM_SEED,
        )
        assert obs_aucs.shape == (2,)
        assert perm_aucs.shape == (8, 2)
        assert p_values.shape == (2,)
        assert np.isfinite(obs_aucs).all()

    def test_requires_groups_for_group_permutation(self):
        X, y, _conditions, _, _, _ = _make_grouped_data(n_items=10)
        with pytest.raises(ValueError, match="requires a groups vector"):
            decode_cross_permutation_auc_pooled(
                X, X, y, StratifiedGroupKFold(n_splits=2),
                **_split_estimator(), groups=None, permute_groups=True,
                n_permutations=2,
            )


class TestClusterCorrection2d:
    def test_keeps_a_planted_blob_and_drops_scattered_noise(self):
        rng = np.random.RandomState(0)
        scores = np.full((8, 8), 0.51)
        scores[2:5, 2:5] = 0.80
        baseline = rng.normal(0.50, 0.03, size=(200, 8, 8))
        mask, p_act = cluster_correction_2d(scores, baseline)
        assert mask[3, 3]
        assert mask[2:5, 2:5].mean() > 0.5
        assert p_act[3, 3] < 0.05
        assert not mask[0, 0]

    def test_null_map_is_empty(self):
        rng = np.random.RandomState(1)
        scores = rng.normal(0.50, 0.02, size=(6, 6))
        baseline = rng.normal(0.50, 0.02, size=(150, 6, 6))
        mask, _ = cluster_correction_2d(scores, baseline)
        assert mask.sum() == 0


class TestWindowSlices:
    def test_skips_out_of_range_windows(self):
        times, slices = window_slices(-0.5, 1.5, 0.3, 0.03, 256, 128)
        assert times[0] == pytest.approx(-0.2)
        starts, ends = zip(*slices)
        assert min(starts) >= 0
        assert max(ends) <= 256
        assert all(end - start == 38 for start, end in slices)
