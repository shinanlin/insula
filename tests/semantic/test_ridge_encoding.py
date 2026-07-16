"""Tests for semantic ridge encoding pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.semantic.design_matrix import (
    TrialDesign,
    is_word_correct,
    parse_condition,
    reshape_r,
    vectorize_y,
)
from src.semantic.load_embeddings import load_embedding_table
from src.semantic.ridge_encode import (
    prepare_fold_neural,
    ridge_encode_group_cv,
    ridge_encode_with_significance,
    shuffle_train_embeddings,
)
from src.semantic.stats import channel_time_cluster_correction


def test_vectorize_reshape_roundtrip():
    y = np.arange(24, dtype=float).reshape(2, 3, 4)
    flat = vectorize_y(y)
    assert flat.shape == (2, 12)
    back = reshape_r(flat[0], 3, 4)
    np.testing.assert_array_equal(back, y[0])


def test_parse_condition_and_word_correct_filter():
  meta = parse_condition("Delay/Yes_No/Word/baron/CORRECT")
  assert meta["lexicality"] == "Word"
  assert meta["token"] == "baron"
  assert meta["remark"] == "CORRECT"
  assert is_word_correct(meta)

  bad = parse_condition("Delay/Yes_No/Word/baron/ERR_TASK")
  assert not is_word_correct(bad)

  nonword = parse_condition("Delay/Yes_No/Nonword/banic/CORRECT")
  assert not is_word_correct(nonword)


def test_embedding_alignment():
    from src.semantic.load_embeddings import align_embeddings

    table = load_embedding_table()
    vecs = table.vectors[[0, 1, 2]]
    aligned = align_embeddings(
        [str(table.tokens[0]), str(table.tokens[1]), str(table.tokens[2])],
        table,
    )
    np.testing.assert_allclose(aligned, vecs)


def test_embedding_unknown_token_raises():
    table = load_embedding_table()
    from src.semantic.load_embeddings import align_embeddings

    with pytest.raises(KeyError):
        align_embeddings(["not_a_real_word_token_xyz"], table)


def test_groupkfold_no_token_leak():
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    X = np.arange(len(groups))[:, None]
    cv = GroupKFold(n_splits=3)
    for train_idx, test_idx in cv.split(X, groups=groups):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        assert train_groups.isdisjoint(test_groups)


def _make_synthetic_design(
    n_items: int = 12,
    reps: int = 4,
    n_channels: int = 4,
    n_times: int = 6,
    n_features: int = 8,
    noise: float = 0.05,
    seed: int = 42,
) -> tuple[TrialDesign, int]:
    """Plant a strong signal at one spatiotemporal column; return its flat index."""
    rng = np.random.default_rng(seed)
    n_flat = n_channels * n_times
    target_col = n_flat // 2
    item_emb = rng.standard_normal((n_items, n_features))
    weights = rng.standard_normal(n_features)
    weights /= np.linalg.norm(weights)
    beta = np.zeros((n_features, n_flat))
    beta[:, target_col] = weights * 5.0

    tokens = [f"w{i:02d}" for i in range(n_items)]
    trial_tokens = []
    rows = []
    for tok_i, tok in enumerate(tokens):
        for _ in range(reps):
            trial_tokens.append(tok)
            x = item_emb[tok_i]
            y = x @ beta + rng.standard_normal(n_flat) * noise
            rows.append(y.reshape(n_channels, n_times))

    X = np.stack([item_emb[int(t[1:])] for t in trial_tokens])
    Y = np.stack(rows)
    groups = np.asarray(trial_tokens, dtype=object)

    design = TrialDesign(
        X=X,
        Y=Y,
        groups=groups,
        tokens=groups.copy(),
        ch_names=[f"ch{i}" for i in range(n_channels)],
        times=np.linspace(-0.5, 1.0, n_times),
        subject="SYN",
        phase="Delay",
        description="Decision",
        tmin=-0.5,
        tmax=1.0,
    )
    return design, target_col


def _make_synthetic_design_time_cluster(
    n_items: int = 12,
    reps: int = 4,
    n_channels: int = 4,
    n_times: int = 8,
    n_features: int = 8,
    noise: float = 0.01,
    seed: int = 42,
) -> tuple[TrialDesign, int, int]:
    """Plant strong signal across adjacent times at one channel."""
    rng = np.random.default_rng(seed)
    n_flat = n_channels * n_times
    ch, t0 = 1, n_times // 2
    item_emb = rng.standard_normal((n_items, n_features))
    weights = rng.standard_normal(n_features)
    weights /= np.linalg.norm(weights)
    beta = np.zeros((n_features, n_flat))
    for t in range(max(0, t0 - 2), min(n_times, t0 + 3)):
        beta[:, ch * n_times + t] = weights * 10.0

    tokens = [f"w{i:02d}" for i in range(n_items)]
    trial_tokens = []
    rows = []
    for tok_i, tok in enumerate(tokens):
        for _ in range(reps):
            trial_tokens.append(tok)
            x = item_emb[tok_i]
            y = x @ beta + rng.standard_normal(n_flat) * noise
            rows.append(y.reshape(n_channels, n_times))

    X = np.stack([item_emb[int(t[1:])] for t in trial_tokens])
    Y = np.stack(rows)
    groups = np.asarray(trial_tokens, dtype=object)

    design = TrialDesign(
        X=X,
        Y=Y,
        groups=groups,
        tokens=groups.copy(),
        ch_names=[f"ch{i}" for i in range(n_channels)],
        times=np.linspace(-0.5, 1.0, n_times),
        subject="SYN",
        phase="Delay",
        description="Decision",
        tmin=-0.5,
        tmax=1.0,
    )
    return design, ch, t0


def test_prepare_fold_neural_fills_nan():
    y_train = np.ones((4, 2, 3))
    y_train[1, 0, :] = np.nan
    y_test = np.ones((2, 2, 3))
    groups = np.array(["a", "a", "b", "b"], dtype=object)
    filled_train, filled_test = prepare_fold_neural(y_train, y_test, groups, random_state=0)
    assert np.all(np.isfinite(filled_train))
    assert np.all(np.isfinite(filled_test))


def test_synthetic_with_nan_runs():
    design, target_col = _make_synthetic_design()
    design.Y[0, 0, :] = np.nan
    result = ridge_encode_group_cv(design, k_pca=6, alpha=0.1, n_splits=4)
    assert np.isfinite(result.r_flat[target_col])


def test_synthetic_recovery():
    design, target_col = _make_synthetic_design()
    result = ridge_encode_group_cv(
        design,
        k_pca=6,
        alpha=0.1,
        n_splits=4,
        random_state=0,
    )
    assert abs(result.r_flat[target_col]) > 0.6
    assert abs(result.r_flat[target_col]) == pytest.approx(
        np.nanmax(np.abs(result.r_flat)), rel=0.05
    )


def test_channel_time_cluster_correction_shape():
    n_times, n_channels, n_perm = 8, 3, 20
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((n_times, n_channels))
    baseline = rng.standard_normal((n_times, n_channels, n_perm))
    mask, p_act = channel_time_cluster_correction(scores, baseline, p_thresh=0.05)
    assert mask.shape == (n_channels, n_times)
    assert p_act.shape == (n_channels, n_times)
    assert mask.dtype == bool
    assert np.all((p_act >= 0) & (p_act <= 1))


def test_fold_inner_shuffle_breaks_signal():
    design, target_col = _make_synthetic_design()
    true_res = ridge_encode_group_cv(design, k_pca=6, alpha=0.1, n_splits=4, random_state=0)
    perm_res = ridge_encode_group_cv(
        design,
        k_pca=6,
        alpha=0.1,
        n_splits=4,
        shuffle_within_folds=True,
        random_state=0,
    )
    assert abs(true_res.r_flat[target_col]) > abs(perm_res.r_flat[target_col]) + 0.15


def test_shuffle_train_embeddings_remaps_tokens():
    rng = np.random.default_rng(0)
    groups = np.array(["a", "a", "b", "b", "c", "c"], dtype=object)
    X = np.arange(6)[:, None].astype(float)
    shuffled = shuffle_train_embeddings(X, groups, rng)
    assert shuffled.shape == X.shape
    for tok in np.unique(groups):
        idx = groups == tok
        assert np.allclose(shuffled[idx], shuffled[idx][0])


def test_significance_detects_planted_column():
    design, ch, t0 = _make_synthetic_design_time_cluster()
    result = ridge_encode_with_significance(
        design,
        k_pca=6,
        alpha=0.1,
        n_splits=4,
        random_state=0,
        n_perm=50,
        p_thresh=0.05,
        n_jobs=1,
    )
    assert result.r_null.shape == (*result.r_map.shape, 50)
    assert result.mask.shape == result.r_map.shape
    assert abs(result.r_map[ch, t0]) > np.nanmean(np.abs(result.r_null[ch, t0, :]))
    assert result.mask[ch, :].any(), "planted channel should have a significant time cluster"


def test_synthetic_permute_collapses_encoding():
    design, target_col = _make_synthetic_design()
    true_res = ridge_encode_group_cv(design, k_pca=6, alpha=0.1, n_splits=4)
    perm_res = ridge_encode_group_cv(
        design,
        k_pca=6,
        alpha=0.1,
        n_splits=4,
        shuffle_within_folds=True,
        random_state=0,
    )
    assert abs(true_res.r_flat[target_col]) > abs(perm_res.r_flat[target_col]) + 0.15


def test_single_column_matches_manual_oof():
    """One (ch,t) column should match slow manual GroupKFold ridge."""
    design, _ = _make_synthetic_design(n_items=8, reps=2, n_channels=2, n_times=3)
    col = 2  # ch0, t2
    Y_col = vectorize_y(design.Y)[:, col]
    X = design.X
    groups = design.groups
    cv = GroupKFold(n_splits=3)
    pred = np.zeros_like(Y_col)
    for train_idx, test_idx in cv.split(X, groups=groups):
        scaler = StandardScaler().fit(X[train_idx])
        Xtr = scaler.transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        pca = PCA(n_components=3, random_state=0).fit(Xtr)
        Xtr_p = pca.transform(Xtr)
        Xte_p = pca.transform(Xte)
        m = Ridge(alpha=1.0).fit(Xtr_p, Y_col[train_idx])
        pred[test_idx] = m.predict(Xte_p)

    from scipy import stats

    manual_r, _ = stats.pearsonr(Y_col, pred)
    pipeline_r = ridge_encode_group_cv(design, k_pca=3, alpha=1.0, n_splits=3).r_flat[col]
    assert manual_r == pytest.approx(pipeline_r, abs=1e-10)
