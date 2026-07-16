"""Group-CV ridge encoding with fold-inner PCA for semantic features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.semantic.design_matrix import TrialDesign, reshape_r, vectorize_y
from src.semantic.stats import channel_time_cluster_correction


@dataclass(frozen=True)
class RidgeEncodeResult:
    """Out-of-fold encoding map and metadata."""

    r_map: np.ndarray  # (n_channels, n_times)
    r_flat: np.ndarray  # (n_channels * n_times,)
    y_pred_oof: np.ndarray  # (n_trials, n_channels * n_times)
    n_splits: int
    k_pca: int
    alpha: float
    permuted: bool


@dataclass(frozen=True)
class RidgeEncodeSignificanceResult:
    """Encoding map with permutation null and cluster-corrected significance."""

    r_map: np.ndarray  # (n_channels, n_times)
    r_flat: np.ndarray
    r_null: np.ndarray  # (n_channels, n_times, n_perm)
    mask: np.ndarray  # (n_channels, n_times)
    p_values: np.ndarray  # (n_channels, n_times)
    y_pred_oof: np.ndarray
    n_splits: int
    k_pca: int
    alpha: float
    n_perm: int
    p_thresh: float


def _effective_n_splits(n_groups: int, n_splits: int) -> int:
    if n_groups < 2:
        raise ValueError("Need at least 2 unique tokens for GroupKFold")
    return min(n_splits, n_groups)


def _pearson_columns(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Column-wise Pearson r between true and predicted matrices."""
    n_cols = y_true.shape[1]
    out = np.full(n_cols, np.nan, dtype=float)
    for j in range(n_cols):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        mask = np.isfinite(yt) & np.isfinite(yp)
        if mask.sum() < 3:
            continue
        if np.std(yt[mask]) < 1e-12 or np.std(yp[mask]) < 1e-12:
            continue
        out[j], _ = stats.pearsonr(yt[mask], yp[mask])
    return out


def shuffle_train_embeddings(
    X_train: np.ndarray,
    groups_train: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Within-fold shuffle: remap each token to a random other token's GloVe row."""
    unique = np.unique(groups_train)
    if len(unique) < 2:
        return np.asarray(X_train, dtype=float).copy()

    shuffled = rng.permutation(unique)
    mapping = {str(a): str(b) for a, b in zip(unique, shuffled)}

    token_to_x: dict[str, np.ndarray] = {}
    for i, g in enumerate(groups_train):
        key = str(g)
        if key not in token_to_x:
            token_to_x[key] = X_train[i]

    X_out = np.empty_like(X_train, dtype=float)
    for i, g in enumerate(groups_train):
        X_out[i] = token_to_x[mapping[str(g)]]
    return X_out


def permute_embedding_labels(design: TrialDesign, rng: np.random.Generator) -> TrialDesign:
    """Shuffle token->embedding mapping across unique tokens (trials follow tokens).

    Deprecated for significance testing; use fold-inner ``shuffle_train_embeddings``.
    """
    unique_tokens = np.unique(design.groups)
    shuffled = rng.permutation(unique_tokens)
    mapping = {str(a): str(b) for a, b in zip(unique_tokens, shuffled)}

    token_to_x: dict[str, np.ndarray] = {}
    for i, tok in enumerate(design.groups):
        key = str(tok)
        if key not in token_to_x:
            token_to_x[key] = design.X[i]

    new_tokens = np.asarray([mapping[str(t)] for t in design.tokens], dtype=object)
    new_groups = np.asarray([mapping[str(t)] for t in design.groups], dtype=object)
    X = np.stack([token_to_x[mapping[str(t)]] for t in design.groups])

    return TrialDesign(
        X=X,
        Y=design.Y,
        groups=new_groups,
        tokens=new_tokens,
        ch_names=design.ch_names,
        times=design.times,
        subject=design.subject,
        phase=design.phase,
        description=design.description,
        tmin=design.tmin,
        tmax=design.tmax,
    )


def prepare_fold_neural(
    y_train: np.ndarray,
    y_test: np.ndarray,
    groups_train: np.ndarray,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-token mixup on train fold neural data (matches decoding sample_fold)."""
    from ieeg.calc.oversample import mixup

    y_train = np.asarray(y_train, dtype=float).copy()
    y_test = np.asarray(y_test, dtype=float).copy()

    for tok in np.unique(groups_train):
        idx = groups_train == tok
        y_tok = y_train[idx]
        is_nan = np.isnan(y_tok)
        if is_nan.any():
            y_tok[is_nan] = 0.0
        mixup(y_tok, obs_axis=0, rng=random_state)
        y_train[idx] = y_tok

    is_nan_train = np.isnan(y_train)
    if is_nan_train.any():
        y_train[is_nan_train] = np.random.default_rng(random_state).normal(
            0.0, 1.0, int(is_nan_train.sum())
        )

    is_nan_test = np.isnan(y_test)
    if is_nan_test.any():
        y_test[is_nan_test] = np.random.default_rng(random_state + 1).normal(
            0.0, 1.0, int(is_nan_test.sum())
        )

    return y_train, y_test


def ridge_encode_group_cv(
    design: TrialDesign,
    k_pca: int = 10,
    alpha: float = 10.0,
    n_splits: int = 5,
    random_state: int = 0,
    permute_labels: bool = False,
    shuffle_within_folds: bool = False,
) -> RidgeEncodeResult:
    """Fit ridge encoding with token-group CV and fold-inner scaler/PCA."""
    rng = np.random.default_rng(random_state)
    if permute_labels:
        design = permute_embedding_labels(design, rng)

    X = np.asarray(design.X, dtype=float)
    Y_flat = vectorize_y(design.Y)
    groups = np.asarray(design.groups)
    n_groups = len(np.unique(groups))
    n_splits_eff = _effective_n_splits(n_groups, n_splits)

    cv = GroupKFold(n_splits=n_splits_eff)
    y_pred = np.zeros_like(Y_flat)
    k_eff = min(k_pca, X.shape[1], max(1, X.shape[0] - 1))

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        groups_train = groups[train_idx]

        if shuffle_within_folds:
            fold_rng = np.random.default_rng(random_state + fold_idx)
            X_train = shuffle_train_embeddings(X_train, groups_train, fold_rng)

        Y_train_3d, _ = prepare_fold_neural(
            design.Y[train_idx],
            design.Y[test_idx],
            groups_train,
            random_state=random_state,
        )
        Y_train = vectorize_y(Y_train_3d)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        pca = PCA(n_components=k_eff, random_state=random_state)
        X_train_p = pca.fit_transform(X_train_s)
        X_test_p = pca.transform(X_test_s)

        model = Ridge(alpha=alpha)
        model.fit(X_train_p, Y_train)
        y_pred[test_idx] = model.predict(X_test_p)

    r_flat = _pearson_columns(Y_flat, y_pred)
    r_map = reshape_r(r_flat, design.n_channels, design.n_times)

    return RidgeEncodeResult(
        r_map=r_map,
        r_flat=r_flat,
        y_pred_oof=y_pred,
        n_splits=n_splits_eff,
        k_pca=k_eff,
        alpha=alpha,
        permuted=permute_labels or shuffle_within_folds,
    )


def _one_permutation_map(
    design: TrialDesign,
    seed: int,
    k_pca: int,
    alpha: float,
    n_splits: int,
) -> np.ndarray:
    result = ridge_encode_group_cv(
        design,
        k_pca=k_pca,
        alpha=alpha,
        n_splits=n_splits,
        random_state=seed,
        shuffle_within_folds=True,
    )
    return result.r_map


def ridge_encode_with_significance(
    design: TrialDesign,
    k_pca: int = 10,
    alpha: float = 10.0,
    n_splits: int = 5,
    random_state: int = 0,
    n_perm: int = 500,
    p_thresh: float = 0.05,
    n_jobs: int = 1,
) -> RidgeEncodeSignificanceResult:
    """Observed encoding plus fold-inner permutation null and time cluster correction."""
    if n_perm < 1:
        raise ValueError("n_perm must be >= 1 for significance testing")

    observed = ridge_encode_group_cv(
        design,
        k_pca=k_pca,
        alpha=alpha,
        n_splits=n_splits,
        random_state=random_state,
        shuffle_within_folds=False,
    )

    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, 2**31 - 1, size=n_perm, dtype=np.int64)

    null_maps = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_one_permutation_map)(design, int(seed), k_pca, alpha, n_splits)
        for seed in seeds
    )
    r_null = np.stack(null_maps, axis=-1)

    mask, p_values = channel_time_cluster_correction(
        observed.r_map.T,
        r_null.transpose(1, 0, 2),
        p_thresh=p_thresh,
        tails=1,
    )

    return RidgeEncodeSignificanceResult(
        r_map=observed.r_map,
        r_flat=observed.r_flat,
        r_null=r_null,
        mask=mask,
        p_values=p_values,
        y_pred_oof=observed.y_pred_oof,
        n_splits=observed.n_splits,
        k_pca=observed.k_pca,
        alpha=observed.alpha,
        n_perm=n_perm,
        p_thresh=p_thresh,
    )
