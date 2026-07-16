"""Multi-block ridge encoding with marginal and controlled (block-shuffle) nulls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.semantic.design_matrix import (
    MultiBlockTrialDesign,
    reshape_r,
    vectorize_y,
)
from src.semantic.ridge_encode import (
    _effective_n_splits,
    _pearson_columns,
    prepare_fold_neural,
    shuffle_train_embeddings,
)
from src.semantic.stats import channel_time_cluster_correction

EncodingModel = Literal[
    "semantic",
    "phon",
    "acoustic",
    "full_perm_semantic",
]

MODEL_SPECS: dict[str, dict] = {
    "semantic": {
        "blocks": ("semantic",),
        "shuffle_block": "semantic",
        "k_pca": {"semantic": 10},
    },
    "phon": {
        "blocks": ("phon",),
        "shuffle_block": "phon",
        "k_pca": {"phon": 8},
    },
    "acoustic": {
        "blocks": ("acoustic",),
        "shuffle_block": "acoustic",
        "k_pca": {"acoustic": 8},
    },
    "full_perm_semantic": {
        "blocks": ("semantic", "phon", "acoustic"),
        "shuffle_block": "semantic",
        "k_pca": {"semantic": 10, "phon": 8, "acoustic": 8},
    },
}


@dataclass(frozen=True)
class MultiBlockEncodeResult:
    """Observed multi-block encoding plus optional permutation significance."""

    r_map: np.ndarray
    r_flat: np.ndarray
    y_pred_oof: np.ndarray
    n_splits: int
    alpha: float
    model: str
    feature_blocks: tuple[str, ...]
    k_pca_per_block: dict[str, int]
    perm_shuffled_block: str | None
    r_null: np.ndarray | None = None
    mask: np.ndarray | None = None
    p_values: np.ndarray | None = None
    n_perm: int = 0
    p_thresh: float = 0.05


def _transform_blocks(
    blocks_train: dict[str, np.ndarray],
    blocks_test: dict[str, np.ndarray],
    k_pca: dict[str, int],
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Per-block StandardScaler + PCA; return concatenated train/test and effective k."""
    train_parts = []
    test_parts = []
    k_eff: dict[str, int] = {}
    for name in blocks_train:
        Xtr = np.asarray(blocks_train[name], dtype=float)
        Xte = np.asarray(blocks_test[name], dtype=float)
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)
        k = min(k_pca[name], Xtr_s.shape[1], max(1, Xtr_s.shape[0] - 1))
        k_eff[name] = int(k)
        pca = PCA(n_components=k, random_state=random_state)
        train_parts.append(pca.fit_transform(Xtr_s))
        test_parts.append(pca.transform(Xte_s))
    return np.hstack(train_parts), np.hstack(test_parts), k_eff


def _select_blocks(
    design: MultiBlockTrialDesign,
    block_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    return {name: np.asarray(design.blocks.get(name), dtype=float) for name in block_names}


def ridge_encode_multi_block(
    design: MultiBlockTrialDesign,
    model: EncodingModel = "full_perm_semantic",
    alpha: float = 10.0,
    n_splits: int = 5,
    random_state: int = 0,
    shuffle_within_folds: bool = False,
    k_pca_override: dict[str, int] | None = None,
) -> MultiBlockEncodeResult:
    """Group-CV ridge for a multi-block encoding model."""
    if model not in MODEL_SPECS:
        raise ValueError(f"Unknown model {model!r}; choose from {list(MODEL_SPECS)}")

    spec = MODEL_SPECS[model]
    block_names: tuple[str, ...] = spec["blocks"]
    shuffle_block: str = spec["shuffle_block"]
    k_pca = dict(spec["k_pca"])
    if k_pca_override:
        k_pca.update(k_pca_override)

    blocks_all = _select_blocks(design, block_names)
    Y_flat = vectorize_y(design.Y)
    groups = np.asarray(design.groups)
    n_splits_eff = _effective_n_splits(len(np.unique(groups)), n_splits)
    cv = GroupKFold(n_splits=n_splits_eff)
    y_pred = np.zeros_like(Y_flat)
    k_eff_last: dict[str, int] = {k: int(v) for k, v in k_pca.items()}

    # Use first block length as dummy for CV split indexing
    n_trials = design.Y.shape[0]
    dummy = np.zeros(n_trials)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(dummy, groups=groups)):
        groups_train = groups[train_idx]
        blocks_train = {n: arr[train_idx] for n, arr in blocks_all.items()}
        blocks_test = {n: arr[test_idx] for n, arr in blocks_all.items()}

        if shuffle_within_folds:
            fold_rng = np.random.default_rng(random_state + fold_idx)
            blocks_train[shuffle_block] = shuffle_train_embeddings(
                blocks_train[shuffle_block],
                groups_train,
                fold_rng,
            )

        Y_train_3d, _ = prepare_fold_neural(
            design.Y[train_idx],
            design.Y[test_idx],
            groups_train,
            random_state=random_state,
        )
        Y_train = vectorize_y(Y_train_3d)

        X_train_p, X_test_p, k_eff_last = _transform_blocks(
            blocks_train, blocks_test, k_pca, random_state
        )
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_p, Y_train)
        y_pred[test_idx] = ridge.predict(X_test_p)

    r_flat = _pearson_columns(Y_flat, y_pred)
    r_map = reshape_r(r_flat, design.n_channels, design.n_times)

    return MultiBlockEncodeResult(
        r_map=r_map,
        r_flat=r_flat,
        y_pred_oof=y_pred,
        n_splits=n_splits_eff,
        alpha=alpha,
        model=model,
        feature_blocks=block_names,
        k_pca_per_block=k_eff_last,
        perm_shuffled_block=shuffle_block if shuffle_within_folds else None,
    )


def _one_multi_perm(
    design: MultiBlockTrialDesign,
    model: EncodingModel,
    seed: int,
    alpha: float,
    n_splits: int,
    k_pca_override: dict[str, int] | None,
) -> np.ndarray:
    result = ridge_encode_multi_block(
        design,
        model=model,
        alpha=alpha,
        n_splits=n_splits,
        random_state=seed,
        shuffle_within_folds=True,
        k_pca_override=k_pca_override,
    )
    return result.r_map


def ridge_encode_multi_with_significance(
    design: MultiBlockTrialDesign,
    model: EncodingModel = "full_perm_semantic",
    alpha: float = 10.0,
    n_splits: int = 5,
    random_state: int = 0,
    n_perm: int = 500,
    p_thresh: float = 0.05,
    n_jobs: int = 1,
    k_pca_override: dict[str, int] | None = None,
) -> MultiBlockEncodeResult:
    """Observed multi-block encoding + same-model shuffle null + time cluster."""
    if n_perm < 1:
        raise ValueError("n_perm must be >= 1 for significance testing")

    observed = ridge_encode_multi_block(
        design,
        model=model,
        alpha=alpha,
        n_splits=n_splits,
        random_state=random_state,
        shuffle_within_folds=False,
        k_pca_override=k_pca_override,
    )

    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, 2**31 - 1, size=n_perm, dtype=np.int64)
    null_maps = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_one_multi_perm)(
            design, model, int(seed), alpha, n_splits, k_pca_override
        )
        for seed in seeds
    )
    r_null = np.stack(null_maps, axis=-1)

    mask, p_values = channel_time_cluster_correction(
        observed.r_map.T,
        r_null.transpose(1, 0, 2),
        p_thresh=p_thresh,
        tails=1,
    )

    return MultiBlockEncodeResult(
        r_map=observed.r_map,
        r_flat=observed.r_flat,
        y_pred_oof=observed.y_pred_oof,
        n_splits=observed.n_splits,
        alpha=observed.alpha,
        model=observed.model,
        feature_blocks=observed.feature_blocks,
        k_pca_per_block=observed.k_pca_per_block,
        perm_shuffled_block=MODEL_SPECS[model]["shuffle_block"],
        r_null=r_null,
        mask=mask,
        p_values=p_values,
        n_perm=n_perm,
        p_thresh=p_thresh,
    )
