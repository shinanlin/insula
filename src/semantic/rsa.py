"""Representational similarity analysis helpers for semantic geometry."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy import stats


def pairwise_rdm(
    patterns: np.ndarray,
    metric: str = "correlation",
) -> np.ndarray:
    """Build an item×item RDM from patterns of shape (n_items, n_features).

    Parameters
    ----------
    patterns
        One row per item.
    metric
        ``correlation`` → 1 − Pearson r; ``euclidean`` → L2 distance.
    """
    x = np.asarray(patterns, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"patterns must be 2-D, got shape {x.shape}")
    n = x.shape[0]
    rdm = np.zeros((n, n), dtype=float)
    if metric == "correlation":
        # Row-wise z-score; NaN-safe enough for v1
        mu = np.nanmean(x, axis=1, keepdims=True)
        sd = np.nanstd(x, axis=1, keepdims=True)
        sd = np.where(sd < 1e-12, 1.0, sd)
        z = (x - mu) / sd
        z = np.nan_to_num(z, nan=0.0)
        sim = z @ z.T / z.shape[1]
        rdm = 1.0 - sim
        np.fill_diagonal(rdm, 0.0)
        return rdm
    if metric == "euclidean":
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(x[i] - x[j]))
                rdm[i, j] = rdm[j, i] = d
        return rdm
    raise ValueError(f"Unknown metric: {metric!r}")


def upper_tri(rdm: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(rdm, k=1)
    return np.asarray(rdm)[idx]


def rsa_spearman(rdm_a: np.ndarray, rdm_b: np.ndarray) -> float:
    """Spearman correlation of upper-triangular RDM entries."""
    a = upper_tri(rdm_a)
    b = upper_tri(rdm_b)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    rho, _ = stats.spearmanr(a[mask], b[mask])
    return float(rho)


def residualize_vector(y: np.ndarray, *controls: np.ndarray) -> np.ndarray:
    """OLS residual of y against stacked control vectors (all 1-D, same length)."""
    y = np.asarray(y, dtype=float)
    cols = [np.asarray(c, dtype=float).reshape(-1) for c in controls]
    if not cols:
        return y.copy()
    X = np.column_stack(cols + [np.ones(len(y))])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_rsa(
    neural_rdm: np.ndarray,
    target_rdm: np.ndarray,
    control_rdms: list[np.ndarray],
) -> float:
    """Spearman RSA after residualizing both sides on control RDM vectors."""
    y = upper_tri(neural_rdm)
    t = upper_tri(target_rdm)
    controls = [upper_tri(c) for c in control_rdms]
    y_res = residualize_vector(y, *controls)
    t_res = residualize_vector(t, *controls)
    mask = np.isfinite(y_res) & np.isfinite(t_res)
    if mask.sum() < 3:
        return float("nan")
    rho, _ = stats.spearmanr(y_res[mask], t_res[mask])
    return float(rho)


def permute_rsa(
    neural_rdm: np.ndarray,
    model_rdm: np.ndarray,
    n_perm: int = 1000,
    rng: Optional[np.random.Generator] = None,
    statistic: Callable[[np.ndarray, np.ndarray], float] = rsa_spearman,
) -> tuple[float, float, np.ndarray]:
    """Label-permutation test by shuffling model RDM rows/cols jointly.

    Returns observed rho, two-sided p-value, and null distribution.
    """
    rng = rng or np.random.default_rng(0)
    obs = statistic(neural_rdm, model_rdm)
    n = model_rdm.shape[0]
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        order = rng.permutation(n)
        shuffled = model_rdm[np.ix_(order, order)]
        null[i] = statistic(neural_rdm, shuffled)
    # two-sided against null mean
    p = (np.sum(np.abs(null) >= abs(obs)) + 1.0) / (n_perm + 1.0)
    return obs, float(p), null
