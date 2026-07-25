"""Deterministic trial-shuffle nulls and multiple-comparison correction."""

from __future__ import annotations

from hashlib import sha256
from typing import Iterable, Literal

import numpy as np


Tail = Literal["two-sided", "greater"]


def stable_seed(base_seed: int, entity_values: Iterable[object]) -> int:
    """Derive a stable uint32 seed from analysis entities."""

    text = "|".join(str(value) for value in entity_values)
    digest = sha256(f"{base_seed}|{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def generate_derangements(
    n_perm: int,
    n_trials: int,
    seed: int,
) -> np.ndarray:
    """Generate target-trial derangements shared by all pairs and metrics."""

    if n_perm < 1:
        raise ValueError("n_perm must be positive")
    if n_trials < 2:
        raise ValueError("At least two trials are required")
    rng = np.random.default_rng(seed)
    identity = np.arange(n_trials)
    dtype = np.uint16 if n_trials <= np.iinfo(np.uint16).max else np.uint32
    out = np.empty((n_perm, n_trials), dtype=dtype)
    for index in range(n_perm):
        candidate = rng.permutation(n_trials)
        while np.any(candidate == identity):
            candidate = rng.permutation(n_trials)
        out[index] = candidate
    return out


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR with NaN-preserving output."""

    p_values = np.asarray(p_values, dtype=float)
    result = np.full(p_values.shape, np.nan, dtype=float)
    valid = np.isfinite(p_values)
    p = p_values[valid]
    if p.size == 0:
        return result
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * p.size / np.arange(1, p.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    result[valid] = adjusted[inverse]
    return result


def scalar_permutation_inference(
    observed: np.ndarray,
    null: np.ndarray,
    *,
    tail: Tail,
    alpha: float = 0.05,
) -> dict[str, np.ndarray]:
    """Raw, BH-FDR, and studentized max-stat inference."""

    observed = np.asarray(observed, dtype=float)
    null = np.asarray(null, dtype=float)
    if null.ndim != 2 or null.shape[1] != observed.size:
        raise ValueError("null must have shape (permutation, observed.size)")

    center = np.nanmean(null, axis=0)
    scale = np.nanstd(null, axis=0, ddof=1)
    valid = np.isfinite(observed) & np.isfinite(center) & (scale > 0)
    obs_score = np.full(observed.shape, np.nan)
    null_score = np.full(null.shape, np.nan)
    if tail == "two-sided":
        obs_score[valid] = np.abs((observed[valid] - center[valid]) / scale[valid])
        null_score[:, valid] = np.abs(
            (null[:, valid] - center[valid]) / scale[valid]
        )
    elif tail == "greater":
        obs_score[valid] = (observed[valid] - center[valid]) / scale[valid]
        null_score[:, valid] = (
            null[:, valid] - center[valid]
        ) / scale[valid]
    else:
        raise ValueError(f"Unknown tail {tail!r}")

    p_uncorrected = np.full(observed.shape, np.nan)
    for index in np.flatnonzero(valid):
        p_uncorrected[index] = (
            1.0 + np.sum(null_score[:, index] >= obs_score[index])
        ) / (null.shape[0] + 1.0)

    finite_null = np.where(np.isfinite(null_score), null_score, -np.inf)
    global_max = np.max(finite_null, axis=1)
    p_fwer = np.full(observed.shape, np.nan)
    for index in np.flatnonzero(valid):
        p_fwer[index] = (
            1.0 + np.sum(global_max >= obs_score[index])
        ) / (null.shape[0] + 1.0)

    q_fdr = benjamini_hochberg(p_uncorrected)
    return {
        "null_mean": center,
        "null_std": scale,
        "observed_score": obs_score,
        "p_uncorrected": p_uncorrected,
        "q_fdr": q_fdr,
        "p_fwer_maxstat": p_fwer,
        "sig_fdr": q_fdr < alpha,
        "sig_fwer": p_fwer < alpha,
        "global_null_max": global_max,
    }
