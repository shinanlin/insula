"""Cluster-based significance for channel x time encoding maps."""

from __future__ import annotations

import numpy as np


def channel_time_cluster_correction(
    scores: np.ndarray,
    baseline: np.ndarray,
    p_thresh: float = 0.05,
    tails: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel time cluster permutation correction.

    Parameters
    ----------
    scores
        Observed statistic, shape ``(n_times, n_channels)``.
    baseline
        Permutation null, shape ``(n_times, n_channels, n_perm)``.
    p_thresh
        Cluster-forming threshold (one-sided, greater-is-better).
    tails
        Tail direction passed to ``ieeg.calc.stats`` helpers.

    Returns
    -------
    mask
        Cluster-corrected significance, shape ``(n_channels, n_times)``.
    p_act
        Point-wise permutation p-values, shape ``(n_channels, n_times)``.
    """
    from ieeg.calc.stats import proportion, tail_compare, time_cluster

    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D (n_times, n_channels), got {scores.shape}")
    if baseline.ndim != 3:
        raise ValueError(
            f"baseline must be 3D (n_times, n_channels, n_perm), got {baseline.shape}"
        )
    if scores.shape[0] != baseline.shape[0] or scores.shape[1] != baseline.shape[1]:
        raise ValueError(
            f"scores and baseline must match in (n_times, n_channels). "
            f"Got scores={scores.shape}, baseline={baseline.shape}"
        )

    n_times, n_channels = scores.shape
    mask = np.zeros((n_channels, n_times), dtype=bool)
    p_act = np.ones((n_channels, n_times), dtype=float)

    for ch in range(n_channels):
        sc = scores[:, ch]
        base = baseline[:, ch, :].T  # (n_perm, n_times)

        diff = base - sc[None, :]
        p_ch = (np.sum(diff >= 0, axis=0) + 1) / (diff.shape[0] + 1)
        p_perm = proportion(diff, tail=tails, axis=0)
        b_act = tail_compare(1.0 - p_ch, 1.0 - p_thresh, tails)
        b_perm = tail_compare(p_perm, 1.0 - p_thresh, tails)
        mask[ch, :] = time_cluster(b_act, b_perm, 1 - p_thresh, tails)
        p_act[ch, :] = p_ch

    return mask, p_act
