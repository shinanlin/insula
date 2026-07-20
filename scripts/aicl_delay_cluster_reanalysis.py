#!/usr/bin/env python3
"""Cluster-correct an existing repeated-CV time-resolved decoding result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage


def max_cluster_length(mask: np.ndarray) -> int:
    labels, n_clusters = ndimage.label(mask)
    return max(
        (int(np.count_nonzero(labels == idx)) for idx in range(1, n_clusters + 1)),
        default=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    with h5py.File(args.input, "r") as handle:
        accuracy = handle["accuracy"][:]  # time, repeat, fold
        baseline = handle["baseline"][:]  # time, repeat, fold, permutation
        times = handle["time"][:]

    observed = accuracy.mean(axis=(1, 2))
    permutation_timecourses = baseline.mean(axis=(1, 2)).T
    n_permutations = permutation_timecourses.shape[0]

    pointwise_p = (
        1 + np.count_nonzero(permutation_timecourses >= observed[None, :], axis=0)
    ) / (n_permutations + 1)
    observed_mask = pointwise_p < args.alpha

    null_max_lengths = np.empty(n_permutations, dtype=int)
    for perm_idx, pseudo_observed in enumerate(permutation_timecourses):
        pseudo_p = (
            1
            + np.count_nonzero(
                permutation_timecourses >= pseudo_observed[None, :], axis=0
            )
        ) / (n_permutations + 1)
        null_max_lengths[perm_idx] = max_cluster_length(pseudo_p < args.alpha)

    labels, n_clusters = ndimage.label(observed_mask)
    clusters = []
    significant_mask = np.zeros_like(observed_mask)
    for cluster_idx in range(1, n_clusters + 1):
        indices = np.flatnonzero(labels == cluster_idx)
        length = len(indices)
        cluster_p = float(
            (1 + np.count_nonzero(null_max_lengths >= length))
            / (n_permutations + 1)
        )
        is_significant = cluster_p < args.alpha
        if is_significant:
            significant_mask[indices] = True
        clusters.append(
            {
                "start_time": round(float(times[indices[0]]), 2),
                "end_time": round(float(times[indices[-1]]), 2),
                "length": int(length),
                "cluster_p_value": cluster_p,
                "significant": is_significant,
            }
        )

    result = {
        "input": str(args.input.resolve()),
        "inference": "pointwise p<0.05; max-cluster-length permutation correction",
        "window_average_inference_deprecated": True,
        "n_permutations": int(n_permutations),
        "n_timepoints": int(len(times)),
        "times": np.round(times, 2).tolist(),
        "observed_balanced_accuracy": observed.tolist(),
        "pointwise_p_values": pointwise_p.tolist(),
        "pointwise_significant_times": np.round(times[observed_mask], 2).tolist(),
        "null_max_cluster_length_95th_percentile": float(
            np.percentile(null_max_lengths, 95)
        ),
        "clusters": clusters,
        "cluster_corrected_significant_times": np.round(
            times[significant_mask], 2
        ).tolist(),
        "any_significant_cluster": bool(significant_mask.any()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
