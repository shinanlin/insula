"""Unit tests for electrode bootstrap NMF rank selection helpers."""

import numpy as np
import pandas as pd

from src.nmf.rank_selection import (
    _accumulate_consensus,
    choose_k,
    cophenetic_correlation,
    mean_pairwise_ari,
)


def _block_matrix(n_per_block: int = 12, n_features: int = 20, seed: int = 0):
    """Three well-separated nonnegative waveform blocks."""

    rng = np.random.default_rng(seed)
    templates = []
    t = np.linspace(0, 1, n_features)
    templates.append(np.exp(-((t - 0.2) / 0.08) ** 2))
    templates.append(np.exp(-((t - 0.5) / 0.12) ** 2))
    templates.append(np.exp(-((t - 0.8) / 0.10) ** 2))
    rows = []
    labels = []
    for c, tmpl in enumerate(templates):
        for _ in range(n_per_block):
            noise = 0.05 * rng.random(n_features)
            row = np.clip(tmpl + noise, 0, None)
            row = row / np.linalg.norm(row)
            rows.append(row)
            labels.append(c)
    X = np.asarray(rows)
    return X, np.asarray(labels)


def test_accumulate_consensus_perfect_blocks():
    _, labels = _block_matrix()
    n = len(labels)
    # Two identical full-sample "bootstraps"
    idx = np.arange(n)
    C = _accumulate_consensus([labels, labels], [idx, idx], n)
    assert C.shape == (n, n)
    np.testing.assert_allclose(np.diag(C), 1.0)
    # Same-cluster off-diagonal should be 1
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                assert C[i, j] == 1.0
            else:
                assert C[i, j] == 0.0


def test_cophenetic_high_for_block_consensus():
    _, labels = _block_matrix()
    n = len(labels)
    idx = np.arange(n)
    C = _accumulate_consensus([labels], [idx], n)
    corr = cophenetic_correlation(C)
    assert corr > 0.9


def test_choose_k_max_cophenetic_tie_break_smaller():
    metrics = pd.DataFrame(
        {
            "k": [2, 3, 4],
            "cophenetic": [0.80, 0.90, 0.90],
            "mean_ari": [0.7, 0.8, 0.75],
        }
    )
    decision = choose_k(metrics)
    assert decision["k"] == 3
    assert 3 in decision["near_tie_ks"] and 4 in decision["near_tie_ks"]


def test_mean_pairwise_ari_perfect_agreement():
    labels = np.array([0, 0, 1, 1, 2, 2])
    idx = np.arange(len(labels))
    # Permuted labels — ARI after Hungarian should still be 1
    perm = np.array([1, 1, 2, 2, 0, 0])
    ari = mean_pairwise_ari(
        [labels, perm],
        [idx, idx],
        k=3,
        random_state=0,
        max_pairs=10,
    )
    assert ari == 1.0
