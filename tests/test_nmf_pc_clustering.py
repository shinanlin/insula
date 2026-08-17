"""Unit tests for Castellucci-style PC clustering compute helpers."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.nmf.pc_clustering import (
    choose_k_max_metric,
    cluster_pc_space,
    n_pcs_above_threshold,
    summarize_cluster_metrics,
    write_pc_tables,
)


def test_n_pcs_above_threshold_counts_and_floors_at_two():
    explained = np.array([0.30, 0.12, 0.06, 0.04, 0.03])
    assert n_pcs_above_threshold(explained, threshold=0.05) == 3
    assert n_pcs_above_threshold(np.array([0.40, 0.03, 0.02]), threshold=0.05) == 2


def test_cluster_pc_space_writes_expected_rows():
    rng = np.random.default_rng(0)
    blocks = [
        rng.normal(loc=center, scale=0.1, size=(12, 3))
        for center in ((0, 0, 0), (3, 0, 0), (0, 3, 0))
    ]
    scores = np.vstack(blocks)
    frame = cluster_pc_space(
        scores,
        k_min=2,
        k_max=3,
        n_iter=4,
        random_state=0,
        methods=("kmeans", "ward"),
    )
    assert set(frame.columns) == {
        "method",
        "k",
        "iteration",
        "silhouette",
        "calinski_harabasz",
    }
    assert set(frame["method"]) == {"kmeans", "ward"}
    assert frame.loc[frame["method"].eq("kmeans")].shape[0] == 4 * 2
    assert frame.loc[frame["method"].eq("ward")].shape[0] == 2
    assert frame["silhouette"].notna().all()


def test_choose_k_max_metric_tie_breaks_smaller():
    summary = pd.DataFrame(
        {
            "method": ["kmeans", "kmeans", "kmeans"],
            "k": [2, 3, 4],
            "silhouette_median": [0.40, 0.50, 0.50],
        }
    )
    choice = choose_k_max_metric(summary, "silhouette_median")
    assert choice["k"] == 3


def test_write_pc_tables_does_not_write_svg(tmp_path: Path):
    explained = np.array([0.4, 0.2, 0.1])
    scores = np.arange(12, dtype=float).reshape(4, 3)
    meta = pd.DataFrame(
        {
            "channel": ["a", "b", "c", "d"],
            "functional_cluster": ["motor", "motor", "sustain", "sensory"],
        }
    )
    iterations = cluster_pc_space(
        scores,
        k_min=2,
        k_max=2,
        n_iter=2,
        random_state=1,
        methods=("kmeans",),
    )
    paths = write_pc_tables(
        explained=explained,
        scores=scores,
        meta_rows=meta,
        iterations=iterations,
        run_meta={"n_embedding_pcs": 3},
        results_dir=tmp_path,
    )
    assert set(paths) == {"scree", "scores", "iterations", "summary", "meta"}
    for path in paths.values():
        assert path.is_file()
        assert path.suffix != ".svg"
    assert list(tmp_path.glob("*.svg")) == []
    summary = summarize_cluster_metrics(iterations)
    assert "silhouette_median" in summary.columns
    scores_frame = pd.read_csv(paths["scores"])
    assert {"channel", "functional_cluster", "PC1", "PC2", "PC3"} <= set(
        scores_frame.columns
    )
