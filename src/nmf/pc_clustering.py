"""Castellucci-style PCA embedding + PC-space clustering (compute only).

Rebuilds the concat-NMF waveform matrix ``X``, writes scree / PC scores /
per-iteration cluster metrics under ``results/nmf/``. Does not write figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids

from src.nmf.rank_selection import build_concat_matrix
from src.nmf.waveform_analysis import TASKS
from src.nmf.waveform_pca import (
    align_matrix_to_assignments,
    fit_waveform_pca,
    load_exclude_channels,
)
from src.paths import (
    RESULTS_ROOT,
    nmf_assignments_path,
    nmf_exclude_channels_path,
    nmf_results_dir,
)


DEFAULT_N_SCREE = 20
DEFAULT_VARIANCE_THRESHOLD = 0.05
DEFAULT_K_MIN = 2
DEFAULT_K_MAX = 10
DEFAULT_N_ITER = 500
DEFAULT_RANDOM_STATE = 42
METHODS = ("kmeans", "kmedoids", "gmm", "ward")
STOCHASTIC_METHODS = frozenset({"kmeans", "kmedoids", "gmm"})


def n_pcs_above_threshold(
    explained: np.ndarray, *, threshold: float = DEFAULT_VARIANCE_THRESHOLD
) -> int:
    """Count leading PCs that each explain more than ``threshold`` variance."""

    explained = np.asarray(explained, dtype=float)
    n = int(np.sum(explained > threshold))
    return max(n, 2) if explained.size >= 2 else int(explained.size)


def _fit_labels(X: np.ndarray, *, method: str, k: int, seed: int) -> np.ndarray:
    if method == "kmeans":
        model = KMeans(n_clusters=k, n_init=1, random_state=seed)
        return np.asarray(model.fit_predict(X), dtype=int)
    if method == "kmedoids":
        model = KMedoids(
            n_clusters=k,
            metric="euclidean",
            method="pam",
            init="random",
            max_iter=300,
            random_state=seed,
        )
        return np.asarray(model.fit_predict(X), dtype=int)
    if method == "gmm":
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=1,
            random_state=seed,
        )
        return np.asarray(model.fit_predict(X), dtype=int)
    if method == "ward":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        return np.asarray(model.fit_predict(X), dtype=int)
    raise ValueError(f"Unknown method {method!r}")


def _safe_scores(X: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(labels)
    counts = np.bincount(labels)
    if labels.min() < 0 or np.unique(labels).size < 2 or np.any(counts[counts > 0] < 2):
        return float("nan"), float("nan")
    return (
        float(silhouette_score(X, labels, metric="euclidean")),
        float(calinski_harabasz_score(X, labels)),
    )


def cluster_pc_space(
    scores: np.ndarray,
    *,
    k_min: int = DEFAULT_K_MIN,
    k_max: int = DEFAULT_K_MAX,
    n_iter: int = DEFAULT_N_ITER,
    random_state: int = DEFAULT_RANDOM_STATE,
    methods: tuple[str, ...] = METHODS,
) -> pd.DataFrame:
    """Sweep clustering methods in PC space; one row per method × k × iteration."""

    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[0] < k_max:
        raise ValueError(
            f"Need at least k_max={k_max} electrodes, got scores {scores.shape}"
        )
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}")
        repeats = 1 if method not in STOCHASTIC_METHODS else n_iter
        print(f"Clustering method={method} ({repeats} run(s))...", flush=True)
        for k in range(k_min, k_max + 1):
            for iteration in range(repeats):
                seed = int(rng.integers(0, np.iinfo(np.int32).max))
                try:
                    labels = _fit_labels(scores, method=method, k=k, seed=seed)
                    sil, ch = _safe_scores(scores, labels)
                except ValueError:
                    sil, ch = float("nan"), float("nan")
                rows.append(
                    {
                        "method": method,
                        "k": k,
                        "iteration": iteration,
                        "silhouette": sil,
                        "calinski_harabasz": ch,
                    }
                )
            done = [row for row in rows if row["method"] == method and row["k"] == k]
            sils = np.asarray([row["silhouette"] for row in done], dtype=float)
            print(
                f"  {method} k={k}: median silhouette="
                f"{np.nanmedian(sils):.4f} (n={len(done)})",
                flush=True,
            )
    return pd.DataFrame(rows)


def summarize_cluster_metrics(iterations: pd.DataFrame) -> pd.DataFrame:
    """Median / IQR of silhouette and C-H by method × k."""

    def _iqr(series: pd.Series) -> float:
        values = series.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return float("nan")
        q75, q25 = np.quantile(values, [0.75, 0.25])
        return float(q75 - q25)

    grouped = iterations.groupby(["method", "k"], sort=True)
    summary = grouped.agg(
        n_runs=("iteration", "count"),
        silhouette_median=("silhouette", "median"),
        silhouette_iqr=("silhouette", _iqr),
        calinski_median=("calinski_harabasz", "median"),
        calinski_iqr=("calinski_harabasz", _iqr),
    ).reset_index()
    return summary


def choose_k_max_metric(summary: pd.DataFrame, column: str) -> dict[str, object]:
    """Pick k by max median metric; ties → smaller k. Uses k-means as primary."""

    kmeans = summary.loc[summary["method"].eq("kmeans")].copy()
    if kmeans.empty:
        kmeans = summary
    ordered = kmeans.sort_values([column, "k"], ascending=[False, True])
    best = ordered.iloc[0]
    return {
        "k": int(best["k"]),
        "method": str(best["method"]),
        "metric": column,
        "value": float(best[column]),
    }


def write_pc_tables(
    *,
    explained: np.ndarray,
    scores: np.ndarray,
    meta_rows: pd.DataFrame,
    iterations: pd.DataFrame,
    run_meta: dict[str, object],
    results_dir: Path,
) -> dict[str, Path]:
    """Write scree, scores, iteration, summary, and meta tables. No figures."""

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    n_comp = scores.shape[1]
    scree = pd.DataFrame(
        {
            "pc": np.arange(1, n_comp + 1),
            "explained_variance_ratio": np.asarray(explained, dtype=float),
            "cumulative_variance_ratio": np.cumsum(explained),
        }
    )
    score_frame = meta_rows.copy()
    for i in range(n_comp):
        score_frame[f"PC{i + 1}"] = scores[:, i]

    summary = summarize_cluster_metrics(iterations)
    sil_choice = choose_k_max_metric(summary, "silhouette_median")
    ch_choice = choose_k_max_metric(summary, "calinski_median")
    payload = {
        **run_meta,
        "k_silhouette": sil_choice,
        "k_calinski": ch_choice,
    }

    paths = {
        "scree": results_dir / "pc_scree.csv",
        "scores": results_dir / "pc_scores.csv",
        "iterations": results_dir / "pc_clustering_iterations.csv",
        "summary": results_dir / "pc_clustering_metrics.csv",
        "meta": results_dir / "pc_clustering_meta.json",
    }
    scree.to_csv(paths["scree"], index=False)
    score_frame.to_csv(paths["scores"], index=False)
    iterations.to_csv(paths["iterations"], index=False)
    summary.to_csv(paths["summary"], index=False)
    paths["meta"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return paths


def run(
    *,
    results_root: Path = RESULTS_ROOT,
    tasks: tuple[str, ...] = TASKS,
    exclude_subjects: set[str] | None = None,
    exclude_channels_file: Path | None = None,
    assignments_path: Path | None = None,
    n_scree: int = DEFAULT_N_SCREE,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
    k_min: int = DEFAULT_K_MIN,
    k_max: int = DEFAULT_K_MAX,
    n_iter: int = DEFAULT_N_ITER,
    random_state: int = DEFAULT_RANDOM_STATE,
    results_dir: Path | None = None,
) -> dict[str, object]:
    """End-to-end compute: build X, PCA, cluster, write ``results/nmf/`` tables."""

    assign_path = Path(assignments_path or nmf_assignments_path())
    exclude_path = (
        Path(exclude_channels_file)
        if exclude_channels_file is not None
        else nmf_exclude_channels_path()
    )
    drop_channels = load_exclude_channels(exclude_path)
    X, channel_meta = build_concat_matrix(
        results_root=results_root,
        tasks=tasks,
        exclude_subjects=exclude_subjects,
        exclude_channels=drop_channels,
    )
    if assign_path.is_file():
        assignments = pd.read_csv(assign_path)
        idx, aligned = align_matrix_to_assignments(channel_meta, assignments)
        X_use = X[idx]
        meta_rows = aligned.copy()
        print(
            f"Aligned {len(aligned)} / {X.shape[0]} concat electrodes "
            f"to {len(assignments)} assignments",
            flush=True,
        )
    else:
        X_use = X
        meta_rows = channel_meta.copy()
        print(
            f"No assignments at {assign_path}; scoring all {X.shape[0]} electrodes",
            flush=True,
        )

    scores_all, explained = fit_waveform_pca(X_use, n_components=n_scree)
    n_embed = n_pcs_above_threshold(explained, threshold=variance_threshold)
    embed = scores_all[:, :n_embed]
    print(
        f"PCA: {scores_all.shape[1]} components; "
        f"{n_embed} PC(s) > {100 * variance_threshold:.0f}% variance "
        + ", ".join(
            f"PC{i + 1}={100 * v:.1f}%" for i, v in enumerate(explained[:n_embed])
        ),
        flush=True,
    )

    iterations = cluster_pc_space(
        embed,
        k_min=k_min,
        k_max=k_max,
        n_iter=n_iter,
        random_state=random_state,
    )
    keep_cols = [
        col
        for col in (
            "channel",
            "subject",
            "functional_cluster",
            "dominance",
            "roi",
            "hemi",
        )
        if col in meta_rows.columns
    ]
    out_dir = Path(results_dir or nmf_results_dir())
    paths = write_pc_tables(
        explained=explained,
        scores=scores_all,
        meta_rows=meta_rows.loc[:, keep_cols],
        iterations=iterations,
        run_meta={
            "construction": "concat_phases_postonset",
            "assignments_path": str(assign_path) if assign_path.is_file() else None,
            "n_electrodes": int(X_use.shape[0]),
            "n_features": int(X_use.shape[1]),
            "n_scree": int(scores_all.shape[1]),
            "n_embedding_pcs": int(n_embed),
            "variance_threshold": float(variance_threshold),
            "k_min": int(k_min),
            "k_max": int(k_max),
            "n_iter": int(n_iter),
            "random_state": int(random_state),
            "methods": list(METHODS),
            "exclude_subjects": sorted(exclude_subjects or {"D0121"}),
            "exclude_channels_file": str(exclude_path) if drop_channels else None,
            "n_exclude_channels": len(drop_channels),
            "tasks": list(tasks),
            "explained_variance_ratio": [float(v) for v in explained],
        },
        results_dir=out_dir,
    )
    for name, path in paths.items():
        print(f"Wrote {name} → {path}", flush=True)
    return {
        "n_electrodes": int(X_use.shape[0]),
        "n_embedding_pcs": int(n_embed),
        "paths": {key: str(path) for key, path in paths.items()},
    }
