"""PCA of the concat-NMF waveform matrix, colored by frozen cluster labels.

Fits PCA on the same post-onset concat shape matrix ``X`` used for NMF
(row L2-normalized HGA). Cluster colors come from
``channel_assignments.csv`` and are not re-derived from the PCs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.nmf.rank_selection import build_concat_matrix
from src.nmf.waveform_analysis import (
    FUNCTION_COLORS,
    TASKS,
    ordered_clusters,
)
from src.paths import (
    RESULTS_ROOT,
    img_dir,
    nmf_assignments_path,
    nmf_exclude_channels_path,
    nmf_results_dir,
    save_svg,
)


def load_exclude_channels(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        if "channel" not in frame.columns:
            raise ValueError(f"{path} has no 'channel' column")
        return set(frame["channel"].astype(str))
    return {line.strip() for line in text.splitlines() if line.strip()}


def align_matrix_to_assignments(
    meta: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return row indices into ``X``/``meta`` and the aligned assignment rows."""

    if "channel" not in meta.columns:
        raise ValueError("concat meta must include a 'channel' column")
    required = {"channel", "functional_cluster"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"assignments missing columns: {sorted(missing)}")

    meta_channels = meta["channel"].astype(str)
    assign = assignments.copy()
    assign["channel"] = assign["channel"].astype(str)
    if assign["channel"].duplicated().any():
        raise ValueError("assignments contain duplicate channel names")

    assign_by_channel = assign.set_index("channel", verify_integrity=True)
    shared = meta_channels.isin(assign_by_channel.index)
    idx = np.flatnonzero(shared.to_numpy())
    if idx.size == 0:
        raise ValueError("No shared channels between concat matrix and assignments")
    aligned = assign_by_channel.loc[meta_channels.iloc[idx]].reset_index()
    return idx, aligned


def fit_waveform_pca(
    X: np.ndarray, *, n_components: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Column-centered PCA of the concat waveform matrix.

    ``X`` is the same row-L2 shape matrix used for NMF. sklearn PCA centers
    features; rows are not re-standardized.
    """

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    n_comp = min(int(n_components), X.shape[0], X.shape[1])
    if n_comp < 2:
        raise ValueError(f"Need at least 2 PCA components, got {n_comp}")
    pca = PCA(n_components=n_comp, svd_solver="full")
    scores = pca.fit_transform(X)
    return scores, np.asarray(pca.explained_variance_ratio_, dtype=float)


def plot_waveform_pca(
    scores: np.ndarray,
    labels: pd.Series | np.ndarray,
    explained: np.ndarray,
    *,
    dominance: np.ndarray | None = None,
    path: Path | None = None,
) -> plt.Figure:
    """Scatter electrodes in PC1–PC2 and PC1–PC3, colored by NMF labels."""

    scores = np.asarray(scores, dtype=float)
    labels = pd.Series(labels, dtype=str)
    if scores.shape[0] != len(labels):
        raise ValueError("scores and labels must have the same length")
    if scores.shape[1] < 2:
        raise ValueError("Need at least 2 PC scores to plot")

    explained = np.asarray(explained, dtype=float)
    n_panels = 2 if scores.shape[1] >= 3 else 1
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(3.6 * n_panels, 3.4),
        squeeze=False,
    )
    pairs = ((0, 1), (0, 2))[:n_panels]
    clusters = ordered_clusters(labels)
    if dominance is None:
        alpha = np.full(len(labels), 0.85)
    else:
        dominance = np.asarray(dominance, dtype=float)
        alpha = np.clip(0.30 + 0.65 * dominance, 0.25, 0.95)

    for axis, (i, j) in zip(axes[0], pairs):
        for cluster in clusters:
            mask = labels.eq(cluster).to_numpy()
            if not mask.any():
                continue
            axis.scatter(
                scores[mask, i],
                scores[mask, j],
                s=22,
                color=FUNCTION_COLORS.get(cluster, "0.35"),
                edgecolor="0.2",
                linewidth=0.35,
                alpha=alpha[mask],
                label=f"{cluster} (n={int(mask.sum())})",
            )
            axis.scatter(
                scores[mask, i].mean(),
                scores[mask, j].mean(),
                s=70,
                marker="X",
                color=FUNCTION_COLORS.get(cluster, "0.35"),
                edgecolor="0.1",
                linewidth=0.6,
                zorder=3,
            )
        axis.set_xlabel(f"PC{i + 1} ({100 * explained[i]:.1f}%)")
        axis.set_ylabel(f"PC{j + 1} ({100 * explained[j]:.1f}%)")
        axis.axhline(0.0, color="0.75", linewidth=0.5)
        axis.axvline(0.0, color="0.75", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle(
        "Concat-NMF labels in waveform PCA space",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()
    if path is not None:
        save_svg(fig, Path(path), close=True)
    return fig


def write_pca_outputs(
    *,
    aligned: pd.DataFrame,
    scores: np.ndarray,
    explained: np.ndarray,
    meta: dict[str, object],
    scores_path: Path,
    meta_path: Path,
) -> None:
    n_comp = scores.shape[1]
    out = aligned[["channel", "functional_cluster"]].copy()
    if "subject" in aligned.columns:
        out.insert(1, "subject", aligned["subject"])
    if "dominance" in aligned.columns:
        out["dominance"] = aligned["dominance"]
    for i in range(n_comp):
        out[f"PC{i + 1}"] = scores[:, i]
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(scores_path, index=False)
    payload = {
        **meta,
        "explained_variance_ratio": [float(v) for v in explained],
        "n_components": int(n_comp),
    }
    meta_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run(
    *,
    results_root: Path = RESULTS_ROOT,
    tasks: tuple[str, ...] = TASKS,
    exclude_subjects: set[str] | None = None,
    exclude_channels_file: Path | None = None,
    assignments_path: Path | None = None,
    n_components: int = 3,
    scores_path: Path | None = None,
    meta_path: Path | None = None,
    svg_path: Path | None = None,
) -> dict[str, object]:
    """Rebuild concat ``X``, align to assignments, write PCA scores and SVG."""

    import matplotlib

    matplotlib.use("Agg")
    plt.rcParams["svg.fonttype"] = "none"

    assign_path = Path(assignments_path or nmf_assignments_path())
    if not assign_path.is_file():
        raise FileNotFoundError(
            f"Missing NMF assignments: {assign_path}. "
            "Run scripts/plot_nmf_concat_phases.py first."
        )
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
    assignments = pd.read_csv(assign_path)
    idx, aligned = align_matrix_to_assignments(channel_meta, assignments)
    X_use = X[idx]
    print(
        f"Aligned {len(aligned)} / {X.shape[0]} concat electrodes "
        f"to {len(assignments)} assignments",
        flush=True,
    )

    scores, explained = fit_waveform_pca(X_use, n_components=n_components)
    print(
        "Explained variance: "
        + ", ".join(f"PC{i + 1}={100 * v:.1f}%" for i, v in enumerate(explained)),
        flush=True,
    )

    results_dir = nmf_results_dir()
    images = img_dir("nmf")
    scores_out = Path(scores_path or results_dir / "waveform_pca_scores.csv")
    meta_out = Path(meta_path or results_dir / "waveform_pca_meta.json")
    svg_out = Path(svg_path or images / "waveform_pca.svg")

    write_pca_outputs(
        aligned=aligned,
        scores=scores,
        explained=explained,
        meta={
            "assignments_path": str(assign_path.resolve()),
            "construction": "concat_phases_postonset",
            "n_concat_electrodes": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_assignments": int(len(assignments)),
            "n_shared": int(len(aligned)),
            "n_unlabeled_in_X": int(X.shape[0] - len(aligned)),
            "exclude_subjects": sorted(exclude_subjects or {"D0121"}),
            "exclude_channels_file": str(exclude_path) if drop_channels else None,
            "n_exclude_channels": len(drop_channels),
            "tasks": list(tasks),
        },
        scores_path=scores_out,
        meta_path=meta_out,
    )
    dominance = (
        aligned["dominance"].to_numpy(dtype=float)
        if "dominance" in aligned.columns
        else None
    )
    plot_waveform_pca(
        scores,
        aligned["functional_cluster"],
        explained,
        dominance=dominance,
        path=svg_out,
    )
    print(f"Wrote scores → {scores_out}", flush=True)
    print(f"Wrote SVG → {svg_out}", flush=True)
    return {
        "n_shared": int(len(aligned)),
        "explained_variance_ratio": [float(v) for v in explained],
        "scores_path": str(scores_out),
        "svg_path": str(svg_out),
    }
