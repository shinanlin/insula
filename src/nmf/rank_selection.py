"""Electrode-subsample bootstrap consensus for concat-NMF rank selection.

Does not use subject split-half or anatomy/behavior external criteria.
Primary decision metric: cophenetic correlation of the consensus matrix.

Outputs are written flat under ``results/nmf/`` and ``img/nmf/`` (SVG only).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, leaves_list, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.nmf.waveform_analysis import (
    PHASE_WINDOWS_POSTONSET,
    TASKS,
    concatenated_phase_matrix,
    discover_paths,
    fit_one_nmf,
    load_hga_rows,
    prepare_shape_matrix,
    restrict_windows,
)
from src.paths import (
    RESULTS_ROOT,
    img_dir,
    nmf_exclude_channels_path,
    nmf_results_dir,
    save_svg,
)


DEFAULT_K_MIN = 2
DEFAULT_K_MAX = 6
DEFAULT_B = 200
DEFAULT_ROW_FRAC = 0.8
DEFAULT_MAX_ITER = 5000
DEFAULT_RANDOM_STATE = 42
COPHENETIC_NEAR_TIE = 0.02
MAX_ARI_PAIRS = 500


def _load_exclude_channels(path: Path | None) -> set[str]:
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


def build_concat_matrix(
    *,
    results_root: Path = RESULTS_ROOT,
    tasks: tuple[str, ...] = TASKS,
    exclude_subjects: set[str] | None = None,
    exclude_channels: set[str] | None = None,
    windows: dict[str, tuple[float, float]] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load Repeat HGA and build post-onset concat shape matrix X."""

    exclude = exclude_subjects if exclude_subjects is not None else {"D0121"}
    win = windows if windows is not None else PHASE_WINDOWS_POSTONSET
    paths = discover_paths(results_root, tasks)
    print(f"Loading HGA from {len(paths)} files ({len(tasks)} tasks)...", flush=True)
    rows = load_hga_rows(
        paths,
        phases=set(win),
        exclude_subjects=exclude,
    )
    drop = exclude_channels or set()
    if drop:
        before = rows["channel"].nunique()
        rows = rows.loc[~rows["channel"].astype(str).isin(drop)].copy()
        after = rows["channel"].nunique()
        print(
            f"Excluded {len(drop)} channels ({before} → {after} unique in HGA rows)",
            flush=True,
        )
    cropped = restrict_windows(rows, windows=win)
    raw_mat, meta, _slices = concatenated_phase_matrix(cropped, tuple(win))
    # Keep channel names as a column (not only index).
    if "channel" not in meta.columns:
        meta = meta.reset_index()
    X, keep = prepare_shape_matrix(raw_mat.to_numpy())
    meta = meta.iloc[np.flatnonzero(keep)].copy()
    if "channel" not in meta.columns:
        meta = meta.reset_index()
    else:
        meta = meta.reset_index(drop=True)
    print(
        f"Concat matrix: {X.shape[0]} electrodes × {X.shape[1]} features",
        flush=True,
    )
    return X, meta


def _accumulate_consensus(
    labels_list: list[np.ndarray],
    index_list: list[np.ndarray],
    n: int,
) -> np.ndarray:
    """Vectorized co-clustering consensus matrix."""

    co = np.zeros((n, n), dtype=np.float64)
    together = np.zeros((n, n), dtype=np.float64)
    for labels, idx in zip(labels_list, index_list, strict=True):
        idx_arr = np.asarray(idx, dtype=int)
        lab_arr = np.asarray(labels, dtype=int)
        together[np.ix_(idx_arr, idx_arr)] += 1.0
        same = (lab_arr[:, None] == lab_arr[None, :]).astype(np.float64)
        co[np.ix_(idx_arr, idx_arr)] += same
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.divide(co, together, out=np.zeros_like(co), where=together > 0)
    np.fill_diagonal(C, 1.0)
    return C


def cophenetic_correlation(C: np.ndarray) -> float:
    """Cophenetic correlation for consensus similarity matrix C."""

    D = np.clip(1.0 - C, 0.0, None)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    condensed = squareform(D, checks=False)
    if not np.isfinite(condensed).all() or np.allclose(condensed, 0.0):
        return float("nan")
    Z = linkage(condensed, method="average")
    corr, _ = cophenet(Z, condensed)
    return float(corr)


def _align_labels(ref: np.ndarray, other: np.ndarray, k: int) -> np.ndarray:
    """Hungarian-align ``other`` labels to ``ref`` cluster ids."""

    contingency = np.zeros((k, k), dtype=np.int64)
    for r, o in zip(ref, other, strict=True):
        contingency[int(r), int(o)] += 1
    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = {int(c): int(r) for r, c in zip(row_ind, col_ind, strict=True)}
    return np.asarray([mapping[int(x)] for x in other], dtype=int)


def mean_pairwise_ari(
    labels_list: list[np.ndarray],
    index_list: list[np.ndarray],
    *,
    k: int,
    random_state: int,
    max_pairs: int = MAX_ARI_PAIRS,
) -> float:
    """Mean ARI over bootstrap pairs on shared electrodes (labels Hungarian-aligned)."""

    n_runs = len(labels_list)
    if n_runs < 2:
        return float("nan")
    pairs = list(itertools.combinations(range(n_runs), 2))
    rng = np.random.default_rng(random_state)
    if len(pairs) > max_pairs:
        pick = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in pick]
    scores: list[float] = []
    for i, j in pairs:
        shared = np.intersect1d(index_list[i], index_list[j], assume_unique=False)
        if shared.size < max(k + 1, 3):
            continue
        pos_i = {int(v): t for t, v in enumerate(index_list[i])}
        pos_j = {int(v): t for t, v in enumerate(index_list[j])}
        lab_i = np.asarray([labels_list[i][pos_i[int(s)]] for s in shared])
        lab_j = np.asarray([labels_list[j][pos_j[int(s)]] for s in shared])
        lab_j = _align_labels(lab_i, lab_j, k)
        scores.append(float(adjusted_rand_score(lab_i, lab_j)))
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def choose_k(metrics: pd.DataFrame) -> dict[str, object]:
    """Pick k by max cophenetic; ties → smaller k. Flag near-ties Δ<0.02."""

    if metrics.empty:
        raise ValueError("Empty metrics; cannot choose k")
    ordered = metrics.sort_values(["cophenetic", "k"], ascending=[False, True])
    best = ordered.iloc[0]
    chosen = int(best["k"])
    near_tie_ks: list[int] = []
    best_coph = float(best["cophenetic"])
    for _, row in metrics.iterrows():
        if abs(float(row["cophenetic"]) - best_coph) < COPHENETIC_NEAR_TIE:
            near_tie_ks.append(int(row["k"]))
    near_tie_ks = sorted(set(near_tie_ks))
    return {
        "k": chosen,
        "rule": "max_cophenetic",
        "tie_break": "smaller_k",
        "cophenetic": best_coph,
        "near_tie_delta": COPHENETIC_NEAR_TIE,
        "near_tie_ks": near_tie_ks,
    }


def bootstrap_consensus_for_k(
    X: np.ndarray,
    *,
    k: int,
    n_boot: int,
    row_frac: float,
    max_iter: int,
    random_state: int,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Return consensus C and per-bootstrap labels/indices for one k."""

    n = X.shape[0]
    n_sample = max(k + 1, int(round(row_frac * n)))
    if n_sample >= n:
        n_sample = n - 1 if n > k + 1 else n
    rng = np.random.default_rng(random_state + 1000 * k)
    labels_list: list[np.ndarray] = []
    index_list: list[np.ndarray] = []
    for b in range(n_boot):
        idx = rng.choice(n, size=n_sample, replace=False)
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        fit = fit_one_nmf(X[idx], k, seed, max_iter=max_iter)
        labels_list.append(np.asarray(fit["labels"], dtype=int))
        index_list.append(np.asarray(idx, dtype=int))
        if (b + 1) % 50 == 0 or b + 1 == n_boot:
            print(f"  k={k}: bootstrap {b + 1}/{n_boot}", flush=True)
    C = _accumulate_consensus(labels_list, index_list, n)
    return C, labels_list, index_list


def full_matrix_secondary_metrics(
    X: np.ndarray, *, k: int, max_iter: int, random_state: int
) -> dict[str, float]:
    """Reconstruction / explained energy / silhouette on the full matrix."""

    fit = fit_one_nmf(X, k, random_state, max_iter=max_iter)
    W = np.asarray(fit["W"])
    H = np.asarray(fit["H"])
    recon = W @ H
    resid_ss = float(np.square(X - recon).sum())
    total_ss = float(np.square(X).sum())
    explained = 1.0 - resid_ss / total_ss
    labels = np.asarray(fit["labels"])
    counts = np.bincount(labels, minlength=k)
    if k > 1 and np.all(counts > 0):
        sil = float(silhouette_score(X, labels, metric="cosine"))
    else:
        sil = float("nan")
    return {
        "reconstruction_error": float(fit["error"]),
        "explained_energy": explained,
        "silhouette_cosine": sil,
    }


def run_rank_selection(
    X: np.ndarray,
    *,
    k_min: int = DEFAULT_K_MIN,
    k_max: int = DEFAULT_K_MAX,
    n_boot: int = DEFAULT_B,
    row_frac: float = DEFAULT_ROW_FRAC,
    max_iter: int = DEFAULT_MAX_ITER,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[int, np.ndarray], dict[str, object]]:
    """Run bootstrap consensus for k_min..k_max and choose k."""

    consensus: dict[int, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for k in range(k_min, k_max + 1):
        print(f"Consensus bootstrap for k={k}...", flush=True)
        C, labels_list, index_list = bootstrap_consensus_for_k(
            X,
            k=k,
            n_boot=n_boot,
            row_frac=row_frac,
            max_iter=max_iter,
            random_state=random_state,
        )
        consensus[k] = C
        coph = cophenetic_correlation(C)
        ari = mean_pairwise_ari(
            labels_list,
            index_list,
            k=k,
            random_state=random_state + k,
        )
        secondary = full_matrix_secondary_metrics(
            X, k=k, max_iter=max_iter, random_state=random_state
        )
        rows.append(
            {
                "k": k,
                "cophenetic": coph,
                "mean_ari": ari,
                "reconstruction_error": secondary["reconstruction_error"],
                "explained_energy": secondary["explained_energy"],
                "silhouette_cosine": secondary["silhouette_cosine"],
                "n_boot": n_boot,
                "row_frac": row_frac,
            }
        )
        print(
            f"  k={k}: cophenetic={coph:.4f} mean_ARI={ari:.4f} "
            f"recon={secondary['reconstruction_error']:.4f} "
            f"explained={secondary['explained_energy']:.4f} "
            f"sil={secondary['silhouette_cosine']:.4f}",
            flush=True,
        )
    metrics = pd.DataFrame(rows)
    decision = choose_k(metrics)
    print(
        f"Chosen k={decision['k']} (cophenetic={decision['cophenetic']:.4f}; "
        f"near_tie_ks={decision['near_tie_ks']})",
        flush=True,
    )
    return metrics, consensus, decision


def plot_metric_curve(
    metrics: pd.DataFrame, column: str, ylabel: str, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(metrics["k"], metrics[column], "o-", color="#333333")
    ax.set_xlabel("k")
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(metrics["k"]))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_svg(fig, path, close=True)


def plot_rank_metrics_overview(metrics: pd.DataFrame, path: Path) -> None:
    """Four-panel SVG: cophenetic, ARI, reconstruction, explained energy."""

    panels = (
        ("cophenetic", "Cophenetic"),
        ("mean_ari", "Mean bootstrap ARI"),
        ("reconstruction_error", "Reconstruction error"),
        ("explained_energy", "Explained energy"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    for ax, (col, ylabel) in zip(axes.ravel(), panels):
        ax.plot(metrics["k"], metrics[col], "o-", color="#333333", lw=1.4)
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(metrics["k"]))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.9", linewidth=0.6)
    axes[1, 0].set_xlabel("k")
    axes[1, 1].set_xlabel("k")
    fig.suptitle("Concat-NMF rank selection metrics", fontsize=11, y=1.01)
    fig.tight_layout()
    save_svg(fig, path, close=True)


def plot_consensus(C: np.ndarray, path: Path, *, k: int) -> None:
    D = np.clip(1.0 - C, 0.0, None)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    condensed = squareform(D, checks=False)
    if np.isfinite(condensed).all() and not np.allclose(condensed, 0.0):
        Z = linkage(condensed, method="average")
        order = leaves_list(Z)
        C_plot = C[np.ix_(order, order)]
    else:
        C_plot = C
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(C_plot, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_title(f"Consensus k={k}")
    ax.set_xlabel("electrode (reordered)")
    ax.set_ylabel("electrode (reordered)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_svg(fig, path, close=True)


def write_rank_selection_outputs(
    *,
    metrics: pd.DataFrame,
    consensus: dict[int, np.ndarray],
    decision: dict[str, object],
    meta: dict[str, object],
    results_dir: Path,
    images_dir: Path,
) -> None:
    """Write flat artifacts under ``results/nmf/`` and ``img/nmf/``."""

    results_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(results_dir / "rank_selection_metrics.csv", index=False)
    (results_dir / "chosen_k.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / "rank_selection_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    for k, C in consensus.items():
        np.save(results_dir / f"consensus_k{k}.npy", C)
        plot_consensus(C, images_dir / f"consensus_k{k}.svg", k=k)
    plot_rank_metrics_overview(metrics, images_dir / "rank_metrics.svg")
    plot_metric_curve(
        metrics, "cophenetic", "Cophenetic correlation", images_dir / "cophenetic_vs_k.svg"
    )
    plot_metric_curve(
        metrics, "mean_ari", "Mean bootstrap ARI", images_dir / "mean_ari_vs_k.svg"
    )
    plot_metric_curve(
        metrics,
        "reconstruction_error",
        "Reconstruction error",
        images_dir / "reconstruction_vs_k.svg",
    )
    plot_metric_curve(
        metrics,
        "explained_energy",
        "Explained energy",
        images_dir / "explained_energy_vs_k.svg",
    )
    print(f"Wrote rank selection → {results_dir}", flush=True)
    print(f"Wrote figures → {images_dir}", flush=True)


def default_output_dirs() -> tuple[Path, Path]:
    """Flat canonical dirs: ``results/nmf/`` and ``img/nmf/``."""

    return nmf_results_dir(), img_dir("nmf")


def run(
    *,
    results_root: Path = RESULTS_ROOT,
    tasks: tuple[str, ...] = TASKS,
    exclude_subjects: set[str] | None = None,
    exclude_channels_file: Path | None = None,
    k_min: int = DEFAULT_K_MIN,
    k_max: int = DEFAULT_K_MAX,
    n_boot: int = DEFAULT_B,
    row_frac: float = DEFAULT_ROW_FRAC,
    max_iter: int = DEFAULT_MAX_ITER,
    random_state: int = DEFAULT_RANDOM_STATE,
    results_dir: Path | None = None,
    images_dir: Path | None = None,
) -> dict[str, object]:
    """End-to-end rank selection: build X, bootstrap, write flat outputs."""

    import matplotlib

    matplotlib.use("Agg")
    plt.rcParams["svg.fonttype"] = "none"

    exclude_path = (
        Path(exclude_channels_file)
        if exclude_channels_file is not None
        else nmf_exclude_channels_path()
    )
    drop_channels = _load_exclude_channels(exclude_path)

    X, channel_meta = build_concat_matrix(
        results_root=results_root,
        tasks=tasks,
        exclude_subjects=exclude_subjects,
        exclude_channels=drop_channels,
    )
    metrics, consensus, decision = run_rank_selection(
        X,
        k_min=k_min,
        k_max=k_max,
        n_boot=n_boot,
        row_frac=row_frac,
        max_iter=max_iter,
        random_state=random_state,
    )
    out_results, out_images = default_output_dirs()
    if results_dir is not None:
        out_results = results_dir
    if images_dir is not None:
        out_images = images_dir
    channel_col = (
        "channel"
        if "channel" in channel_meta.columns
        else channel_meta.columns[0]
    )
    meta = {
        "tasks": list(tasks),
        "windows": {
            phase: list(bounds) for phase, bounds in PHASE_WINDOWS_POSTONSET.items()
        },
        "n_electrodes": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_boot": n_boot,
        "row_frac": row_frac,
        "k_min": k_min,
        "k_max": k_max,
        "max_iter": max_iter,
        "random_state": random_state,
        "exclude_subjects": sorted(exclude_subjects or {"D0121"}),
        "exclude_channels_file": str(exclude_path) if drop_channels else None,
        "n_exclude_channels": len(drop_channels),
        "channels": channel_meta[channel_col].astype(str).tolist(),
    }
    write_rank_selection_outputs(
        metrics=metrics,
        consensus=consensus,
        decision=decision,
        meta=meta,
        results_dir=out_results,
        images_dir=out_images,
    )
    return decision
