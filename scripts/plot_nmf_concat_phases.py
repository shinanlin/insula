#!/usr/bin/env python3
"""Concatenated multi-phase NMF: one shared electrode clustering across epochs.

Builds channel × [stimulus|delay|go|response] features (intersection of channels
present in all phases), rectifies and L2-normalizes the full concatenated
vector, fits NMF, and names components from early−late shape on the *stimulus
segment* of H (not by matching separate per-phase fits).

Avoids the old notebook pitfalls (per-epoch demean + global minimum shift).
Default outputs: tables under ``results/nmf/``, SVG figures under ``img/nmf/``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.nmf.waveform_analysis import (
    CLUSTER_ORDER,
    FUNCTION_COLORS,
    PHASE_WINDOWS,
    PHASE_WINDOWS_POSTONSET,
    TASKS,
    concatenated_phase_matrix,
    discover_paths,
    fit_nmf_grid,
    load_hga_rows,
    ordered_clusters,
    orient_components_on_phase_segment,
    plot_spatial,
    plot_waveforms,
    prepare_shape_matrix,
    restrict_windows,
    split_concat_components,
    summarize_held_out,
)
from src.paths import (
    RESULTS_ROOT,
    img_dir,
    nmf_exclude_channels_path,
    nmf_results_dir,
    save_svg,
)


DISCOVERY_PHASES = tuple(PHASE_WINDOWS)
WINDOW_PRESETS: dict[str, dict[str, tuple[float, float]]] = {
    "default": PHASE_WINDOWS,
    "postonset": PHASE_WINDOWS_POSTONSET,
}


def concat_dirs(
    k: int,
    hemi: str | None = None,
    *,
    run_tag: str | None = None,
) -> tuple[Path, Path]:
    """Flat output dirs: ``results/nmf/`` and ``img/nmf/``."""

    del k, hemi, run_tag  # kept for call-site compatibility
    results = nmf_results_dir()
    images = img_dir("nmf")
    results.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    return results, images


def hemi_scope_label(hemi: str | None) -> str:
    if hemi is None:
        return "bilateral (L+R pooled)"
    return f"hemi={hemi} only"


def windows_label(windows: dict[str, tuple[float, float]]) -> str:
    return "; ".join(f"{p}=({a:g},{b:g})" for p, (a, b) in windows.items())


def stimulus_only_reference(k: int) -> pd.Series | None:
    base = nmf_results_dir() if k == 2 else nmf_results_dir() / f"k{k}"
    path = base / "channel_assignments.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path).set_index("channel")["functional_cluster"]


def write_h_long(
    H_by_phase: dict[str, tuple[np.ndarray, np.ndarray]],
    names: dict[int, str],
    path: Path,
) -> None:
    rows = []
    for phase, (H, times) in H_by_phase.items():
        for component in range(H.shape[0]):
            rows.extend(
                {
                    "component": component,
                    "functional_cluster": names[component],
                    "phase": phase,
                    "time": float(t),
                    "normalized_H": float(v),
                }
                for t, v in zip(times, H[component])
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_h_overview(
    H_by_phase: dict[str, tuple[np.ndarray, np.ndarray]],
    names: dict[int, str],
    *,
    k: int,
    path: Path,
    hemi: str | None = None,
) -> None:
    """One panel per phase; all components overlaid (waveforms-style)."""

    clusters = ordered_clusters(list(names.values()))
    fig, axes = plt.subplots(
        1,
        len(DISCOVERY_PHASES),
        figsize=(3.0 * len(DISCOVERY_PHASES), 3.0),
        sharey=True,
        squeeze=False,
    )
    for col, phase in enumerate(DISCOVERY_PHASES):
        axis = axes[0, col]
        H, times = H_by_phase[phase]
        for cluster in clusters:
            component = next(idx for idx, name in names.items() if name == cluster)
            axis.plot(
                times,
                H[component],
                color=FUNCTION_COLORS.get(cluster, "0.35"),
                linewidth=1.5,
                label=cluster,
            )
        axis.axvline(0, color="0.25", linestyle="--", linewidth=0.6)
        axis.axhline(0, color="0.6", linewidth=0.4)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_title(phase.capitalize(), fontsize=8)
        axis.set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("H (norm.)")
    axes[0, -1].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle(
        f"Concat-NMF H by phase (k={k}; {hemi_scope_label(hemi)}); "
        "names from stimulus-segment early−late",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_svg(fig, path.with_suffix(".svg"), close=True)


def run_for_k(
    *,
    k: int,
    held_out_rows: pd.DataFrame,
    raw_mat: pd.DataFrame,
    meta: pd.DataFrame,
    phase_slices: dict[str, slice],
    X: np.ndarray,
    metrics: pd.DataFrame,
    fits: dict[int, dict[str, object]],
    hemi: str | None = None,
    run_tag: str | None = None,
    windows: dict[str, tuple[float, float]] | None = None,
) -> None:
    windows = PHASE_WINDOWS if windows is None else windows
    results_dir, img_out = concat_dirs(k, hemi=hemi, run_tag=run_tag)
    print(
        f"\n=== Concatenated multi-phase NMF "
        f"(k={k}; {hemi_scope_label(hemi)}; windows={windows_label(windows)}) ===",
        flush=True,
    )
    if k not in fits:
        raise ValueError(f"Need at least {k} electrodes, got {X.shape[0]}")
    fit = fits[k]
    W = np.asarray(fit["W"])
    H = np.asarray(fit["H"])
    labels = np.asarray(fit["labels"], dtype=int)
    sil = float(metrics.loc[metrics["k"] == k, "silhouette_cosine"].iloc[0])
    metrics.to_csv(results_dir / "model_selection_metrics.csv", index=False)

    names = orient_components_on_phase_segment(
        H, raw_mat.columns, phase_slices, name_phase="stimulus"
    )
    H_by_phase = split_concat_components(H, raw_mat.columns, phase_slices)

    assign = meta.reset_index().copy()
    assign["component"] = labels
    assign["functional_cluster"] = np.array([names[int(c)] for c in labels])
    for idx, name in names.items():
        assign[f"loading_{name}"] = W[:, idx]
    assign["dominance"] = W.max(axis=1) / np.maximum(W.sum(axis=1), 1e-12)
    assign.to_csv(results_dir / "channel_assignments.csv", index=False)
    write_h_long(H_by_phase, names, results_dir / "H_by_phase.csv")

    ref = stimulus_only_reference(k)
    if ref is not None:
        shared = assign.set_index("channel").index.intersection(ref.index)
        ari = float(
            adjusted_rand_score(
                ref.loc[shared].to_numpy(),
                assign.set_index("channel").loc[shared, "functional_cluster"].to_numpy(),
            )
        )
    else:
        ari = float("nan")
        shared = []

    cluster_index = ordered_clusters(assign["functional_cluster"])
    crosstab = pd.crosstab(assign["functional_cluster"], assign["roi"]).reindex(
        index=cluster_index, columns=["AIC", "PIC"], fill_value=0
    )
    summary: dict[str, object] = {
        "k": k,
        "hemi": hemi if hemi is not None else "both",
        "run_tag": run_tag if run_tag is not None else "default",
        "windows": windows_label(windows),
        "n_electrodes": len(assign),
        "n_features": int(X.shape[1]),
        "n_AIC": int((assign["roi"] == "AIC").sum()),
        "n_PIC": int((assign["roi"] == "PIC").sum()),
        "n_L": int((assign["hemi"] == "L").sum()),
        "n_R": int((assign["hemi"] == "R").sum()),
        "silhouette_cosine": sil,
        "ari_vs_stimulus_only_nmf": ari,
        "n_shared_vs_stimulus_only": len(shared),
        "naming": "stimulus_segment_early_late",
        "phases": "|".join(windows),
    }
    for name in CLUSTER_ORDER:
        if name in crosstab.index:
            subset = assign.loc[assign["functional_cluster"] == name]
            summary[f"{name}_AIC"] = int(crosstab.loc[name, "AIC"])
            summary[f"{name}_PIC"] = int(crosstab.loc[name, "PIC"])
            summary[f"{name}_y_median"] = float(subset["y"].median())
    pd.DataFrame([summary]).to_csv(results_dir / "concat_nmf_summary.csv", index=False)
    print(
        f"  n={len(assign)} sil={sil:.3f} ARI_vs_stimulus_only={ari:.3f}",
        flush=True,
    )
    print("  cluster × ROI:", flush=True)
    print(crosstab.to_string(), flush=True)

    plot_h_overview(
        H_by_phase, names, k=k, path=img_out / "H_overview", hemi=hemi
    )

    fig = plot_spatial(assign)
    fig.suptitle(
        f"Concat-NMF k={k}; {hemi_scope_label(hemi)}; "
        f"windows={windows_label(windows)}  "
        f"(n={len(assign)}, sil={sil:.2f})",
        fontsize=8,
        y=1.02,
    )
    fig.tight_layout()
    save_svg(fig, img_out / "spatial_yz.svg", close=True)

    # In-sample means on the same windows used for discovery.
    waveform_summary, coverage = summarize_held_out(
        held_out_rows, assign, windows=windows
    )
    waveform_summary.to_csv(results_dir / "waveform_summary.csv", index=False)
    coverage.to_csv(results_dir / "waveform_coverage.csv", index=False)
    fig = plot_waveforms(waveform_summary, coverage)
    fig.suptitle(
        f"In-sample waveforms for concat-NMF clusters "
        f"(k={k}; {hemi_scope_label(hemi)}; windows={windows_label(windows)})",
        fontsize=8,
        y=1.04,
    )
    fig.tight_layout()
    save_svg(fig, img_out / "waveforms.svg", close=True)

    manifest = {
        "published_at": datetime.now(timezone.utc).isoformat(),
        "assignments": str(results_dir / "channel_assignments.csv"),
        "construction": "concat_phases",
        "windows": windows_label(windows),
        "k": k,
        "tasks": list(TASKS),
        "condition": "Repeat",
        "n_electrodes": len(assign),
        "cluster_names": list(CLUSTER_ORDER),
    }
    (results_dir / "nmf_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote tables → {results_dir.resolve()}", flush=True)
    print(f"Wrote SVGs → {img_out.resolve()}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument(
        "--exclude-subject",
        action="append",
        default=["D0121"],
        help="Repeatable; D0121 matches vizpub/fig2.ipynb",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        choices=(2, 3, 4, 5, 6),
        help="Export labeled figures for this k (default: 3)",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=3,
        help="Fit NMF for k=1..k_max (default: 3)",
    )
    parser.add_argument(
        "--hemi",
        choices=("both", "L", "R"),
        default="both",
        help="Electrode pool: both, or L/R only",
    )
    parser.add_argument(
        "--windows",
        choices=tuple(WINDOW_PRESETS),
        default="postonset",
        help="Phase time crops (default: postonset)",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Ignored; outputs always go to results/nmf and img/nmf",
    )
    parser.add_argument(
        "--exclude-channels-file",
        type=Path,
        default=None,
        help=(
            "Text file (one channel per line) or CSV with a channel column. "
            f"Default: {nmf_exclude_channels_path()}"
        ),
    )
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _load_exclude_channels(path: Path | None) -> set[str]:
    if path is None:
        return set()
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        if "channel" not in frame.columns:
            raise SystemExit(f"{path} has no 'channel' column")
        return set(frame["channel"].astype(str))
    return {line.strip() for line in text.splitlines() if line.strip()}


def run_hemi_scope(
    *,
    hemi: str | None,
    held_out_rows: pd.DataFrame,
    ks: tuple[int, ...],
    k_max: int,
    n_init: int,
    max_iter: int,
    random_state: int,
    windows: dict[str, tuple[float, float]],
    run_tag: str | None,
) -> None:
    scope = held_out_rows
    if hemi is not None:
        scope = held_out_rows.loc[held_out_rows["hemi"].eq(hemi)].copy()
        if scope.empty:
            raise ValueError(f"No HGA rows for hemi={hemi!r}")

    phases = tuple(windows)
    all_phase_rows = restrict_windows(scope, windows=windows)
    raw_mat, meta, phase_slices = concatenated_phase_matrix(
        all_phase_rows, phases
    )
    X, keep = prepare_shape_matrix(raw_mat.to_numpy())
    raw_mat = raw_mat.iloc[np.flatnonzero(keep)]
    meta = meta.iloc[np.flatnonzero(keep)]
    print(
        f"Concat matrix ({hemi_scope_label(hemi)}; {windows_label(windows)}): "
        f"{X.shape[0]} electrodes × {X.shape[1]} features; "
        f"AIC={int(meta['roi'].eq('AIC').sum())}, "
        f"PIC={int(meta['roi'].eq('PIC').sum())}; "
        f"L={int(meta['hemi'].eq('L').sum())}, "
        f"R={int(meta['hemi'].eq('R').sum())}",
        flush=True,
    )

    print(
        f"Fitting NMF grid k=1..{k_max} ({hemi_scope_label(hemi)})...",
        flush=True,
    )
    metrics, fits = fit_nmf_grid(
        X,
        k_max=k_max,
        n_init=n_init,
        random_state=random_state,
        max_iter=max_iter,
    )
    shared_results = nmf_results_dir()
    shared_results.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(shared_results / "model_selection_metrics.csv", index=False)
    print(
        f"Wrote model selection metrics → "
        f"{shared_results / 'model_selection_metrics.csv'}",
        flush=True,
    )

    for k in ks:
        run_for_k(
            k=k,
            held_out_rows=scope,
            raw_mat=raw_mat,
            meta=meta,
            phase_slices=phase_slices,
            X=X,
            metrics=metrics,
            fits=fits,
            hemi=hemi,
            run_tag=run_tag,
            windows=windows,
        )


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    plt.rcParams["svg.fonttype"] = "none"

    args = parse_args()
    all_paths = discover_paths(args.results_root, tuple(args.tasks))
    exclude = set(args.exclude_subject)
    ks = (args.k,)
    k_max = max(args.k_max, max(ks))
    hemi = None if args.hemi == "both" else args.hemi
    windows = WINDOW_PRESETS[args.windows]
    run_tag = args.run_tag

    print("Loading all-phase HGA rows...", flush=True)
    held_out_rows = load_hga_rows(
        all_paths,
        phases=set(PHASE_WINDOWS),
        exclude_subjects=exclude,
    )
    exclude_path = args.exclude_channels_file
    if exclude_path is None:
        exclude_path = nmf_exclude_channels_path()
    drop_channels = _load_exclude_channels(
        exclude_path if exclude_path.is_file() else None
    )
    if drop_channels:
        before = held_out_rows["channel"].nunique()
        held_out_rows = held_out_rows.loc[
            ~held_out_rows["channel"].astype(str).isin(drop_channels)
        ].copy()
        after = held_out_rows["channel"].nunique()
        print(
            f"Excluded {len(drop_channels)} channels from {exclude_path} "
            f"({before} → {after} unique channels in HGA rows)",
            flush=True,
        )
    elif args.exclude_channels_file is None:
        print(
            f"No exclude file at {nmf_exclude_channels_path()}; "
            "using all loaded channels",
            flush=True,
        )
    run_hemi_scope(
        hemi=hemi,
        held_out_rows=held_out_rows,
        ks=ks,
        k_max=k_max,
        n_init=args.n_init,
        max_iter=args.max_iter,
        random_state=args.random_state,
        windows=windows,
        run_tag=run_tag,
    )


if __name__ == "__main__":
    main()
