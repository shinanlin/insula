"""Decision vs Repeat univariate brains colored by NMF functional cluster."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.nmf.waveform_analysis import FUNCTION_COLORS
from src.paths import PROJECT_ROOT, nmf_assignments_path as canonical_nmf_assignments_path
from src.univariate.viz_mean import (
    BrainSurfaceContext,
    attach_metadata,
    filter_qc,
    filter_insula_electrodes,
    load_coord_metadata,
    load_mean_contrasts,
    resolve_mean_results_root,
    select_significant,
)

logger = logging.getLogger(__name__)

TASK = "LexicalDelay"
CONTRAST = "DecisionVsRepeatMean"
PHASES = ("Stimulus", "Delay", "Go", "Response")
DIRECTIONS = ("Decision", "Repeat")

CLUSTER_COLORS = dict(FUNCTION_COLORS)
CLUSTERS_BY_K = {
    2: ("sustain", "sensory"),
    3: ("sustain", "motor", "sensory"),
}


def _title_cluster_legend(k: int) -> str:
    if k == 3:
        return "gold=sustained; red=motor; blue=sensory"
    return "gold=sustained; blue=sensory"


def _fname_suffix(k: int) -> str:
    return f"_k{k}" if k != 2 else ""


def resolve_nmf_assignments_path(project_root: Path, k: int) -> Path:
    """Resolve published NMF assignments (k kept for API compatibility)."""
    del k  # canonical CSV is always the published concat crop k=3 table
    root = Path(project_root)
    candidates = [
        root / "results" / "nmf" / "channel_assignments.csv",
        canonical_nmf_assignments_path(),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No NMF channel_assignments.csv in {candidates}"
    )


def load_nmf_assignments(
    project_root: Path | None = None,
    *,
    k: int = 2,
) -> pd.DataFrame:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    if k not in CLUSTERS_BY_K:
        raise ValueError(f"k must be one of {sorted(CLUSTERS_BY_K)}, got {k}")
    path = resolve_nmf_assignments_path(root, k)
    clusters = CLUSTERS_BY_K[k]
    df = pd.read_csv(path)
    keep = df["functional_cluster"].isin(clusters)
    cols = ["channel", "functional_cluster", "hemi", "roi", "x", "y", "z"]
    cols = [c for c in cols if c in df.columns]
    out = df.loc[keep, cols].copy()
    logger.info("Loaded NMF k=%d assignments from %s (%d channels)", k, path, len(out))
    return out


def load_decision_vs_repeat_table(
    results_root: Path | None = None,
    *,
    project_root: Path | None = None,
    k: int = 2,
) -> tuple[pd.DataFrame, Path]:
    """Significant AIC/PIC DecisionVsRepeat rows merged with NMF clusters."""
    root = resolve_mean_results_root(results_root)
    mean_df = load_mean_contrasts(root, tasks=(TASK,))
    if mean_df.empty:
        raise FileNotFoundError(f"No mean contrasts loaded from {root}")
    coords = load_coord_metadata(root, tasks=(TASK,))
    df = attach_metadata(mean_df, coords)
    df = filter_qc(df)
    sig = select_significant(df, contrast=CONTRAST, tasks=(TASK,))
    sig = filter_insula_electrodes(sig)
    assignments = load_nmf_assignments(project_root, k=k)
    keep_cols = [
        c
        for c in (
            "channel",
            "subject",
            "phase",
            "contrast",
            "task",
            "mean_diff",
            "direction",
            "significant",
            "p_value",
            "p_fdr",
        )
        if c in sig.columns
    ]
    merged = sig[keep_cols].merge(assignments, on="channel", how="inner")
    if "phase" in merged.columns:
        merged["phase"] = merged["phase"].astype(str)
    logger.info(
        "DecisionVsRepeat cluster table (k=%d): %d significant assigned rows "
        "(%d Decision, %d Repeat) from %s",
        k,
        len(merged),
        int((merged["direction"] == "Decision").sum()),
        int((merged["direction"] == "Repeat").sum()),
        root,
    )
    return merged, root


def union_direction_channels(table: pd.DataFrame, direction: str) -> pd.DataFrame:
    """Union significant channels across all phases for one direction.

    One row per channel; ``mean_diff`` = value with largest |mean_diff| across phases.
    """
    sub = table.loc[table["direction"] == direction].copy()
    if sub.empty:
        return sub
    sub["_abs"] = sub["mean_diff"].abs()
    idx = sub.groupby("channel", sort=False)["_abs"].idxmax()
    out = sub.loc[idx].drop(columns=["_abs"]).reset_index(drop=True)
    return out


def _spatial_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Frame expected by ``render_insula_hemisphere_cluster``."""
    out = df.copy()
    out["pattern"] = out["mean_diff"].abs().astype(float)
    out["significant"] = True
    return out.reset_index(drop=True)


def _save_svg(fig: plt.Figure, path: Path) -> Path:
    path = path.with_suffix(".svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def phase_has_mean_files(results_root: Path, phase: str) -> bool:
    from src.univariate.viz_mean import discover_mean_paths

    return bool(discover_mean_paths(results_root, TASK, phase, CONTRAST))


def _select_direction_phase(
    table: pd.DataFrame,
    *,
    direction: str,
    phase: str,
) -> pd.DataFrame:
    phase_l = phase.lower()
    mask = (table["direction"] == direction) & (table["phase"].str.lower() == phase_l)
    return table.loc[mask].copy()


def plot_direction_union_brain(
    table: pd.DataFrame,
    out_dir: Path,
    *,
    direction: str,
    k: int = 2,
    ctx: BrainSurfaceContext | None = None,
    size_by_effect: bool = False,
) -> Path:
    """Left | Right brain: union of electrodes significant in any phase.

    When ``size_by_effect`` is True, marker size scales with ``|mean_diff|``.
    """
    from src.decoding.viz_insula_patterns import render_insula_hemisphere_cluster

    surface_ctx = ctx or BrainSurfaceContext()
    united = union_direction_channels(table, direction)
    frame = _spatial_frame(united)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
    for ax, (hemi, label) in zip(axes, (("lh", "Left"), ("rh", "Right"))):
        if frame.empty:
            ax.text(0.5, 0.5, "no channels", ha="center", va="center")
            ax.axis("off")
            ax.set_title(label, fontsize=8)
            continue
        img = render_insula_hemisphere_cluster(
            frame, hemi, surface_ctx, size_by_pattern=size_by_effect
        )
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(label, fontsize=8)

    n = len(frame)
    size_note = "; size=|mean_diff|" if size_by_effect else ""
    fig.suptitle(
        f"{TASK} DecisionVsRepeat — {direction} larger, phase union (n={n})  "
        f"({_title_cluster_legend(k)}{size_note})",
        fontsize=9,
    )
    fig.tight_layout()
    fname = (
        f"{TASK}_DecisionVsRepeat_{direction}_union_cluster_brain{_fname_suffix(k)}"
    )
    return _save_svg(fig, out_dir / fname)

def plot_direction_all_phases_brain(
    table: pd.DataFrame,
    results_root: Path,
    out_dir: Path,
    *,
    direction: str,
    k: int = 2,
    ctx: BrainSurfaceContext | None = None,
) -> Path:
    """One figure, 2×4: rows Left/Right, cols Stimulus/Delay/Go/Response."""
    from src.decoding.viz_insula_patterns import render_insula_hemisphere_cluster

    surface_ctx = ctx or BrainSurfaceContext()
    rows_spec = (("lh", "Left"), ("rh", "Right"))
    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(PHASES),
        figsize=(3.0 * len(PHASES), 5.4),
        squeeze=False,
    )
    for row, (hemi, hemi_label) in enumerate(rows_spec):
        for col, phase in enumerate(PHASES):
            ax = axes[row, col]
            available = phase_has_mean_files(results_root, phase)
            frame = _spatial_frame(
                _select_direction_phase(table, direction=direction, phase=phase)
            )
            if not available:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
                ax.axis("off")
            elif frame.empty:
                ax.text(0.5, 0.5, "no channels", ha="center", va="center")
                ax.axis("off")
            else:
                img = render_insula_hemisphere_cluster(frame, hemi, surface_ctx)
                ax.imshow(img)
                ax.axis("off")
            if row == 0:
                ax.set_title(phase, fontsize=7)
            if col == 0:
                ax.set_ylabel(hemi_label, fontsize=7)

    fig.suptitle(
        f"{TASK} DecisionVsRepeat — {direction} larger by phase  "
        f"({_title_cluster_legend(k)})",
        fontsize=8,
    )
    fig.tight_layout()
    fname = (
        f"{TASK}_DecisionVsRepeat_{direction}_allphases_cluster_brain{_fname_suffix(k)}"
    )
    return _save_svg(fig, out_dir / fname)


def run_decision_vs_repeat_cluster_figures(
    *,
    results_root: Path | None = None,
    project_root: Path | None = None,
    out_dir: Path | None = None,
    k: int = 2,
) -> list[Path]:
    """Per direction: (1) phase-union Left|Right, (2) 2×4 all-phases grid.

    Does not write separate per-phase figure files.
    k=2 keeps original filenames; k=3 appends ``_k3``.
    """
    project_root = Path(project_root or PROJECT_ROOT)
    from src.paths import img_dir

    out_dir = Path(out_dir or img_dir("univariate"))
    out_dir.mkdir(parents=True, exist_ok=True)

    table, root = load_decision_vs_repeat_table(
        results_root, project_root=project_root, k=k
    )
    ctx = BrainSurfaceContext()
    saved: list[Path] = []
    for direction in DIRECTIONS:
        saved.append(
            plot_direction_union_brain(
                table, out_dir, direction=direction, k=k, ctx=ctx
            )
        )
        saved.append(
            plot_direction_all_phases_brain(
                table, root, out_dir, direction=direction, k=k, ctx=ctx
            )
        )
    return saved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DecisionVsRepeat brains colored by NMF functional clusters."
    )
    parser.add_argument("--k", type=int, default=2, choices=(2, 3))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    paths = run_decision_vs_repeat_cluster_figures(
        results_root=args.results_root,
        project_root=args.project_root,
        out_dir=args.out_dir,
        k=args.k,
    )
    for p in paths:
        print(p)
