"""Visualization helpers for INS whole-window Haufe decoding patterns."""

from __future__ import annotations

import gc
import logging
import resource
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne_bids import BIDSPath

from src.decoding.run_decoding_patterns import pattern_datatype
from src.paths import PROJECT_ROOT, RESULTS_ROOT, decoding_task_dir
from src.univariate.viz_mean import (
    BrainSurfaceContext,
    INSULA_LABEL_PATTERNS,
    project_to_pial,
)

logger = logging.getLogger(__name__)

PHASES = ("Stimulus", "Delay", "Go", "Response")
PSEUDO_SUBJECTS = ("INSl", "INSr")
BAND = "highgamma"
REF = "bipolar"
CLUSTERS = ("sustained_ramping", "intermediate", "sensory_transient")
CLUSTER_ORDER = {
    "sustained_ramping": 0,
    "intermediate": 1,
    "sensory_transient": 2,
}
# Drop a channel if this fraction of its significant samples falls before onset (t < 0).
PRESTIM_FRAC_THRESHOLD = 0.95
# Brain plots only: electrode is significant if cluster mask hits this closed window
# (relative to that phase's onset). Heatmaps keep the full epoch axis.
PHASE_SIG_WINDOWS: dict[str, tuple[float, float]] = {
    "Stimulus": (0.0, 0.8),
    "Delay": (0.0, 0.8),
    "Go": (0.0, 0.5),
    "Response": (0.0, 1.0),
}

RED = "#A9373B"
BLUE = "#2369BD"
ORANGE = "#CC8963"
GREEN = "#009944"

GOLD = "#C4A35A"

CLUSTER_COLORS = {
    "sustained_ramping": RED,
    "intermediate": GOLD,
    "sensory_transient": BLUE,
}
CLUSTER_LABELS = {
    "sustained_ramping": "sustained / ramping",
    "intermediate": "intermediate",
    "sensory_transient": "sensory / transient",
}

FEATURE_COLORS = {
    "lexicality": BLUE,
    "phoneme": ORANGE,
    "articulator": RED,
}

# Brain electrode styling: color = NMF functional cluster (not pattern sign).
NSIG_COLOR = "#666666"
NSIG_SIZE = 12.0
SIG_SIZE = 40.0

LEXICAL_GRID = {
    "task": "LexicalDelay",
    "descriptions": ("Repeat", "Decision"),
    "features": ("lexicality", "phoneme", "articulator"),
}
PHONEME_GRID = {
    "task": "PhonemeSequence",
    "descriptions": ("Repeat",),
    "features": ("phoneme", "articulator"),
}


@dataclass(frozen=True)
class PatternSpec:
    task: str
    subject: str
    description: str
    feature: str
    phase: str


def pattern_results_root(project_root: Path, task: str) -> Path:
    if project_root.resolve() == PROJECT_ROOT.resolve():
        return decoding_task_dir(task)
    return project_root / "results" / "decoding" / task


def pattern_h5_path(
    project_root: Path,
    *,
    task: str,
    subject: str,
    feature: str,
    phase: str,
    description: str,
) -> Path:
    recording = "1" if task == "PhonemeSequence" else None
    path = BIDSPath(
        root=str(pattern_results_root(project_root, task)),
        datatype=pattern_datatype(feature),
        subject=subject,
        task=task,
        suffix=BAND,
        processing=phase,
        recording=recording,
        description=description,
        extension=".h5",
        check=False,
    )
    return Path(path.fpath)


def load_pattern(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        pattern = np.asarray(handle["pattern"][()], dtype=float)
        pattern_mask = np.asarray(handle["pattern_mask"][()], dtype=bool)
        times = np.asarray(handle["times"][()], dtype=float)
        channels = [
            c.decode() if isinstance(c, bytes) else str(c) for c in handle["channel"][()]
        ]
        class_names = None
        if "class_names" in handle:
            class_names = [
                c.decode() if isinstance(c, bytes) else str(c)
                for c in handle["class_names"][()]
            ]
    return {
        "path": path,
        "pattern": pattern,
        "pattern_mask": pattern_mask,
        "times": times,
        "channels": channels,
        "class_names": class_names,
        "multiclass": pattern.ndim == 3,
    }


def load_assignments(project_root: Path | None = None) -> pd.DataFrame:
    from src.paths import PROJECT_ROOT as _PKG_ROOT
    from src.paths import nmf_assignments_path

    root = Path(project_root) if project_root is not None else _PKG_ROOT
    path = root / "results" / "nmf" / "channel_assignments.csv"
    if not path.exists():
        path = nmf_assignments_path()
    df = pd.read_csv(path)
    keep = df["functional_cluster"].isin(CLUSTERS)
    return df.loc[keep, ["channel", "functional_cluster", "hemi", "x", "y", "z"]].copy()


def masked_pattern_ct(pattern: np.ndarray, pattern_mask: np.ndarray) -> np.ndarray:
    if pattern.ndim != 2:
        raise ValueError(f"Expected binary pattern (C,T), got {pattern.shape}")
    return np.where(pattern_mask, pattern, np.nan)


def union_multiclass_pattern_ct(pattern: np.ndarray, pattern_mask: np.ndarray) -> np.ndarray:
    if pattern.ndim != 3:
        raise ValueError(f"Expected OvR pattern (K,C,T), got {pattern.shape}")
    mask = pattern_mask.astype(bool)
    numer = np.where(mask, pattern, 0.0).sum(axis=0)
    denom = mask.sum(axis=0)
    out = np.full(pattern.shape[1:], np.nan, dtype=float)
    np.divide(numer, denom, out=out, where=denom > 0)
    return out


def significance_mask_ct(pattern: np.ndarray, pattern_mask: np.ndarray) -> np.ndarray:
    """Return (C, T) bool significance after OvR aggregation if needed."""
    if pattern.ndim == 3:
        return pattern_mask.astype(bool).any(axis=0)
    return pattern_mask.astype(bool)


def phase_time_mask(times: np.ndarray, phase: str) -> np.ndarray:
    """Closed interval [t0, t1] for brain significance (see ``PHASE_SIG_WINDOWS``)."""
    if phase not in PHASE_SIG_WINDOWS:
        raise KeyError(f"Unknown phase {phase!r}; expected one of {list(PHASE_SIG_WINDOWS)}")
    t0, t1 = PHASE_SIG_WINDOWS[phase]
    t = np.asarray(times, dtype=float)
    return (t >= t0) & (t <= t1)


def restrict_mask_to_phase_window(
    mask_ct: np.ndarray,
    times: np.ndarray,
    phase: str,
) -> np.ndarray:
    """Zero out significant samples outside the phase brain window."""
    if mask_ct.ndim != 2:
        raise ValueError(f"Expected (C,T) mask, got {mask_ct.shape}")
    in_win = phase_time_mask(times, phase)
    return np.asarray(mask_ct, dtype=bool) & in_win[None, :]


def channel_keep_mask(
    sig_ct: np.ndarray,
    times: np.ndarray,
    *,
    prestim_frac: float = PRESTIM_FRAC_THRESHOLD,
) -> np.ndarray:
    """Keep channel unless >= ``prestim_frac`` of its significant samples are t < 0.

    Channels with no significant samples are kept (they simply stay blank).
    Touching pre-onset alone is not enough to drop a channel.
    """
    if sig_ct.ndim != 2:
        raise ValueError(f"Expected (C,T) significance mask, got {sig_ct.shape}")
    pre = np.asarray(times, dtype=float) < 0.0
    keep = np.ones(sig_ct.shape[0], dtype=bool)
    for ch_idx in range(sig_ct.shape[0]):
        sig = sig_ct[ch_idx]
        n_sig = int(sig.sum())
        if n_sig == 0:
            continue
        if int(sig[pre].sum()) / n_sig >= prestim_frac:
            keep[ch_idx] = False
    return keep


def filter_prestimulus_channels(
    channels: list[str],
    display_or_mask: np.ndarray,
    times: np.ndarray,
    *,
    prestim_frac: float = PRESTIM_FRAC_THRESHOLD,
) -> tuple[list[str], np.ndarray]:
    """Drop mostly-prestimulus channels from a (C,T) array and channel list."""
    if display_or_mask.dtype == bool:
        sig_ct = display_or_mask
        keep = channel_keep_mask(sig_ct, times, prestim_frac=prestim_frac)
        return [ch for ch, k in zip(channels, keep) if k], display_or_mask[keep]
    sig_ct = np.isfinite(display_or_mask)
    keep = channel_keep_mask(sig_ct, times, prestim_frac=prestim_frac)
    return [ch for ch, k in zip(channels, keep) if k], display_or_mask[keep]


def cluster_channel_order(
    channels: Iterable[str],
    assignments: pd.DataFrame,
) -> list[str]:
    rows = assignments.set_index("channel").reindex(list(channels))
    rows = rows.dropna(subset=["functional_cluster"])
    rows = rows.assign(
        cluster_rank=rows["functional_cluster"].map(CLUSTER_ORDER).fillna(99),
        y_sort=rows["y"],
    )
    rows = rows.sort_values(["cluster_rank", "y_sort", "channel"])
    return rows.index.tolist()


def split_channels_by_cluster(
    channels: list[str],
    assignments: pd.DataFrame,
) -> list[tuple[str, list[str]]]:
    ordered = cluster_channel_order(channels, assignments)
    blocks: list[tuple[str, list[str]]] = []
    for cluster in CLUSTERS:
        chs = [ch for ch in ordered if assignments.set_index("channel").at[ch, "functional_cluster"] == cluster]
        if chs:
            blocks.append((cluster, chs))
    return blocks


def heatmap_vmax(display: np.ndarray) -> float:
    vals = display[np.isfinite(display)]
    if vals.size == 0:
        return 1.0
    bound = float(np.percentile(np.abs(vals), 98))
    return bound or float(np.max(np.abs(vals))) or 1.0


def draw_signed_heatmap(
    ax,
    display: np.ndarray,
    times: np.ndarray,
    channels: list[str],
    assignments: pd.DataFrame,
    *,
    title: str = "",
    show_ylabel: bool = True,
    vmax: float | None = None,
) -> None:
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="0.92")
    if vmax is None:
        vmax = heatmap_vmax(display)
    im = ax.imshow(
        display,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        extent=[times[0], times[-1], -0.5, len(channels) - 0.5],
        interpolation="nearest",
    )
    ax.axvline(0, color="0.5", ls="--", lw=0.6)
    if title:
        ax.set_title(title, fontsize=7)
    ax.set_xlabel("Time (s)", fontsize=7)
    if show_ylabel:
        ax.set_ylabel("Channel", fontsize=7)
    else:
        ax.set_yticks([])
    yticks = np.arange(len(channels))
    ax.set_yticks(yticks)
    ax.set_yticklabels(channels, fontsize=4)
    cluster_by_ch = assignments.set_index("channel")["functional_cluster"]
    for tick, ch in zip(ax.get_yticklabels(), channels):
        tick.set_color(CLUSTER_COLORS.get(cluster_by_ch.get(ch, ""), "0.2"))
    prev_cluster = None
    for idx, ch in enumerate(channels):
        cluster = cluster_by_ch.get(ch)
        if prev_cluster is not None and cluster != prev_cluster:
            ax.axhline(idx - 0.5, color="0.25", lw=0.8)
        prev_cluster = cluster


def collect_heatmap_panels(
    project_root: Path,
    assignments: pd.DataFrame,
    *,
    task: str,
    description: str,
    feature: str,
    class_name: str | None = None,
) -> tuple[list[dict], float]:
    panels: list[dict] = []
    all_vals: list[np.ndarray] = []
    for subject in PSEUDO_SUBJECTS:
        for phase in PHASES:
            path = pattern_h5_path(
                project_root,
                task=task,
                subject=subject,
                feature=feature,
                phase=phase,
                description=description,
            )
            if not path.exists():
                panels.append({"missing": True, "subject": subject, "phase": phase})
                continue
            data = load_pattern(path)
            pattern = data["pattern"]
            mask = data["pattern_mask"]
            if data["multiclass"]:
                if class_name is None:
                    display = union_multiclass_pattern_ct(pattern, mask)
                else:
                    class_idx = data["class_names"].index(class_name)
                    display = masked_pattern_ct(pattern[class_idx], mask[class_idx])
            else:
                display = masked_pattern_ct(pattern, mask)
            channels = cluster_channel_order(data["channels"], assignments)
            ch_idx = [data["channels"].index(ch) for ch in channels]
            display = display[ch_idx, :]
            channels, display = filter_prestimulus_channels(
                channels, display, data["times"]
            )
            panels.append(
                {
                    "missing": False,
                    "subject": subject,
                    "phase": phase,
                    "display": display,
                    "times": data["times"],
                    "channels": channels,
                }
            )
            vals = display[np.isfinite(display)]
            if vals.size:
                all_vals.append(vals)
    if all_vals:
        merged = np.concatenate(all_vals)
        global_vmax = float(np.percentile(np.abs(merged), 98)) or 1.0
    else:
        global_vmax = 1.0
    return panels, global_vmax


def channel_significant_scalar(
    pattern: np.ndarray,
    pattern_mask: np.ndarray,
    times: np.ndarray | None = None,
    *,
    phase: str | None = None,
) -> np.ndarray:
    """Per-channel |mean| over significant times.

    If ``phase`` is set (brain plots), only samples in ``PHASE_SIG_WINDOWS[phase]``
    count. Heatmaps should leave ``phase=None`` and keep the full epoch.
    """
    if pattern.ndim == 3:
        pat = union_multiclass_pattern_ct(pattern, pattern_mask)
        mask = pattern_mask.any(axis=0)
    else:
        pat = pattern
        mask = pattern_mask.astype(bool)
    if phase is not None:
        if times is None:
            raise ValueError("times is required when phase window filtering is enabled")
        mask = restrict_mask_to_phase_window(mask, times, phase)
        # Window is already post-onset; skip prestim drop used for full-epoch heatmaps.
        keep = np.ones(mask.shape[0], dtype=bool)
    else:
        keep = (
            channel_keep_mask(mask, times)
            if times is not None
            else np.ones(mask.shape[0], dtype=bool)
        )
    out = np.full(pat.shape[0], np.nan, dtype=float)
    for ch_idx in range(pat.shape[0]):
        if not keep[ch_idx]:
            continue
        sig = mask[ch_idx]
        if sig.any():
            out[ch_idx] = float(np.nanmean(pat[ch_idx, sig]))
    return out


def significant_feature_mask(
    path: Path,
    *,
    phase: str | None = None,
) -> dict[str, np.ndarray]:
    data = load_pattern(path)
    channels = data["channels"]
    sig_ct = significance_mask_ct(data["pattern"], data["pattern_mask"])
    if phase is not None:
        sig_ct = restrict_mask_to_phase_window(sig_ct, data["times"], phase)
        mask_any = sig_ct.any(axis=1)
    else:
        keep = channel_keep_mask(sig_ct, data["times"])
        mask_any = sig_ct.any(axis=1) & keep
    return {"channels": channels, "significant": mask_any}


def build_pattern_spatial_frame(
    channels: list[str],
    values: np.ndarray,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """All pattern channels with MNI coords; ``pattern`` is |mean| if significant else NaN."""
    df = pd.DataFrame({"channel": list(channels), "pattern": np.asarray(values, dtype=float)})
    df = df.merge(assignments, on="channel", how="inner")
    df = df.dropna(subset=["x", "y", "z"]).copy()
    df["significant"] = df["pattern"].notna()
    # Visualization ignores sign; keep absolute magnitude only.
    df.loc[df["significant"], "pattern"] = df.loc[df["significant"], "pattern"].abs()
    return df.reset_index(drop=True)


def _raise_nofile_soft_limit(target: int = 65536) -> None:
    """Raise soft RLIMIT_NOFILE so repeated Brain renders do not hit Errno 24."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft = min(hard, max(soft, target))
        if new_soft > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    except (ValueError, OSError):
        pass


def _ensure_mne_notebook_backend() -> None:
    """MNE 3D backends are only ``notebook`` / ``pyvistaqt`` (not ``agg``)."""
    import os

    import mne
    import pyvista as pv

    _raise_nofile_soft_limit()
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    pv.OFF_SCREEN = True
    # Reject stale env (e.g. Slurm scripts that exported MNE_3D_BACKEND=agg).
    if os.environ.get("MNE_3D_BACKEND", "").lower() == "agg":
        os.environ["MNE_3D_BACKEND"] = "notebook"
    mne.viz.set_3d_backend("notebook")


def _close_brain(brain) -> None:
    """Best-effort teardown for MNE Brain / PyVista plotters (FD leak guard)."""
    if brain is None:
        return
    try:
        plotter = getattr(getattr(brain, "_renderer", None), "plotter", None)
        if plotter is not None:
            try:
                plotter.clear()
            except Exception:
                pass
            try:
                plotter.close()
            except Exception:
                pass
    except Exception:
        pass
    try:
        brain.close()
    except Exception:
        pass
    try:
        import pyvista as pv

        pv.close_all()
    except Exception:
        pass
    gc.collect()


def _make_insula_brain(hemi: str, ctx: BrainSurfaceContext):
    from mne.viz import Brain

    _ensure_mne_notebook_backend()
    brain = Brain(
        ctx.fs_subject,
        subjects_dir=str(ctx.recon_dir),
        surf="pial",
        hemi=hemi,
        background="white",
        show=False,
        cortex=(0.9, 0.9, 0.9),
        alpha=0.05,
        size=(800, 800),
    )
    for label in ctx.labels:
        if label.hemi == hemi and any(p in label.name for p in INSULA_LABEL_PATTERNS):
            brain.add_label(label, borders=False, color=(0.9, 0.9, 0.9), alpha=0.6)
    return brain


def _rgb(color) -> tuple[float, float, float]:
    if isinstance(color, str):
        return tuple(float(c) for c in mcolors.to_rgb(color))
    return tuple(float(c) for c in color[:3])


def _add_points_to_brain(
    brain,
    side: pd.DataFrame,
    hemi: str,
    ctx: BrainSurfaceContext,
    *,
    colors: list,
    sizes: list[float],
) -> None:
    """Add electrodes as few PolyData meshes (grouped by color/size) to limit FD use."""
    import pyvista as pv

    if side.empty:
        return
    coords = side[["x", "y", "z"]].to_numpy(float)
    projected = project_to_pial(coords, hemi, ctx)
    groups: dict[tuple[tuple[float, float, float], float], list[np.ndarray]] = defaultdict(list)
    for pt, color, size in zip(projected, colors, sizes):
        groups[(_rgb(color), float(size))].append(pt)
    for (rgb, size), pts in groups.items():
        cloud = pv.PolyData(np.vstack(pts))
        brain._renderer.plotter.add_mesh(
            cloud,
            render_points_as_spheres=True,
            point_size=size,
            color=rgb,
            opacity=0.95,
            lighting=False,
        )


def _electrode_point_sizes(
    side: pd.DataFrame,
    spatial: pd.DataFrame,
    *,
    size_by_pattern: bool,
    size_range: tuple[float, float],
    size_gamma: float = 2.0,
) -> list[float]:
    """Fixed SIG_SIZE, or scale significant ``pattern`` into ``size_range`` (shared ref).

    ``size_gamma`` > 1 compresses small effects toward ``vmin`` while keeping the
    largest near ``vmax`` (wider perceived dynamic range).
    """
    if not size_by_pattern:
        return [
            SIG_SIZE
            if bool(row["significant"]) and np.isfinite(row["pattern"])
            else NSIG_SIZE
            for _, row in side.iterrows()
        ]
    vals = spatial.loc[
        spatial["significant"].astype(bool) & np.isfinite(spatial["pattern"]),
        "pattern",
    ].to_numpy(dtype=float)
    ref = float(np.percentile(np.abs(vals), 95)) if vals.size else 1.0
    ref = max(ref, 1e-6)
    vmin, vmax = size_range
    gamma = max(float(size_gamma), 1e-6)
    sizes: list[float] = []
    for _, row in side.iterrows():
        if bool(row["significant"]) and np.isfinite(row["pattern"]):
            t = float(np.clip(abs(float(row["pattern"])) / ref, 0.0, 1.0))
            sizes.append(vmin + (vmax - vmin) * (t**gamma))
        else:
            sizes.append(NSIG_SIZE)
    return sizes


def render_insula_hemisphere_cluster(
    spatial: pd.DataFrame,
    hemi: str,
    ctx: BrainSurfaceContext,
    *,
    size_by_pattern: bool = False,
    size_range: tuple[float, float] = (8.0, 48.0),
    size_gamma: float = 2.0,
) -> np.ndarray:
    """Screenshot one hemisphere: gray ns + cluster colors (sustained=red, intermediate=gold, sensory=blue).

    When ``size_by_pattern`` is True, significant electrode size scales with
    ``|pattern|`` (95th-percentile of the full frame as the upper reference;
    ``size_gamma``>1 keeps small effects closer to ``size_range[0]``).
    """
    brain = None
    try:
        brain = _make_insula_brain(hemi, ctx)
        side_mask = spatial["x"].lt(0) if hemi == "lh" else spatial["x"].gt(0)
        side = spatial.loc[side_mask]
        if len(side):
            colors = []
            for _, row in side.iterrows():
                if bool(row["significant"]) and np.isfinite(row["pattern"]):
                    cluster = row.get("functional_cluster", "")
                    colors.append(CLUSTER_COLORS.get(cluster, NSIG_COLOR))
                else:
                    colors.append(NSIG_COLOR)
            sizes = _electrode_point_sizes(
                side,
                spatial,
                size_by_pattern=size_by_pattern,
                size_range=size_range,
                size_gamma=size_gamma,
            )
            _add_points_to_brain(brain, side, hemi, ctx, colors=colors, sizes=sizes)
        brain.show_view(
            azimuth=180 if hemi == "lh" else 0,
            elevation=90,
            distance=180,
            focalpoint=ctx.insula_centers[hemi],
        )
        return brain.screenshot(mode="rgb")
    finally:
        _close_brain(brain)


# Back-compat alias
render_insula_hemisphere_signed = render_insula_hemisphere_cluster


def render_insula_hemisphere_categorical(
    spatial: pd.DataFrame,
    hemi: str,
    ctx: BrainSurfaceContext,
) -> np.ndarray:
    """Screenshot one hemisphere: gray ns base + categorical significant colors."""
    brain = None
    try:
        brain = _make_insula_brain(hemi, ctx)
        side_mask = spatial["x"].lt(0) if hemi == "lh" else spatial["x"].gt(0)
        side = spatial.loc[side_mask]
        if len(side):
            colors = []
            sizes = []
            for _, row in side.iterrows():
                if bool(row["significant"]):
                    colors.append(row["color"])
                    sizes.append(SIG_SIZE)
                else:
                    colors.append(NSIG_COLOR)
                    sizes.append(NSIG_SIZE)
            _add_points_to_brain(brain, side, hemi, ctx, colors=colors, sizes=sizes)
        brain.show_view(
            azimuth=180 if hemi == "lh" else 0,
            elevation=90,
            distance=180,
            focalpoint=ctx.insula_centers[hemi],
        )
        return brain.screenshot(mode="rgb")
    finally:
        _close_brain(brain)


def overlay_color_for_features(active: tuple[str, ...]) -> str:
    palette = {
        ("lexicality",): FEATURE_COLORS["lexicality"],
        ("phoneme",): FEATURE_COLORS["phoneme"],
        ("articulator",): FEATURE_COLORS["articulator"],
        ("lexicality", "phoneme"): "#6A5ACD",
        ("lexicality", "articulator"): "#20B2AA",
        ("phoneme", "articulator"): "#CC8963",
        ("articulator", "phoneme"): "#CC8963",
        ("lexicality", "phoneme", "articulator"): GREEN,
        ("lexicality", "articulator", "phoneme"): GREEN,
    }
    return palette.get(tuple(sorted(active)), "#555555")


def iter_grid_specs() -> list[PatternSpec]:
    specs: list[PatternSpec] = []
    for grid in (LEXICAL_GRID, PHONEME_GRID):
        for subject in PSEUDO_SUBJECTS:
            for description in grid["descriptions"]:
                for feature in grid["features"]:
                    for phase in PHASES:
                        specs.append(
                            PatternSpec(
                                task=grid["task"],
                                subject=subject,
                                description=description,
                                feature=feature,
                                phase=phase,
                            )
                        )
    return specs


def save_figure(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    """Save SVG only (PNG not written)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".svg"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path.with_suffix(".svg"))


def plot_subject_phase_heatmaps(
    project_root: Path,
    assignments: pd.DataFrame,
    out_dir: Path,
    *,
    task: str,
    description: str,
    feature: str,
    class_name: str | None = None,
    panel_tag: str = "heatmap",
) -> None:
    fig, axes = plt.subplots(
        nrows=len(PSEUDO_SUBJECTS),
        ncols=len(PHASES),
        figsize=(2.4 * len(PHASES), 1.8 * len(PSEUDO_SUBJECTS)),
        squeeze=False,
    )
    any_panel = False
    panels, global_vmax = collect_heatmap_panels(
        project_root,
        assignments,
        task=task,
        description=description,
        feature=feature,
        class_name=class_name,
    )
    for row_idx, subject in enumerate(PSEUDO_SUBJECTS):
        for col_idx, phase in enumerate(PHASES):
            ax = axes[row_idx, col_idx]
            panel = panels[row_idx * len(PHASES) + col_idx]
            if panel.get("missing"):
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{subject} {phase}", fontsize=7)
                continue
            if not panel["channels"]:
                ax.text(0.5, 0.5, "no channels", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{subject} {phase}", fontsize=7)
                continue
            draw_signed_heatmap(
                ax,
                panel["display"],
                panel["times"],
                panel["channels"],
                assignments,
                title=f"{subject} {phase}",
                show_ylabel=col_idx == 0,
                vmax=global_vmax,
            )
            any_panel = True
    if not any_panel:
        plt.close(fig)
        logger.warning("Skip heatmap (no data): %s %s %s", task, description, feature)
        return
    suffix = "binary" if feature == "lexicality" else (class_name or "union")
    fname = f"{task}_{description}_{feature}_{suffix}_{panel_tag}"
    fig.suptitle(f"{task} {description} {feature}" + (f" {class_name}" if class_name else ""), fontsize=9)
    fig.tight_layout()
    save_figure(fig, out_dir / fname)


def _load_phase_spatial(
    project_root: Path,
    assignments: pd.DataFrame,
    *,
    task: str,
    subject: str,
    feature: str,
    phase: str,
    description: str,
) -> pd.DataFrame | None:
    path = pattern_h5_path(
        project_root,
        task=task,
        subject=subject,
        feature=feature,
        phase=phase,
        description=description,
    )
    if not path.exists():
        return None
    data = load_pattern(path)
    values = channel_significant_scalar(
        data["pattern"],
        data["pattern_mask"],
        data["times"],
        phase=phase,
    )
    return build_pattern_spatial_frame(data["channels"], values, assignments)


def plot_single_feature_brain(
    project_root: Path,
    assignments: pd.DataFrame,
    out_dir: Path,
    *,
    task: str,
    description: str,
    feature: str,
    ctx: BrainSurfaceContext | None = None,
    subject: str | None = None,
) -> None:
    """One figure per task×description×feature.

    Row 0 = Left insula (INSl), row 1 = Right insula (INSr);
    columns = Stimulus / Delay / Go / Response.
    ``subject`` is ignored (kept for call-site compatibility).
    """
    del subject  # one combined figure; do not split by pseudo-subject
    surface_ctx = ctx or BrainSurfaceContext()
    # (row hemi key, ylabel, pattern H5 subject)
    rows_spec = (
        ("lh", "Left", "INSl"),
        ("rh", "Right", "INSr"),
    )

    spatial: dict[tuple[str, str], pd.DataFrame | None] = {}
    any_data = False
    for _, _, subj in rows_spec:
        for phase in PHASES:
            frame = _load_phase_spatial(
                project_root,
                assignments,
                task=task,
                subject=subj,
                feature=feature,
                phase=phase,
                description=description,
            )
            spatial[(subj, phase)] = frame
            if frame is not None and not frame.empty:
                any_data = True

    if not any_data:
        logger.warning(
            "Skip brain single (missing): %s %s %s", task, description, feature
        )
        return

    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(PHASES),
        figsize=(3.0 * len(PHASES), 5.2),
        squeeze=False,
    )
    any_panel = False
    for row, (hemi, hemi_label, subj) in enumerate(rows_spec):
        for col, phase in enumerate(PHASES):
            ax = axes[row, col]
            frame = spatial[(subj, phase)]
            if frame is None or frame.empty:
                ax.text(
                    0.5, 0.5,
                    "missing" if frame is None else "no channels",
                    ha="center", va="center",
                )
                ax.axis("off")
                if row == 0:
                    ax.set_title(phase, fontsize=7)
                if col == 0:
                    ax.set_ylabel(hemi_label, fontsize=7)
                continue
            img = render_insula_hemisphere_cluster(frame, hemi, surface_ctx)
            ax.imshow(img)
            ax.axis("off")
            if row == 0:
                ax.set_title(phase, fontsize=7)
            if col == 0:
                ax.set_ylabel(hemi_label, fontsize=7)
            any_panel = True

    if not any_panel:
        plt.close(fig)
        return
    fname = f"{task}_{description}_{feature}_brain_single"
    fig.suptitle(
        f"{task} {description} {feature}  "
        f"(red=sustained; gold=intermediate; blue=sensory; gray=ns)",
        fontsize=8,
    )
    fig.tight_layout()
    save_figure(fig, out_dir / fname)


def union_spatial_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Union channels across tasks: significant if significant in any; |pattern| = max."""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()

    channels = sorted(set().union(*(set(f["channel"]) for f in frames)))
    by_ch = {ch: [] for ch in channels}
    for frame in frames:
        for _, row in frame.iterrows():
            by_ch[row["channel"]].append(row)

    rows = []
    for ch in channels:
        parts = by_ch[ch]
        if not parts:
            continue
        base = parts[0]
        sig = any(bool(p["significant"]) for p in parts)
        pats = [
            float(p["pattern"])
            for p in parts
            if bool(p["significant"]) and np.isfinite(p["pattern"])
        ]
        pattern = float(np.max(pats)) if pats else np.nan
        rows.append(
            {
                "channel": ch,
                "pattern": pattern if sig else np.nan,
                "significant": sig,
                "functional_cluster": base["functional_cluster"],
                "hemi": base["hemi"],
                "x": float(base["x"]),
                "y": float(base["y"]),
                "z": float(base["z"]),
            }
        )
    return pd.DataFrame(rows)


def plot_cross_task_repeat_brain(
    project_root: Path,
    assignments: pd.DataFrame,
    out_dir: Path,
    *,
    feature: str,
    ctx: BrainSurfaceContext | None = None,
    tasks: tuple[str, ...] = ("LexicalDelay", "PhonemeSequence"),
    description: str = "Repeat",
) -> Path | None:
    """Union significant electrodes across tasks on one 2×4 brain grid.

    For each hemi×phase panel, channels significant in *any* of ``tasks``
    (same feature / Repeat) are plotted together. No task-level markers.
    Color = NMF cluster (sustained=red, intermediate=gold, sensory=blue). Per-task figures unchanged.
    """
    surface_ctx = ctx or BrainSurfaceContext()
    rows_spec = (
        ("lh", "Left", "INSl"),
        ("rh", "Right", "INSr"),
    )

    spatial: dict[tuple[str, str], pd.DataFrame | None] = {}
    any_data = False
    for _, _, subj in rows_spec:
        for phase in PHASES:
            frames = [
                _load_phase_spatial(
                    project_root,
                    assignments,
                    task=task,
                    subject=subj,
                    feature=feature,
                    phase=phase,
                    description=description,
                )
                for task in tasks
            ]
            frames = [f for f in frames if f is not None]
            if not frames:
                spatial[(subj, phase)] = None
                continue
            united = union_spatial_frames(frames)
            spatial[(subj, phase)] = united
            if not united.empty:
                any_data = True

    if not any_data:
        logger.warning(
            "Skip cross-task brain (missing): %s %s %s",
            "+".join(tasks),
            description,
            feature,
        )
        return None

    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(PHASES),
        figsize=(3.0 * len(PHASES), 5.2),
        squeeze=False,
    )
    any_panel = False
    for row, (hemi, hemi_label, subj) in enumerate(rows_spec):
        for col, phase in enumerate(PHASES):
            ax = axes[row, col]
            frame = spatial[(subj, phase)]
            if frame is None or frame.empty:
                ax.text(
                    0.5,
                    0.5,
                    "missing" if frame is None else "no channels",
                    ha="center",
                    va="center",
                )
                ax.axis("off")
                if row == 0:
                    ax.set_title(phase, fontsize=7)
                if col == 0:
                    ax.set_ylabel(hemi_label, fontsize=7)
                continue
            img = render_insula_hemisphere_cluster(frame, hemi, surface_ctx)
            ax.imshow(img)
            ax.axis("off")
            if row == 0:
                ax.set_title(phase, fontsize=7)
            if col == 0:
                ax.set_ylabel(hemi_label, fontsize=7)
            any_panel = True

    if not any_panel:
        plt.close(fig)
        return None

    task_tag = "_".join(tasks)
    fname = f"cross_task_{description}_{feature}_{task_tag}_union_brain"
    fig.suptitle(
        f"{' ∪ '.join(tasks)} {description} {feature}  "
        f"(sig union; red=sustained; gold=intermediate; blue=sensory; gray=ns)",
        fontsize=8,
    )
    fig.tight_layout()
    out_path = out_dir / fname
    save_figure(fig, out_path)
    return out_path.with_suffix(".svg")


def _overlay_spatial_for_subject(
    project_root: Path,
    assignments: pd.DataFrame,
    *,
    task: str,
    description: str,
    subject: str,
    features: tuple[str, ...],
    phase: str,
) -> tuple[pd.DataFrame | None, dict[str, str]]:
    """Build categorical spatial frame for one pseudo-subject × phase."""
    channel_active: dict[str, set[str]] = {}
    all_channels: set[str] = set()
    any_h5 = False
    for feature in features:
        path = pattern_h5_path(
            project_root,
            task=task,
            subject=subject,
            feature=feature,
            phase=phase,
            description=description,
        )
        if not path.exists():
            continue
        any_h5 = True
        sig = significant_feature_mask(path, phase=phase)
        for ch, is_sig in zip(sig["channels"], sig["significant"]):
            all_channels.add(ch)
            if is_sig:
                channel_active.setdefault(ch, set()).add(feature)
    if not any_h5:
        return None, {}
    legend_entries: dict[str, str] = {}
    rows = []
    for ch in sorted(all_channels):
        row = assignments.loc[assignments["channel"] == ch]
        if row.empty:
            continue
        active = tuple(sorted(channel_active.get(ch, ())))
        significant = bool(active)
        color = overlay_color_for_features(active) if significant else NSIG_COLOR
        if significant:
            legend_entries[str(active)] = color
        rows.append(
            {
                "channel": ch,
                "x": float(row["x"].iloc[0]),
                "y": float(row["y"].iloc[0]),
                "z": float(row["z"].iloc[0]),
                "color": color,
                "significant": significant,
            }
        )
    return pd.DataFrame(rows), legend_entries


def plot_feature_overlay_brain(
    project_root: Path,
    assignments: pd.DataFrame,
    out_dir: Path,
    *,
    task: str,
    description: str,
    features: tuple[str, ...],
    ctx: BrainSurfaceContext | None = None,
    subject: str | None = None,
) -> Path | None:
    """One overlay figure per task×description: Left=INSl, Right=INSr × phases.

    ``subject`` is ignored (kept for call-site compatibility).
    """
    del subject
    surface_ctx = ctx or BrainSurfaceContext()
    rows_spec = (
        ("lh", "Left", "INSl"),
        ("rh", "Right", "INSr"),
    )
    spatial: dict[tuple[str, str], pd.DataFrame | None] = {}
    legend_entries: dict[str, str] = {}
    any_data = False
    for _, _, subj in rows_spec:
        for phase in PHASES:
            frame, legends = _overlay_spatial_for_subject(
                project_root,
                assignments,
                task=task,
                description=description,
                subject=subj,
                features=features,
                phase=phase,
            )
            spatial[(subj, phase)] = frame
            legend_entries.update(legends)
            if frame is not None and not frame.empty:
                any_data = True

    if not any_data:
        logger.warning("Skip brain overlay (missing): %s %s", task, description)
        return None

    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(PHASES),
        figsize=(3.0 * len(PHASES), 5.6),
        squeeze=False,
    )
    any_panel = False
    for row, (hemi, hemi_label, subj) in enumerate(rows_spec):
        for col, phase in enumerate(PHASES):
            ax = axes[row, col]
            frame = spatial[(subj, phase)]
            if frame is None or frame.empty:
                ax.text(
                    0.5,
                    0.5,
                    "missing" if frame is None else "no channels",
                    ha="center",
                    va="center",
                )
                ax.axis("off")
                if row == 0:
                    ax.set_title(phase, fontsize=7)
                if col == 0:
                    ax.set_ylabel(hemi_label, fontsize=7)
                continue
            img = render_insula_hemisphere_categorical(frame, hemi, surface_ctx)
            ax.imshow(img)
            ax.axis("off")
            if row == 0:
                ax.set_title(phase, fontsize=7)
            if col == 0:
                ax.set_ylabel(hemi_label, fontsize=7)
            any_panel = True

    if not any_panel:
        plt.close(fig)
        return None

    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=NSIG_COLOR,
            markersize=6, label="non-significant",
        )
    ]
    handles.extend(
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=color,
            markersize=7, label=label,
        )
        for label, color in sorted(legend_entries.items(), key=lambda item: item[0])
    )
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), fontsize=6)
    fname = f"{task}_{description}_brain_overlay"
    fig.suptitle(f"{task} {description} feature overlay (Left=INSl, Right=INSr)", fontsize=8)
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    out_path = out_dir / fname
    save_figure(fig, out_path)
    return out_path.with_suffix(".svg")


def run_all_figures(
    project_root: Path | None = None,
    out_dir: Path | None = None,
) -> None:
    project_root = Path(project_root or RESULTS_ROOT.parent)
    from src.paths import img_dir

    out_dir = Path(out_dir or img_dir("insula_patterns"))
    assignments = load_assignments(project_root)
    ctx = BrainSurfaceContext()

    for grid in (LEXICAL_GRID, PHONEME_GRID):
        task = grid["task"]
        for description in grid["descriptions"]:
            for feature in grid["features"]:
                plot_subject_phase_heatmaps(
                    project_root, assignments, out_dir,
                    task=task, description=description, feature=feature,
                )

            for feature in grid["features"]:
                plot_single_feature_brain(
                    project_root, assignments, out_dir,
                    task=task, description=description, feature=feature,
                    ctx=ctx,
                )

            plot_feature_overlay_brain(
                project_root, assignments, out_dir,
                task=task, description=description,
                features=tuple(grid["features"]),
                ctx=ctx,
            )

    # LexicalDelay + PhonemeSequence Repeat combined (phoneme / articulator).
    for feature in ("phoneme", "articulator"):
        plot_cross_task_repeat_brain(
            project_root, assignments, out_dir, feature=feature, ctx=ctx,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_all_figures()
