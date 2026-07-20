"""Loaders and spatial plotting for window-mean univariate results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from mne_bids import BIDSPath

from src.paths import RESULTS_ROOT

ATLAS = "hammers"
REFERENCE = "bipolar"
BAND = "highgamma"
DEFAULT_TASKS = ("LexicalDelay", "LexicalNoDelay")

PHASE_WINDOWS: dict[str, tuple[float, float]] = {
    "Stimulus": (0.0, 0.5),
    "Delay": (0.0, 0.5),
    "Go": (0.0, 0.5),
    "Response": (-0.5, 0.5),
}

CONTRAST_DESCRIPTIONS = {
    "DecisionVsRepeat": "DecisionVsRepeatMean",
    "WordVsNonwordDecision": "WordVsNonwordDecisionMean",
    "WordVsNonwordRepeat": "WordVsNonwordRepeatMean",
}

EXCLUDE_ROIS = {
    "Unknown",
    "BrainStem",
    "Thal",
    "CC",
    "Caud",
    "Put",
    "Amyg",
    "Hipp",
    "LinG",
    "Cun",
    "mOccG",
    "PhG",
    "FuG",
    "GRect",
    "LatV",
    "GSubcallosal",
    "Intersection",
    "BrainStem–Thal",
}

INSULA_ROIS = {"AIC", "PIC"}

INSULA_LABEL_PATTERNS = (
    "G_insular_short",
    "G_Ins_lg_and_S_cent_ins",
    "S_circular_insula_ant",
    "S_circular_insula_inf",
    "S_circular_insula_sup",
)

DEFAULT_RECON_DIR = Path("/cwork/ns458/ECoG_Recon")
DEFAULT_FS_SUBJECT = "cvs_avg35_inMNI152"

ViewMode = Literal["insula", "wholebrain"]


def _phase_processing(phase: str) -> str:
    phase_key = phase.strip().lower()
    for name in PHASE_WINDOWS:
        if name.lower() == phase_key:
            return name
    raise ValueError(f"Unknown phase {phase!r}; expected one of {list(PHASE_WINDOWS)}")


def task_results_dir(results_root: Path, task: str) -> Path:
    return results_root / f"{task}({REFERENCE})({ATLAS})"


def discover_mean_paths(
    results_root: Path,
    task: str,
    phase: str,
    contrast_desc: str,
) -> list[BIDSPath]:
    processing = _phase_processing(phase)
    return BIDSPath(
        root=str(task_results_dir(results_root, task)),
        datatype="univariate",
        processing=processing,
        description=contrast_desc,
        suffix=BAND,
        extension=".csv",
        check=False,
    ).match()


def load_mean_contrasts(
    results_root: Path | None = None,
    *,
    tasks: tuple[str, ...] | list[str] = DEFAULT_TASKS,
    phase: str | None = None,
    contrast_desc: str | None = None,
) -> pd.DataFrame:
    """Load window-mean univariate CSVs into one long table."""
    root = Path(results_root or RESULTS_ROOT)
    frames: list[pd.DataFrame] = []

    for task in tasks:
        if phase is not None and contrast_desc is not None:
            path_list = discover_mean_paths(root, task, phase, contrast_desc)
        else:
            path_list = BIDSPath(
                root=str(task_results_dir(root, task)),
                datatype="univariate",
                suffix=BAND,
                extension=".csv",
                check=False,
            ).match()
            path_list = [p for p in path_list if p.description and p.description.endswith("Mean")]

        for path in path_list:
            df = pd.read_csv(path)
            df["task"] = path.task or task
            df["subject"] = path.subject or df.get("subject")
            df["processing"] = path.processing
            if "phase" not in df.columns and path.processing:
                df["phase"] = str(path.processing).lower()
            if "contrast" not in df.columns and path.description:
                df["contrast"] = path.description
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_coord_metadata(
    results_root: Path | None = None,
    *,
    tasks: tuple[str, ...] | list[str] = DEFAULT_TASKS,
) -> pd.DataFrame:
    root = Path(results_root or RESULTS_ROOT)
    frames: list[pd.DataFrame] = []
    for task in tasks:
        paths = BIDSPath(
            root=str(task_results_dir(root, task)),
            datatype="HGA",
            suffix="coord",
            extension=".csv",
            check=False,
        ).match()
        for path in paths:
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    coords = pd.concat(frames, ignore_index=True)
    return coords.drop_duplicates(subset=["channel"], keep="first")


def attach_metadata(df: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    meta_cols = [c for c in ("channel", "roi", "hemi", "label", "x", "y", "z", "mix") if c in coords.columns]
    if not meta_cols:
        return df.copy()
    meta = coords[meta_cols].drop_duplicates(subset=["channel"])
    return df.merge(meta, on="channel", how="left")


def filter_qc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "mix" in out.columns:
        out = out[~out["mix"].eq(True)]
    if "label" in out.columns:
        label_num = pd.to_numeric(out["label"], errors="coerce")
        out = out[~(label_num == 0)]
    if "roi" in out.columns:
        out = out[~out["roi"].isin(EXCLUDE_ROIS)]
    return out.reset_index(drop=True)


def _direction_labels(contrast: str, mean_diff: pd.Series) -> pd.Series:
    contrast = str(contrast)
    if "DecisionVsRepeat" in contrast:
        pos, neg = "Decision", "Repeat"
    elif "WordVsNonword" in contrast:
        pos, neg = "Word", "Nonword"
    else:
        pos, neg = "positive", "negative"
    return np.where(mean_diff >= 0, pos, neg)


def select_significant(
    df: pd.DataFrame,
    *,
    phase: str | None = None,
    contrast: str | None = None,
    tasks: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    out = df.loc[df["significant"].astype(bool)].copy()
    if phase is not None:
        out = out.loc[out["phase"].str.lower() == phase.lower()]
    if contrast is not None:
        out = out.loc[out["contrast"] == contrast]
    if tasks is not None:
        out = out.loc[out["task"].isin(tasks)]
    if out.empty:
        return out
    contrast_name = out["contrast"].iloc[0]
    out["direction"] = _direction_labels(contrast_name, out["mean_diff"])
    return out.reset_index(drop=True)


def filter_insula_electrodes(df: pd.DataFrame) -> pd.DataFrame:
    """Keep Hammers pure insula electrodes (AIC/PIC) only."""
    if df.empty or "roi" not in df.columns:
        return df.iloc[0:0].copy()
    return df.loc[df["roi"].isin(INSULA_ROIS)].reset_index(drop=True)


def roi_counts(df: pd.DataFrame) -> pd.Series:
    if df.empty or "roi" not in df.columns:
        return pd.Series(dtype=int)
    return df.groupby("roi")["channel"].nunique().sort_values(ascending=False)


@dataclass
class BrainSurfaceContext:
    recon_dir: Path = DEFAULT_RECON_DIR
    fs_subject: str = DEFAULT_FS_SUBJECT
    pial: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    trees: dict = field(default_factory=dict, init=False)
    labels: list = field(default_factory=list, init=False)
    insula_centers: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    valid_vertices: dict[str, np.ndarray] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        import mne
        from scipy.spatial import cKDTree

        self.recon_dir = Path(self.recon_dir)
        self.labels = mne.read_labels_from_annot(
            subject=self.fs_subject,
            parc="aparc.a2009s",
            surf_name="pial",
            hemi="both",
            subjects_dir=str(self.recon_dir),
        )
        for hemi in ("lh", "rh"):
            coords, _ = mne.read_surface(
                str(self.recon_dir / self.fs_subject / "surf" / f"{hemi}.pial")
            )
            self.pial[hemi] = coords
            self.trees[hemi] = cKDTree(coords)
            vertices = []
            for label in self.labels:
                if label.hemi == hemi and any(p in label.name for p in INSULA_LABEL_PATTERNS):
                    vertices.extend(label.vertices)
            if vertices:
                self.insula_centers[hemi] = coords[np.asarray(vertices, dtype=int)].mean(axis=0)
            else:
                self.insula_centers[hemi] = coords.mean(axis=0)
            mask = np.zeros(len(coords), dtype=bool)
            for label in self.labels:
                if label.hemi == hemi:
                    mask[label.vertices] = True
            self.valid_vertices[hemi] = mask


def project_to_pial(coords: np.ndarray, hemi: str, ctx: BrainSurfaceContext) -> np.ndarray:
    if coords.size == 0:
        return coords.reshape(0, 3)
    _, indices = ctx.trees[hemi].query(coords)
    return ctx.pial[hemi][indices]


def _effect_bounds(values: np.ndarray, cmap_bounds: str) -> float:
    if values.size == 0:
        return 1.0
    if cmap_bounds == "p95":
        bound = float(np.percentile(np.abs(values), 95))
    else:
        bound = float(np.max(np.abs(values)))
    return max(bound, 1e-6)


def plot_signed_electrodes_brain(
    df: pd.DataFrame,
    *,
    view_mode: ViewMode = "insula",
    title: str = "",
    cmap_bounds: str = "p95",
    ctx: BrainSurfaceContext | None = None,
    red: str = "#A9373B",
    blue: str = "#2369BD",
    roi_filter: str | None = None,
):
    """Render signed mean_diff electrodes on insula zoom or whole-brain lateral views."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import mne
    import pyvista as pv
    from mne.viz import Brain

    if df.empty:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "No significant electrodes", ha="center", va="center")
        ax.axis("off")
        ax.set_title(title)
        return fig

    plot_df = df.copy()
    if view_mode == "insula":
        plot_df = filter_insula_electrodes(plot_df)
    elif roi_filter == "insula" and "roi" in plot_df.columns:
        plot_df = plot_df.loc[plot_df["roi"].isin(INSULA_ROIS)]

    if plot_df.empty:
        empty_msg = (
            "No insula electrodes"
            if view_mode == "insula"
            else f"No electrodes after roi_filter={roi_filter!r}"
        )
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, empty_msg, ha="center", va="center")
        ax.axis("off")
        ax.set_title(title)
        return fig

    mne.viz.set_3d_backend("notebook")
    surface_ctx = ctx or BrainSurfaceContext()
    max_abs = _effect_bounds(plot_df["mean_diff"].to_numpy(float), cmap_bounds)
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    cmap = mcolors.LinearSegmentedColormap.from_list("rb", [blue, "#f7f7f7", red])

    size_min, size_max = 4.0, 28.0
    alpha_min, alpha_max = 0.35, 0.95

    def _size(v: float) -> float:
        x = np.clip(abs(v) / max_abs, 0.0, 1.0)
        return size_min + x * (size_max - size_min)

    def _alpha(v: float) -> float:
        x = np.clip(abs(v) / max_abs, 0.0, 1.0)
        return alpha_min + x * (alpha_max - alpha_min)

    screenshots: list[np.ndarray] = []
    panel_titles: list[str] = []

    for hemi, panel_title in (("lh", "Left"), ("rh", "Right")):
        brain = Brain(
            surface_ctx.fs_subject,
            subjects_dir=str(surface_ctx.recon_dir),
            surf="pial",
            hemi=hemi,
            background="white",
            show=False,
            cortex=(0.9, 0.9, 0.9),
            alpha=0.1 if view_mode == "wholebrain" else 0.08,
            size=(700, 700),
        )
        if view_mode == "insula":
            for label in surface_ctx.labels:
                if label.hemi == hemi and any(p in label.name for p in INSULA_LABEL_PATTERNS):
                    brain.add_label(label, borders=False, color=(0.88, 0.88, 0.88), alpha=0.45)

        side_mask = plot_df["x"].lt(0) if hemi == "lh" else plot_df["x"].gt(0)
        side = plot_df.loc[side_mask]
        coords = side[["x", "y", "z"]].to_numpy(float)
        if len(coords):
            projected = project_to_pial(coords, hemi, surface_ctx)
            _, indices = surface_ctx.trees[hemi].query(coords)
            keep = surface_ctx.valid_vertices[hemi][indices]
            for pt, val in zip(projected[keep], side["mean_diff"].to_numpy(float)[keep]):
                color = cmap(norm(val))[:3]
                cloud = pv.PolyData(pt.reshape(1, 3))
                brain._renderer.plotter.add_mesh(
                    cloud,
                    render_points_as_spheres=True,
                    point_size=_size(val),
                    color=color,
                    opacity=_alpha(val),
                    lighting=False,
                )

        if view_mode == "insula":
            brain.show_view(
                azimuth=180 if hemi == "lh" else 0,
                elevation=90,
                distance=180,
                focalpoint=surface_ctx.insula_centers[hemi],
            )
        else:
            brain.show_view(view="lateral", distance=400)
        screenshots.append(brain.screenshot(mode="rgb"))
        panel_titles.append(panel_title)
        brain.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, image, panel_title in zip(axes, screenshots, panel_titles):
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(panel_title, fontsize=8)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return fig
