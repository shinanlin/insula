#!/usr/bin/env python3
"""Overlay significant OAEC electrodes on the focused combined pial map.

Does not modify ``plot_nmf_whole_brain_projection.py``.  Background is the
existing focused npz cache; electrodes are unique Insula seeds and extra-insula
partners from within-pair OAEC (same HGA-sig rule as the ROI-family brains).
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")
os.environ.setdefault("PYVISTA_USE_PANEL", "false")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.sparse import coo_matrix

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.nmf.oaec_focused_overlay import (
    assign_focused_membership,
    focused_mask_payload,
    mark_partner_significance,
    restrict_overlay_to_focused_partners,
    select_overlay_electrodes,
)
from src.nmf.waveform_analysis import CLUSTER_LABELS, CLUSTER_ORDER, FUNCTION_COLORS
from src.nmf.whole_brain_projection_viz import (
    combined_focused_rgba,
    offset_vertices_along_normals,
    refocus_display_masks,
    retain_largest_surface_clusters,
    surface_label_boundary_edges,
)
from src.paths import img_dir, nmf_wholebrain_dir, save_svg

logger = logging.getLogger(__name__)

CM = 1 / 2.54
plt.rcParams["svg.fonttype"] = "none"
COMPONENTS = tuple(CLUSTER_ORDER)
COMPONENT_LABELS = dict(CLUSTER_LABELS)
UNASSIGNED_SEED_COLOR = "#D4AF37"
UNASSIGNED_PARTNER_COLOR = "#BDBDBD"
NONSIG_COLOR = "#111111"
ELECTRODE_OFFSET_MM = 1.2
SEED_POINT_SIZE = 13.0
FOCUSED_PARTNER_SIZE = 9.0
OTHER_PARTNER_SIZE = 5.0
MAX_FOCUSED_DISTANCE = 15.0
MOTIF_CONTOUR_OFFSET_MM = 0.7
MOTIF_CONTOUR_WIDTH = 2.2
CONTOUR_WHITE_MIX = 0.58
CONTOUR_CLOSE_STEPS = 2
CONTOUR_MAX_CLUSTERS = 3
CONTOUR_MIN_VERTICES = 150


def _load_script(name: str, relative: str):
    path = PROJECT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _flag(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def _electrode_color(row: pd.Series) -> str:
    if "plot_color" in row.index and pd.notna(row.get("plot_color")):
        return str(row["plot_color"])
    if row["role"] == "seed":
        cluster = row.get("seed_cluster")
        if pd.notna(cluster) and str(cluster) in FUNCTION_COLORS:
            return FUNCTION_COLORS[str(cluster)]
        return UNASSIGNED_SEED_COLOR
    if "has_sig" in row.index and not _flag(row.get("has_sig", True)):
        return NONSIG_COLOR
    cluster = row.get("insula_cluster")
    if pd.notna(cluster) and str(cluster) in FUNCTION_COLORS:
        return FUNCTION_COLORS[str(cluster)]
    return UNASSIGNED_PARTNER_COLOR


def _electrode_size(row: pd.Series) -> float:
    if row["role"] == "seed":
        return SEED_POINT_SIZE
    if "has_sig" in row.index and not _flag(row.get("has_sig", True)):
        return OTHER_PARTNER_SIZE
    if _flag(row.get("in_focused", False)):
        return FOCUSED_PARTNER_SIZE
    return OTHER_PARTNER_SIZE


def _add_electrode_spheres(brain, hemi: str, frame: pd.DataFrame) -> None:
    import pyvista as pv

    if frame.empty:
        return
    coords = np.asarray(brain.geo[hemi].coords, dtype=float)
    faces = np.asarray(brain.geo[hemi].faces, dtype=np.int64)
    lifted = offset_vertices_along_normals(coords, faces, ELECTRODE_OFFSET_MM)
    work = frame.loc[frame["vertex_index"].ge(0)].copy()
    if work.empty:
        return
    work["color"] = work.apply(_electrode_color, axis=1)
    work["point_size"] = work.apply(_electrode_size, axis=1)
    # Smaller partners first so seeds remain visible on top.
    work = work.sort_values("point_size", ascending=True)
    for (color, size), group in work.groupby(["color", "point_size"], sort=False):
        index = group["vertex_index"].to_numpy(dtype=int)
        valid = (index >= 0) & (index < len(lifted))
        if not valid.any():
            continue
        cloud = pv.PolyData(lifted[index[valid]])
        brain._renderer.plotter.add_mesh(
            cloud,
            render_points_as_spheres=True,
            point_size=float(size),
            color=str(color),
            lighting=False,
            smooth_shading=False,
            opacity=0.95 if str(color) != NONSIG_COLOR else 0.90,
        )


def _pale_color(hex_color: str, white: float = CONTOUR_WHITE_MIX) -> tuple[float, float, float]:
    rgb = np.asarray(mcolors.to_rgb(hex_color), dtype=float)
    return tuple(white + (1.0 - white) * rgb)


def _mesh_adjacency(n_vertices: int, faces: np.ndarray):
    start = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    end = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    adj = coo_matrix(
        (np.ones(len(start), dtype=np.uint8), (start, end)),
        shape=(n_vertices, n_vertices),
    )
    adj = adj + adj.T
    adj.data[:] = 1
    return adj.tocsr()


def _close_surface_mask(mask: np.ndarray, adjacency, steps: int = CONTOUR_CLOSE_STEPS):
    closed = np.asarray(mask, dtype=bool)
    for _ in range(steps):
        closed = adjacency.dot(closed.astype(np.int32)) > 0
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    for _ in range(steps):
        neighbors = adjacency.dot(closed.astype(np.int32))
        closed = closed & (neighbors >= np.maximum(0.5 * degree, 1))
    return closed


def _contour_mask(mask: np.ndarray, faces: np.ndarray, adjacency) -> np.ndarray:
    closed = _close_surface_mask(mask, adjacency)
    return retain_largest_surface_clusters(
        closed,
        faces,
        max_clusters=CONTOUR_MAX_CLUSTERS,
        min_vertices=CONTOUR_MIN_VERTICES,
    )


def _add_motif_contours(brain, hemi: str, payload: dict[str, np.ndarray]) -> None:
    import pyvista as pv

    coords = np.asarray(brain.geo[hemi].coords, dtype=float)
    faces = np.asarray(brain.geo[hemi].faces, dtype=np.int64)
    lifted = offset_vertices_along_normals(coords, faces, MOTIF_CONTOUR_OFFSET_MM)
    adjacency = _mesh_adjacency(len(coords), faces)
    for component in COMPONENTS:
        raw = np.asarray(payload[f"{hemi}_{component}_display_mask"], dtype=bool)
        if len(raw) != len(coords):
            raise ValueError(f"{hemi} {component} mask does not match mesh")
        mask = _contour_mask(raw, faces, adjacency)
        edges = surface_label_boundary_edges(faces, mask)
        if len(edges) == 0:
            continue
        cells = np.empty(len(edges) * 3, dtype=np.int64)
        cells[0::3] = 2
        cells[1::3] = edges[:, 0]
        cells[2::3] = edges[:, 1]
        mesh = pv.PolyData()
        mesh.points = lifted
        mesh.lines = cells
        brain._renderer.plotter.add_mesh(
            mesh,
            color=_pale_color(FUNCTION_COLORS[component]),
            line_width=MOTIF_CONTOUR_WIDTH,
            lighting=False,
            render_lines_as_tubes=False,
        )


def _render_views(
    *,
    nmf_plot,
    payload: dict[str, np.ndarray],
    thresholds: dict[str, float],
    value_max: float,
    electrodes: pd.DataFrame,
    hemi: str,
    surf: str,
    views: tuple[str, ...],
    overlay_alpha: float = 1.0,
    territory_style: str = "fill",
    draw_landmarks: bool = True,
) -> dict[str, np.ndarray]:
    import mne
    from mne.viz import Brain

    brain = Brain(
        nmf_plot.TEMPLATE_SUBJECT,
        subjects_dir=str(nmf_plot.RECON_DIR),
        surf=surf,
        hemi=hemi,
        background="white",
        show=False,
        cortex=(0.88, 0.88, 0.88),
        alpha=1.0,
        size=(550, 450),
    )
    if territory_style == "contour":
        _add_motif_contours(brain, hemi, payload)
    else:
        rgba = combined_focused_rgba(payload, hemi, thresholds, value_max)
        if overlay_alpha != 1.0:
            rgba = np.array(rgba, copy=True)
            rgba[:, 3] *= float(overlay_alpha)
        nmf_plot._add_rgba_overlay(brain, hemi, rgba)
        if draw_landmarks:
            nmf_plot._add_landmark_outlines(brain, hemi)
    if "surf_hemi" in electrodes.columns:
        local = electrodes.loc[electrodes["surf_hemi"].astype(str).eq(hemi)]
    else:
        hemi_code = "L" if hemi == "lh" else "R"
        local = electrodes.loc[electrodes["hemi"].astype(str).isin({hemi_code, hemi})]
    _add_electrode_spheres(brain, hemi, local)
    distance = 550.0 if surf == "inflated" else 450.0
    images: dict[str, np.ndarray] = {}
    for view in views:
        brain.show_view(view=view, distance=distance)
        images[view] = brain.screenshot(mode="rgb")
    brain.close()
    return images


def _legend_handles(
    *,
    in_cluster_only: bool = False,
    hide_seeds: bool = False,
    nonsig_gray: bool = False,
    territory_style: str = "fill",
) -> list:
    if territory_style == "contour":
        handles = [
            Line2D(
                [0],
                [0],
                color=_pale_color(FUNCTION_COLORS[name]),
                linewidth=2.2,
                label=f"{COMPONENT_LABELS[name]} territory",
            )
            for name in COMPONENTS
        ]
    else:
        handles = [
            Patch(facecolor=FUNCTION_COLORS[name], label=f"{COMPONENT_LABELS[name]} territory")
            for name in COMPONENTS
        ]
    if not hide_seeds:
        handles.extend(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=FUNCTION_COLORS[name],
                markeredgecolor=FUNCTION_COLORS[name],
                markersize=7,
                label=f"Insula seed ({COMPONENT_LABELS[name]})",
            )
            for name in COMPONENTS
        )
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=UNASSIGNED_SEED_COLOR,
                markeredgecolor=UNASSIGNED_SEED_COLOR,
                markersize=7,
                label="Insula seed (unassigned)",
            )
        )
    if hide_seeds:
        handles.extend(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=FUNCTION_COLORS[name],
                markeredgecolor=FUNCTION_COLORS[name],
                markersize=6,
                label=f"OAEC to {COMPONENT_LABELS[name]} insula",
            )
            for name in COMPONENTS
        )
    else:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#2369BD",
                markeredgecolor="#2369BD",
                markersize=6,
                label="OAEC partner in territory",
            )
        )
    if not in_cluster_only:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=UNASSIGNED_PARTNER_COLOR,
                markeredgecolor=UNASSIGNED_PARTNER_COLOR,
                markersize=4,
                alpha=0.7,
                label="OAEC partner outside territory",
            )
        )
    if nonsig_gray:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=NONSIG_COLOR,
                markeredgecolor=NONSIG_COLOR,
                markersize=4,
                alpha=0.7,
                label="Never-sig partner in territory",
            )
        )
    return handles


def plot_overlay(
    *,
    payload: dict[str, np.ndarray],
    thresholds: dict[str, float],
    value_max: float,
    electrodes: pd.DataFrame,
    output: Path,
    surf: str,
    nmf_plot,
    subtitle: str,
    in_cluster_only: bool = False,
    hide_seeds: bool = False,
    nonsig_gray: bool = False,
    overlay_alpha: float = 1.0,
    territory_style: str = "fill",
    draw_landmarks: bool = True,
) -> Path:
    import mne

    mne.viz.set_3d_backend("notebook")
    views = (("lateral", "Lateral"), ("medial", "Medial"))
    hemis = (("lh", "Left"), ("rh", "Right"))
    screenshots = {
        hemi: _render_views(
            nmf_plot=nmf_plot,
            payload=payload,
            thresholds=thresholds,
            value_max=value_max,
            electrodes=electrodes,
            hemi=hemi,
            surf=surf,
            views=tuple(name for name, _ in views),
            overlay_alpha=overlay_alpha,
            territory_style=territory_style,
            draw_landmarks=draw_landmarks,
        )
        for hemi, _ in hemis
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.0 * CM, 12.2 * CM))
    for row, (view, view_label) in enumerate(views):
        for column, (hemi, hemi_label) in enumerate(hemis):
            axes[row, column].imshow(screenshots[hemi][view])
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(hemi_label, fontsize=9)
            if column == 0:
                axes[row, column].text(
                    0.01,
                    0.5,
                    view_label,
                    transform=axes[row, column].transAxes,
                    rotation=90,
                    va="center",
                    fontsize=8,
                )
    fig.legend(
        handles=_legend_handles(
            in_cluster_only=in_cluster_only,
            hide_seeds=hide_seeds,
            nonsig_gray=nonsig_gray,
            territory_style=territory_style,
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.12),
        ncol=3,
        frameon=False,
        fontsize=6.0,
    )
    fig.suptitle(
        f"OAEC electrodes on focused motif territories ({surf})",
        x=0.02,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    fig.subplots_adjust(
        left=0.04, right=0.99, top=0.90, bottom=0.18, wspace=0.02, hspace=0.02
    )
    fig.text(0.02, 0.012, subtitle, fontsize=5.4)
    return save_svg(fig, output, close=True)


def _count_line(electrodes: pd.DataFrame) -> str:
    seeds = electrodes.loc[electrodes["role"].eq("seed")]
    partners = electrodes.loc[electrodes["role"].eq("partner")]
    focused_partners = partners.loc[partners["in_focused"].eq(True)]
    return (
        f"{len(seeds)} insula seeds | "
        f"{len(partners)} partners | "
        f"{len(focused_partners)} partners in focused territories"
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=nmf_wholebrain_dir() / "visualization" / "surface_motif_maps_focused.npz",
    )
    parser.add_argument(
        "--projection",
        type=Path,
        default=nmf_wholebrain_dir() / "electrode_projection.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=img_dir("nmf/whole_brain_projection")
        / "wholebrain_surface_motifs_focused_combined_pial_oaec_electrodes.svg",
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=nmf_wholebrain_dir() / "visualization" / "oaec_overlay_electrodes.csv",
    )
    parser.add_argument("--surf", choices=("pial", "inflated"), default="pial")
    parser.add_argument(
        "--in-cluster",
        action="store_true",
        help="Plot only partners inside focused territories and seeds that connect to them.",
    )
    parser.add_argument(
        "--hide-seeds",
        action="store_true",
        help="Do not draw Insula seed electrodes; partners only.",
    )
    parser.add_argument(
        "--nonsig-gray",
        action="store_true",
        help="Draw focused partners with no significant OAEC pair in gray.",
    )
    parser.add_argument(
        "--partner-hga-sig",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict partners to HGA-sig targets (default: true).",
    )
    parser.add_argument(
        "--keep-top",
        type=float,
        default=0.50,
        help="Retain this upper fraction of eligible vertices (matches the explore notebook).",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Write the electrode table only; skip the 3D brain render.",
    )
    args = parser.parse_args(argv)

    if not args.cache.is_file():
        raise FileNotFoundError(f"Focused surface cache not found: {args.cache}")

    nmf_plot = _load_script(
        "plot_nmf_whole_brain_projection",
        "scripts/plot_nmf_whole_brain_projection.py",
    )
    plot_oaec = _load_script(
        "plot_oaec_pooled_subject_lh",
        "scripts/plot_oaec_pooled_subject_lh.py",
    )
    payload, thresholds, value_max = nmf_plot.load_focused_surface_cache(args.cache)

    import mne

    vertices_by_hemi = {}
    faces_by_hemi = {}
    for hemi in ("lh", "rh"):
        vertices, faces = mne.read_surface(
            str(nmf_plot.RECON_DIR / nmf_plot.TEMPLATE_SUBJECT / "surf" / f"{hemi}.pial")
        )
        vertices_by_hemi[hemi] = vertices
        faces_by_hemi[hemi] = faces
    payload, thresholds = refocus_display_masks(
        payload,
        faces_by_hemi,
        keep_top=args.keep_top,
        components=COMPONENTS,
        min_subjects=3,
        min_positive_fraction=0.50,
        max_clusters=5,
        min_cluster_vertices=20,
    )
    mask_by_hemi, specificity_by_hemi = focused_mask_payload(payload)

    channel_meta, _exclude = plot_oaec.load_channel_meta()
    raw = plot_oaec.load_metric_repeat("oaec")
    pairs = plot_oaec.annotate_within_pair(
        raw,
        channel_meta,
        partner_hga_sig=args.partner_hga_sig,
        sig_only=not args.nonsig_gray,
    )
    projection = pd.read_csv(args.projection) if args.projection.is_file() else None
    electrodes = select_overlay_electrodes(pairs, channel_meta, projection)
    electrodes = assign_focused_membership(
        electrodes,
        vertices_by_hemi=vertices_by_hemi,
        mask_by_hemi=mask_by_hemi,
        specificity_by_hemi=specificity_by_hemi,
        max_distance=MAX_FOCUSED_DISTANCE,
    )
    from src.nmf.oaec_focused_overlay import assign_partner_insula_cluster
    plotted_pairs = pairs
    if args.in_cluster:
        plotted_pairs, electrodes = restrict_overlay_to_focused_partners(
            pairs, electrodes
        )
        logger.info(
            "in-cluster filter: %s pairs / %s electrodes retained",
            len(plotted_pairs),
            len(electrodes),
        )
    electrodes = assign_partner_insula_cluster(plotted_pairs, electrodes)
    if args.nonsig_gray:
        electrodes = mark_partner_significance(plotted_pairs, electrodes)
        sig_seeds = set(
            plotted_pairs.loc[
                plotted_pairs["sig_within_pair"].astype(bool), "source_channel"
            ].astype(str)
        )
        keep = electrodes["role"].ne("seed") | electrodes["channel"].astype(str).isin(
            sig_seeds
        )
        electrodes = electrodes.loc[keep].copy()
        n_gray = int(
            (electrodes["role"].eq("partner") & ~electrodes["has_sig"].astype(bool)).sum()
        )
        n_sig_p = int(
            (electrodes["role"].eq("partner") & electrodes["has_sig"].astype(bool)).sum()
        )
        logger.info("nonsig-gray: %s sig partners | %s never-sig partners", n_sig_p, n_gray)
    if args.hide_seeds:
        electrodes = electrodes.loc[electrodes["role"].ne("seed")].copy()
        logger.info("hide-seeds: %s partner electrodes retained", len(electrodes))
    args.table_output.parent.mkdir(parents=True, exist_ok=True)
    electrodes.to_csv(args.table_output, index=False)

    n_pairs = int(plotted_pairs["pair_key"].nunique()) if len(plotted_pairs) else 0
    n_subjects = int(plotted_pairs["subject"].nunique()) if len(plotted_pairs) else 0
    n_seeds = int(electrodes["role"].eq("seed").sum())
    n_partners = int(electrodes["role"].eq("partner").sum())
    n_focused = int(
        (electrodes["role"].eq("partner") & electrodes["in_focused"].eq(True)).sum()
    )
    hga = "partner HGA-sig" if args.partner_hga_sig else "all QC partners"
    cluster_note = (
        "Partners restricted to focused territories; insula seeds kept only if they connect to those partners. "
        if args.in_cluster
        else ""
    )
    seed_note = "Insula seed electrodes omitted. " if args.hide_seeds else ""
    color_note = (
        "Partners colored by whole-brain best component. "
        if args.hide_seeds
        else "Seeds colored by NMF cluster; partners by whole-brain best component. "
    )
    gray_note = ""
    if args.nonsig_gray and "has_sig" in electrodes.columns:
        n_sig_p = int(
            (electrodes["role"].eq("partner") & electrodes["has_sig"].astype(bool)).sum()
        )
        n_gray = int(
            (electrodes["role"].eq("partner") & ~electrodes["has_sig"].astype(bool)).sum()
        )
        n_sig_pairs = int(
            plotted_pairs.loc[plotted_pairs["sig_within_pair"].astype(bool), "pair_key"].nunique()
        ) if len(plotted_pairs) else 0
        gray_note = (
            f"Gray: focused HGA-sig partners with no significant pair ({n_gray}). "
            f"Colored partners: ≥1 sig pair ({n_sig_p}). "
        )
        counts = (
            f"{n_subjects} subjects, {n_sig_pairs} significant pairs, {n_seeds} insula seeds, "
            f"{n_partners} focused partners ({n_sig_p} sig / {n_gray} never-sig). "
        )
    elif args.hide_seeds:
        counts = (
            f"{n_subjects} subjects, {n_pairs} unique pairs, "
            f"{n_partners} partners in focused territories. "
        )
    else:
        counts = (
            f"{n_subjects} subjects, {n_pairs} unique pairs, {n_seeds} insula seeds, "
            f"{n_partners} partners ({n_focused} in focused territories). "
        )
    subtitle = (
        "Background: same focused masks as wholebrain_surface_motifs_focused_combined_pial.svg. "
        f"Points: unique electrodes on Insula→partner OAEC (phase union, {hga}). "
        f"{cluster_note}{seed_note}{gray_note}{counts}{color_note}"
        "No pair lines."
    )
    logger.info("%s", _count_line(electrodes))
    logger.info("pairs=%s subjects=%s table=%s", n_pairs, n_subjects, args.table_output)
    if args.skip_plot:
        logger.info("skip-plot: wrote %s", args.table_output)
        return 0
    written = plot_overlay(
        payload=payload,
        thresholds=thresholds,
        value_max=value_max,
        electrodes=electrodes,
        output=args.output,
        surf=args.surf,
        nmf_plot=nmf_plot,
        subtitle=subtitle,
        in_cluster_only=args.in_cluster,
        hide_seeds=args.hide_seeds,
        nonsig_gray=args.nonsig_gray,
    )
    logger.info("Wrote %s", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
