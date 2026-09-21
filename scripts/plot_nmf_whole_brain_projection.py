#!/usr/bin/env python3
"""Visualize where whole-brain HGA resembles the three frozen Insula motifs."""

from __future__ import annotations

import argparse
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
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.nmf.waveform_analysis import CLUSTER_LABELS, CLUSTER_ORDER, FUNCTION_COLORS
from src.nmf.whole_brain_projection_viz import (
    add_template_specificity,
    combined_focused_rgba,
    prepare_projection_frame,
    retain_largest_surface_clusters,
    subject_balanced_surface,
    subject_balanced_surface_values,
    subject_roi_scores,
    summarize_rois,
    surface_label_boundary_edges,
    offset_vertices_along_normals,
)
from src.paths import img_dir, nmf_wholebrain_dir, save_svg

RECON_DIR = Path("/cwork/ns458/ECoG_Recon")
TEMPLATE_SUBJECT = "cvs_avg35_inMNI152"
COMPONENTS = tuple(CLUSTER_ORDER)
COMPONENT_LABELS = dict(CLUSTER_LABELS)
CM = 1 / 2.54
LANDMARK_ROIS = (
    ("HG", "transversetemporal", "#111111"),
    ("STG", "superiortemporal", "#1B4332"),
    ("PrG", "precentral", "#1D3557"),
    ("PoG", "postcentral", "#6B3F2A"),
    ("Insula", "insula", "#4A4A4A"),
)
OVERLAY_NORMAL_OFFSET_MM = 0.5
OUTLINE_NORMAL_OFFSET_MM = 0.7


def _component_cmap(component: str, *, negative: bool = False):
    color = FUNCTION_COLORS[component]
    if negative:
        return LinearSegmentedColormap.from_list(
            f"centered_{component}", ["#5D6570", "#FFFFFF", color]
        )
    return LinearSegmentedColormap.from_list(
        f"expression_{component}", ["#FFFFFF", color]
    )


def _rgba_matrix(values: np.ndarray, *, centered: bool) -> np.ndarray:
    rgba = np.ones((*values.shape, 4), dtype=float)
    if centered:
        limit = max(0.01, float(np.nanmax(np.abs(values))))
        normalized = np.clip((values + limit) / (2 * limit), 0.0, 1.0)
    else:
        limit = max(0.01, float(np.nanmax(values)))
        normalized = np.clip(values / limit, 0.0, 1.0)
    for column, component in enumerate(COMPONENTS):
        cmap = _component_cmap(component, negative=centered)
        rgba[:, column] = cmap(normalized[:, column])
    return rgba


def _ordered_rois(summary: pd.DataFrame) -> list[str]:
    matrix = summary.pivot(index="roi", columns="component", values="mean_enrichment")
    matrix = matrix.reindex(columns=COMPONENTS)
    winner = matrix.to_numpy().argmax(axis=1)
    strength = matrix.to_numpy().max(axis=1)
    ordering = pd.DataFrame(
        {"roi": matrix.index, "winner": winner, "strength": strength}
    ).sort_values(["winner", "strength"], ascending=[True, False])
    return ordering["roi"].tolist()


def plot_roi_heatmap(summary: pd.DataFrame, output: Path) -> None:
    """Plot subject-balanced ROI expression and within-subject enrichment."""

    rois = _ordered_rois(summary)
    score = (
        summary.pivot(index="roi", columns="component", values="mean_score")
        .reindex(index=rois, columns=COMPONENTS)
        .to_numpy()
    )
    enrichment = (
        summary.pivot(index="roi", columns="component", values="mean_enrichment")
        .reindex(index=rois, columns=COMPONENTS)
        .to_numpy()
    )
    coverage = (
        summary.drop_duplicates("roi").set_index("roi").reindex(rois)[
            ["n_subjects", "n_contacts"]
        ]
    )

    height = max(12.0, 0.47 * len(rois) + 3.6)
    fig = plt.figure(figsize=(18.0 * CM, height * CM))
    grid = fig.add_gridspec(
        1, 3, width_ratios=(3.0, 3.0, 1.35), wspace=0.12
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    for ax, values, centered, title in zip(
        axes,
        (score, enrichment),
        (False, True),
        ("A  Template-expression score", "B  Within-subject enrichment"),
    ):
        ax.imshow(_rgba_matrix(values, centered=centered), aspect="auto")
        ax.set_xticks(range(len(COMPONENTS)))
        ax.set_xticklabels(
            [COMPONENT_LABELS[name] for name in COMPONENTS],
            rotation=35,
            ha="right",
            fontsize=7,
        )
        ax.set_yticks(range(len(rois)))
        ax.set_yticklabels(rois if ax is axes[0] else [], fontsize=6.5)
        ax.tick_params(length=0)
        ax.set_title(title, loc="left", fontsize=8, pad=8)
        for row in range(len(rois)):
            for column in range(len(COMPONENTS)):
                value = values[row, column]
                label = f"{value:.2f}" if not centered else f"{value:+.2f}"
                ax.text(column, row, label, ha="center", va="center", fontsize=5.4)
        ax.set_xticks(np.arange(-0.5, len(COMPONENTS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rois), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    coverage_ax = fig.add_subplot(grid[0, 2])
    coverage_ax.set_xlim(-0.5, 1.5)
    coverage_ax.set_ylim(len(rois) - 0.5, -0.5)
    coverage_ax.set_xticks([0, 1])
    coverage_ax.set_xticklabels(["Subjects", "Contacts"], rotation=35, ha="right", fontsize=7)
    coverage_ax.set_yticks([])
    coverage_ax.tick_params(length=0)
    coverage_ax.set_title("Coverage", loc="left", fontsize=8, pad=8)
    for row, (_, values) in enumerate(coverage.iterrows()):
        coverage_ax.text(0, row, f"{int(values.n_subjects)}", ha="center", va="center", fontsize=5.5)
        coverage_ax.text(1, row, f"{int(values.n_contacts)}", ha="center", va="center", fontsize=5.5)
    for spine in coverage_ax.spines.values():
        spine.set_visible(False)

    fig.suptitle(
        "Whole-brain expression of frozen Insula HGA motifs",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.012,
        "Held-out contacts only; mixed/unknown ROIs excluded. Contacts are averaged within subject × ROI first. "
        "Score = explained energy × NNLS proportion; enrichment = ROI score − subject's sampled-brain mean.",
        fontsize=5.8,
        ha="left",
    )
    save_svg(fig, output, close=True)


def _surface_cmap(component: str) -> ListedColormap:
    base = np.asarray(mcolors.to_rgb(FUNCTION_COLORS[component]))
    colors = np.ones((256, 4), dtype=float)
    colors[0] = [1, 1, 1, 0]
    for index in range(1, 256):
        fraction = index / 255
        colors[index, :3] = (1 - fraction) * np.ones(3) + fraction * base
        colors[index, 3] = 0.92
    return ListedColormap(colors)


def _render_surface_panel(
    data: np.ndarray,
    coverage: np.ndarray,
    *,
    component: str,
    hemi: str,
    minimum_coverage: int,
    value_max: float,
) -> np.ndarray:
    import mne
    from mne.viz import Brain

    values = np.zeros(len(data), dtype=float)
    valid = (coverage >= minimum_coverage) & np.isfinite(data)
    values[valid] = np.clip(data[valid] / max(value_max, 1e-12), 0.0, 1.0) * 255
    brain = Brain(
        TEMPLATE_SUBJECT,
        subjects_dir=str(RECON_DIR),
        surf="inflated",
        hemi=hemi,
        background="white",
        show=False,
        cortex=(0.88, 0.88, 0.88),
        alpha=0.28,
        size=(550, 450),
    )
    brain.add_data(
        values,
        hemi=hemi,
        colormap=_surface_cmap(component),
        alpha=1.0,
        colorbar=False,
        fmin=0,
        fmax=255,
    )
    brain.show_view(view="lateral", distance=550.0)
    image = brain.screenshot(mode="rgb")
    brain.close()
    return image


def _render_focused_surface_panel(
    data: np.ndarray,
    display_mask: np.ndarray,
    *,
    component: str,
    hemi: str,
    threshold: float,
    value_max: float,
) -> np.ndarray:
    """Render only reliable, high-specificity vertices."""

    import mne
    from mne.viz import Brain

    values = np.zeros(len(data), dtype=float)
    valid = np.asarray(display_mask, dtype=bool) & np.isfinite(data)
    denominator = max(value_max - threshold, 1e-12)
    # The threshold already carries the screening burden; start retained
    # vertices at a visible quarter-scale color instead of making them nearly
    # white again.
    values[valid] = 64 + np.clip(
        (data[valid] - threshold) / denominator, 0.0, 1.0
    ) * 191
    brain = Brain(
        TEMPLATE_SUBJECT,
        subjects_dir=str(RECON_DIR),
        surf="inflated",
        hemi=hemi,
        background="white",
        show=False,
        cortex=(0.88, 0.88, 0.88),
        alpha=0.28,
        size=(550, 450),
    )
    brain.add_data(
        values,
        hemi=hemi,
        colormap=_surface_cmap(component),
        alpha=1.0,
        colorbar=False,
        fmin=0,
        fmax=255,
    )
    brain.show_view(view="lateral", distance=550.0)
    image = brain.screenshot(mode="rgb")
    brain.close()
    return image


def _add_rgba_overlay(brain, hemi: str, rgba: np.ndarray) -> None:
    import pyvista as pv

    coords = np.asarray(brain.geo[hemi].coords, dtype=float)
    faces = np.asarray(brain.geo[hemi].faces, dtype=np.int64)
    if len(coords) != len(rgba):
        raise ValueError(
            f"{hemi} overlay length {len(rgba)} does not match mesh {len(coords)}"
        )
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"{hemi} faces must have shape (n_faces, 3)")
    visible = np.asarray(rgba[:, 3], dtype=float) > 1e-6
    keep = visible[faces].any(axis=1)
    if not keep.any():
        return
    lifted = offset_vertices_along_normals(
        coords, faces, OVERLAY_NORMAL_OFFSET_MM
    )
    cells = np.hstack(
        [np.full((int(keep.sum()), 1), 3, dtype=np.int64), faces[keep]]
    ).ravel()
    mesh = pv.PolyData(lifted, cells)
    mesh["rgba"] = (np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8)
    brain._renderer.plotter.add_mesh(
        mesh,
        scalars="rgba",
        rgba=True,
        lighting=False,
        smooth_shading=True,
    )


def _aparc_label_map(hemi: str) -> dict[str, object]:
    import mne

    labels = mne.read_labels_from_annot(
        TEMPLATE_SUBJECT,
        parc="aparc",
        subjects_dir=str(RECON_DIR),
        hemi=hemi,
    )
    mapped: dict[str, object] = {}
    for label in labels:
        stem = label.name.rsplit("-", 1)[0]
        mapped[stem] = label
    return mapped


def _add_landmark_outlines(brain, hemi: str) -> None:
    """Draw aparc ROI borders on top of the displayed mesh."""

    import pyvista as pv

    coords = np.asarray(brain.geo[hemi].coords, dtype=float)
    faces = np.asarray(brain.geo[hemi].faces, dtype=np.int64)
    lifted = offset_vertices_along_normals(
        coords, faces, OUTLINE_NORMAL_OFFSET_MM
    )
    labels = _aparc_label_map(hemi)
    for _, aparc_name, color in LANDMARK_ROIS:
        label = labels.get(aparc_name)
        if label is None:
            continue
        mask = np.zeros(len(coords), dtype=bool)
        vertices = np.asarray(label.vertices, dtype=int)
        valid = (vertices >= 0) & (vertices < len(coords))
        mask[vertices[valid]] = True
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
            color=color,
            line_width=2.4,
            lighting=False,
            render_lines_as_tubes=False,
        )


def _render_combined_focused_views(
    payload: dict[str, np.ndarray],
    *,
    hemi: str,
    thresholds: dict[str, float],
    value_max: float,
    surf: str = "inflated",
    views: tuple[str, ...] = ("lateral", "medial"),
) -> dict[str, np.ndarray]:
    """One hemisphere, screenshot at each requested view."""

    import mne
    from mne.viz import Brain

    brain = Brain(
        TEMPLATE_SUBJECT,
        subjects_dir=str(RECON_DIR),
        surf=surf,
        hemi=hemi,
        background="white",
        show=False,
        cortex=(0.88, 0.88, 0.88),
        alpha=1.0,
        size=(550, 450),
    )
    _add_rgba_overlay(
        brain,
        hemi,
        combined_focused_rgba(payload, hemi, thresholds, value_max),
    )
    _add_landmark_outlines(brain, hemi)
    distance = 550.0 if surf == "inflated" else 450.0
    images: dict[str, np.ndarray] = {}
    for view in views:
        brain.show_view(view=view, distance=distance)
        images[view] = brain.screenshot(mode="rgb")
    brain.close()
    return images


def plot_combined_focused_surface_maps(
    payload: dict[str, np.ndarray],
    output: Path,
    *,
    thresholds: dict[str, float],
    value_max: float,
    min_surface_subjects: int,
    min_positive_fraction: float,
    surface_quantile: float,
    max_clusters: int,
    surf: str = "inflated",
) -> None:
    """2x2 brains: lateral then medial, left then right, with landmark ROI outlines."""

    import mne

    mne.viz.set_3d_backend("notebook")
    views = (("lateral", "Lateral"), ("medial", "Medial"))
    hemis = (("lh", "Left"), ("rh", "Right"))
    screenshots = {
        hemi: _render_combined_focused_views(
            payload,
            hemi=hemi,
            thresholds=thresholds,
            value_max=value_max,
            surf=surf,
            views=tuple(name for name, _ in views),
        )
        for hemi, _ in hemis
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.0 * CM, 11.6 * CM))
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
    handles = [
        Patch(
            facecolor=FUNCTION_COLORS[name],
            label=COMPONENT_LABELS[name],
        )
        for name in COMPONENTS
    ]
    handles.extend(
        Line2D([0], [0], color=color, lw=1.8, label=name)
        for name, _, color in LANDMARK_ROIS
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.11),
        ncol=4,
        frameon=False,
        fontsize=6.5,
    )
    fig.suptitle(
        f"Focused cortical territories on one brain ({surf})",
        x=0.02,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    fig.subplots_adjust(
        left=0.04, right=0.99, top=0.90, bottom=0.16, wspace=0.02, hspace=0.02
    )
    threshold_text = ", ".join(
        f"{COMPONENT_LABELS[name]} {thresholds[name]:.2f}" for name in COMPONENTS
    )
    fig.text(
        0.02,
        0.012,
        "Same focused masks as wholebrain_surface_motifs_focused.svg; "
        "overlapping vertices take the higher-specificity template. "
        "Outlines: Desikan HG / STG / PrG / PoG / insula. "
        "Top: lateral; bottom: medial. "
        f"≥{min_surface_subjects} subjects; positive in ≥{100 * min_positive_fraction:.0f}% "
        f"of contributing subjects; top {100 * (1 - surface_quantile):.0f}% retained; "
        f"up to {max_clusters} mesh clusters/hemisphere. Thresholds: {threshold_text}.",
        fontsize=5.5,
    )
    save_svg(fig, output, close=True)


def plot_surface_maps(
    frame: pd.DataFrame,
    output: Path,
    cache: Path,
    *,
    bandwidth: float,
    max_distance: float,
    min_surface_subjects: int,
) -> None:
    """Render bilateral, subject-balanced cortical surface maps."""

    import mne

    mne.viz.set_3d_backend("notebook")
    payload: dict[str, np.ndarray] = {}
    for hemi, hemi_label in (("lh", "L"), ("rh", "R")):
        vertices, _ = mne.read_surface(
            str(RECON_DIR / TEMPLATE_SUBJECT / "surf" / f"{hemi}.pial")
        )
        hemi_frame = frame.loc[frame["hemi"].astype(str).eq(hemi_label)]
        for component in COMPONENTS:
            data, coverage = subject_balanced_surface(
                hemi_frame,
                vertices,
                component,
                bandwidth=bandwidth,
                max_distance=max_distance,
            )
            payload[f"{hemi}_{component}_score"] = data
            payload[f"{hemi}_{component}_coverage"] = coverage

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **payload)
    valid_values = [
        payload[f"{hemi}_{component}_score"][
            payload[f"{hemi}_{component}_coverage"] >= min_surface_subjects
        ]
        for hemi in ("lh", "rh")
        for component in COMPONENTS
    ]
    finite = np.concatenate([values[np.isfinite(values)] for values in valid_values])
    value_max = float(np.quantile(finite, 0.98)) if len(finite) else 1.0

    fig, axes = plt.subplots(2, 3, figsize=(18.0 * CM, 10.0 * CM))
    for row, hemi in enumerate(("lh", "rh")):
        for column, component in enumerate(COMPONENTS):
            image = _render_surface_panel(
                payload[f"{hemi}_{component}_score"],
                payload[f"{hemi}_{component}_coverage"],
                component=component,
                hemi=hemi,
                minimum_coverage=min_surface_subjects,
                value_max=value_max,
            )
            axes[row, column].imshow(image)
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(
                    COMPONENT_LABELS[component],
                    color=FUNCTION_COLORS[component],
                    fontsize=9,
                    fontweight="bold",
                )
            if column == 0:
                axes[row, column].text(
                    0.01,
                    0.5,
                    "Left" if hemi == "lh" else "Right",
                    transform=axes[row, column].transAxes,
                    rotation=90,
                    va="center",
                    fontsize=8,
                )

    fig.suptitle(
        "Subject-balanced cortical template expression",
        x=0.02,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.015,
        f"Local Gaussian mean (bandwidth {bandwidth:g} mm; radius {max_distance:g} mm); "
        f"vertices shown only where ≥{min_surface_subjects} subjects contribute. Common scale 0–{value_max:.2f}.",
        fontsize=6,
    )
    save_svg(fig, output, close=True)


def plot_focused_surface_maps(
    frame: pd.DataFrame,
    output: Path,
    cache: Path,
    *,
    bandwidth: float,
    max_distance: float,
    min_surface_subjects: int,
    min_positive_fraction: float,
    surface_quantile: float,
    max_clusters: int,
    min_cluster_vertices: int,
) -> None:
    """Render the most reproducible template-specific cortical territories."""

    import mne

    if not 0 <= min_positive_fraction <= 1:
        raise ValueError("min_positive_fraction must be in [0, 1]")
    if not 0 <= surface_quantile < 1:
        raise ValueError("surface_quantile must be in [0, 1)")

    specific = add_template_specificity(frame)
    mne.viz.set_3d_backend("notebook")
    payload: dict[str, np.ndarray] = {}
    faces_by_hemi: dict[str, np.ndarray] = {}
    for hemi, hemi_label in (("lh", "L"), ("rh", "R")):
        vertices, faces = mne.read_surface(
            str(RECON_DIR / TEMPLATE_SUBJECT / "surf" / f"{hemi}.pial")
        )
        faces_by_hemi[hemi] = faces
        hemi_frame = specific.loc[specific["hemi"].astype(str).eq(hemi_label)]
        for component in COMPONENTS:
            data, coverage, positive_fraction = subject_balanced_surface_values(
                hemi_frame,
                vertices,
                f"specificity_{component}",
                bandwidth=bandwidth,
                max_distance=max_distance,
            )
            payload[f"{hemi}_{component}_specificity"] = data
            payload[f"{hemi}_{component}_coverage"] = coverage
            payload[f"{hemi}_{component}_positive_fraction"] = positive_fraction

    thresholds: dict[str, float] = {}
    eligible_by_component: dict[str, np.ndarray] = {}
    for component in COMPONENTS:
        eligible_parts: list[np.ndarray] = []
        for hemi in ("lh", "rh"):
            data = payload[f"{hemi}_{component}_specificity"]
            coverage = payload[f"{hemi}_{component}_coverage"]
            positive_fraction = payload[
                f"{hemi}_{component}_positive_fraction"
            ]
            eligible = (
                (coverage >= min_surface_subjects)
                & (positive_fraction >= min_positive_fraction)
                & np.isfinite(data)
                & (data > 0)
            )
            eligible_parts.append(data[eligible])
        combined = np.concatenate(eligible_parts)
        eligible_by_component[component] = combined
        thresholds[component] = (
            float(np.quantile(combined, surface_quantile)) if len(combined) else np.inf
        )

    all_eligible = np.concatenate(
        [values for values in eligible_by_component.values() if len(values)]
    )
    value_max = float(np.quantile(all_eligible, 0.99)) if len(all_eligible) else 1.0
    for component in COMPONENTS:
        threshold = thresholds[component]
        for hemi in ("lh", "rh"):
            data = payload[f"{hemi}_{component}_specificity"]
            coverage = payload[f"{hemi}_{component}_coverage"]
            positive_fraction = payload[
                f"{hemi}_{component}_positive_fraction"
            ]
            candidate = (
                (coverage >= min_surface_subjects)
                & (positive_fraction >= min_positive_fraction)
                & np.isfinite(data)
                & (data >= threshold)
            )
            payload[f"{hemi}_{component}_display_mask"] = (
                retain_largest_surface_clusters(
                    candidate,
                    faces_by_hemi[hemi],
                    max_clusters=max_clusters,
                    min_vertices=min_cluster_vertices,
                )
            )
    payload["display_thresholds"] = np.array(
        [thresholds[component] for component in COMPONENTS]
    )
    payload["display_value_max"] = np.array([value_max])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **payload)

    fig, axes = plt.subplots(2, 3, figsize=(18.0 * CM, 10.0 * CM))
    for row, hemi in enumerate(("lh", "rh")):
        for column, component in enumerate(COMPONENTS):
            image = _render_focused_surface_panel(
                payload[f"{hemi}_{component}_specificity"],
                payload[f"{hemi}_{component}_display_mask"],
                component=component,
                hemi=hemi,
                threshold=thresholds[component],
                value_max=value_max,
            )
            axes[row, column].imshow(image)
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(
                    COMPONENT_LABELS[component],
                    color=FUNCTION_COLORS[component],
                    fontsize=9,
                    fontweight="bold",
                )
            if column == 0:
                axes[row, column].text(
                    0.01,
                    0.5,
                    "Left" if hemi == "lh" else "Right",
                    transform=axes[row, column].transAxes,
                    rotation=90,
                    va="center",
                    fontsize=8,
                )

    fig.suptitle(
        "Focused cortical territories: template-specific expression",
        x=0.02,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    threshold_text = ", ".join(
        f"{COMPONENT_LABELS[name]} {thresholds[name]:.2f}" for name in COMPONENTS
    )
    fig.text(
        0.02,
        0.015,
        f"Specificity = target score − best competing-template score. ≥{min_surface_subjects} subjects; "
        f"positive in ≥{100 * min_positive_fraction:.0f}% of contributing subjects; top "
        f"{100 * (1 - surface_quantile):.0f}% retained; up to {max_clusters} mesh clusters/hemisphere. "
        f"Thresholds: {threshold_text}.",
        fontsize=5.7,
    )
    save_svg(fig, output, close=True)
    _write_combined_surface_versions(
        payload,
        output.with_name(f"{output.stem}_combined.svg"),
        thresholds=thresholds,
        value_max=value_max,
        min_surface_subjects=min_surface_subjects,
        min_positive_fraction=min_positive_fraction,
        surface_quantile=surface_quantile,
        max_clusters=max_clusters,
    )


def _write_combined_surface_versions(
    payload: dict[str, np.ndarray],
    inflated_output: Path,
    **kwargs,
) -> list[Path]:
    """Write inflated and pial combined overlays from the same focused payload."""

    written = []
    for surf, path in (
        ("inflated", inflated_output),
        ("pial", inflated_output.with_name(f"{inflated_output.stem}_pial.svg")),
    ):
        plot_combined_focused_surface_maps(payload, path, surf=surf, **kwargs)
        written.append(path)
        print(f"Wrote {path}")
    return written


def load_focused_surface_cache(
    cache: Path,
) -> tuple[dict[str, np.ndarray], dict[str, float], float]:
    """Reload the focused surface payload written by ``plot_focused_surface_maps``."""

    with np.load(cache) as raw:
        payload = {key: raw[key] for key in raw.files}
    if "display_thresholds" not in payload or "display_value_max" not in payload:
        raise ValueError(f"{cache} is missing focused display metadata")
    thresholds = {
        name: float(payload["display_thresholds"][index])
        for index, name in enumerate(COMPONENTS)
    }
    value_max = float(np.asarray(payload["display_value_max"]).reshape(-1)[0])
    return payload, thresholds, value_max


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=nmf_wholebrain_dir() / "electrode_projection.csv",
    )
    parser.add_argument("--min-roi-subjects", type=int, default=10)
    parser.add_argument("--min-surface-subjects", type=int, default=5)
    parser.add_argument("--focused-min-subjects", type=int, default=3)
    parser.add_argument("--focused-positive-fraction", type=float, default=0.50)
    parser.add_argument("--focused-quantile", type=float, default=0.50)
    parser.add_argument("--focused-max-clusters", type=int, default=5)
    parser.add_argument("--focused-min-cluster-vertices", type=int, default=20)
    parser.add_argument("--bandwidth", type=float, default=8.0)
    parser.add_argument("--max-distance", type=float, default=15.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--skip-surfaces", action="store_true")
    parser.add_argument(
        "--combined-from-cache",
        action="store_true",
        help="Render only the 3-cluster overlay from the existing focused npz cache.",
    )
    args = parser.parse_args(argv)

    result_dir = nmf_wholebrain_dir() / "visualization"
    figure_dir = img_dir("nmf/whole_brain_projection")
    result_dir.mkdir(parents=True, exist_ok=True)

    if args.combined_from_cache:
        cache = result_dir / "surface_motif_maps_focused.npz"
        if not cache.exists():
            raise FileNotFoundError(
                f"Focused surface cache not found: {cache}. "
                "Run without --combined-from-cache first."
            )
        payload, thresholds, value_max = load_focused_surface_cache(cache)
        _write_combined_surface_versions(
            payload,
            figure_dir / "wholebrain_surface_motifs_focused_combined.svg",
            thresholds=thresholds,
            value_max=value_max,
            min_surface_subjects=args.focused_min_subjects,
            min_positive_fraction=args.focused_positive_fraction,
            surface_quantile=args.focused_quantile,
            max_clusters=args.focused_max_clusters,
        )
        return 0

    raw = pd.read_csv(args.input)
    prepared = prepare_projection_frame(raw)
    subject_roi = subject_roi_scores(prepared)
    summary = summarize_rois(
        subject_roi,
        min_subjects=args.min_roi_subjects,
        bootstrap_samples=args.bootstrap_samples,
    )

    prepared.to_csv(result_dir / "heldout_projection_scores.csv", index=False)
    subject_roi.to_csv(result_dir / "subject_roi_scores.csv", index=False)
    summary.to_csv(result_dir / "roi_motif_summary.csv", index=False)
    plot_roi_heatmap(summary, figure_dir / "wholebrain_roi_motifs.svg")
    print(
        f"ROI summary: {summary['roi'].nunique()} ROIs, "
        f"{prepared['subject'].nunique()} subjects, {len(prepared)} held-out contacts"
    )

    if not args.skip_surfaces:
        plot_surface_maps(
            prepared,
            figure_dir / "wholebrain_surface_motifs.svg",
            result_dir / "surface_motif_maps.npz",
            bandwidth=args.bandwidth,
            max_distance=args.max_distance,
            min_surface_subjects=args.min_surface_subjects,
        )
        plot_focused_surface_maps(
            prepared,
            figure_dir / "wholebrain_surface_motifs_focused.svg",
            result_dir / "surface_motif_maps_focused.npz",
            bandwidth=args.bandwidth,
            max_distance=args.max_distance,
            min_surface_subjects=args.focused_min_subjects,
            min_positive_fraction=args.focused_positive_fraction,
            surface_quantile=args.focused_quantile,
            max_clusters=args.focused_max_clusters,
            min_cluster_vertices=args.focused_min_cluster_vertices,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
