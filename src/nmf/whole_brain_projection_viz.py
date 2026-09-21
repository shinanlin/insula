"""Group summaries for fixed-template whole-brain HGA projection.

The projection table contains one NNLS mixture per electrode.  Group maps must
not simply count electrodes: SEEG sampling differs strongly across subjects and
regions.  This module therefore averages electrodes within subject first, then
averages subjects.  It also keeps the Insula electrodes used to discover the
templates out of the primary held-out summaries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from src.nmf.waveform_analysis import CLUSTER_ORDER

DEFAULT_COMPONENTS = tuple(CLUSTER_ORDER)


def retain_largest_surface_clusters(
    mask: np.ndarray,
    faces: np.ndarray,
    *,
    max_clusters: int = 3,
    min_vertices: int = 50,
) -> np.ndarray:
    """Remove isolated surface speckles and retain the largest mesh clusters."""

    mask = np.asarray(mask, dtype=bool)
    faces = np.asarray(faces, dtype=int)
    if mask.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (n_faces, 3)")
    if max_clusters < 1 or min_vertices < 1:
        raise ValueError("cluster limits must be positive")
    selected = np.flatnonzero(mask)
    if len(selected) == 0:
        return mask.copy()

    inverse = np.full(len(mask), -1, dtype=int)
    inverse[selected] = np.arange(len(selected))
    edge_start = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    edge_end = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    valid_edges = mask[edge_start] & mask[edge_end]
    rows = inverse[edge_start[valid_edges]]
    columns = inverse[edge_end[valid_edges]]
    graph = coo_matrix(
        (
            np.ones(2 * len(rows), dtype=np.uint8),
            (np.concatenate([rows, columns]), np.concatenate([columns, rows])),
        ),
        shape=(len(selected), len(selected)),
    ).tocsr()
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    eligible = np.flatnonzero(sizes >= min_vertices)
    if len(eligible) == 0:
        return np.zeros_like(mask)
    order = eligible[np.argsort(sizes[eligible])[::-1][:max_clusters]]
    keep_selected = np.isin(labels, order)
    keep = np.zeros_like(mask)
    keep[selected[keep_selected]] = True
    return keep


def refocus_display_masks(
    payload: Mapping[str, np.ndarray],
    faces_by_hemi: Mapping[str, np.ndarray],
    *,
    keep_top: float,
    components: Sequence[str] = DEFAULT_COMPONENTS,
    min_subjects: int = 3,
    min_positive_fraction: float = 0.50,
    max_clusters: int = 5,
    min_cluster_vertices: int = 20,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Rebuild focused display masks at a new keep-top fraction.

    Eligible vertices still require ``min_subjects`` coverage, a positive
    fraction of at least ``min_positive_fraction``, and finite positive
    specificity.  ``keep_top`` is the retained upper tail of that eligible
    pool (0.50 keeps the top half).
    """

    if not 0.0 < keep_top <= 1.0:
        raise ValueError("keep_top must be in (0, 1]")
    components = tuple(components)
    if not components:
        raise ValueError("components must be non-empty")
    out = dict(payload)
    surface_quantile = 1.0 - keep_top
    thresholds: dict[str, float] = {}
    for component in components:
        eligible_parts = []
        for hemi in ("lh", "rh"):
            data = np.asarray(out[f"{hemi}_{component}_specificity"], dtype=float)
            coverage = np.asarray(out[f"{hemi}_{component}_coverage"])
            positive_fraction = np.asarray(
                out[f"{hemi}_{component}_positive_fraction"]
            )
            eligible = (
                (coverage >= min_subjects)
                & (positive_fraction >= min_positive_fraction)
                & np.isfinite(data)
                & (data > 0)
            )
            eligible_parts.append(data[eligible])
        combined = np.concatenate(eligible_parts) if eligible_parts else np.array([])
        thresholds[component] = (
            float(np.quantile(combined, surface_quantile)) if len(combined) else np.inf
        )
        for hemi in ("lh", "rh"):
            if hemi not in faces_by_hemi:
                raise KeyError(f"faces_by_hemi missing {hemi}")
            data = np.asarray(out[f"{hemi}_{component}_specificity"], dtype=float)
            coverage = np.asarray(out[f"{hemi}_{component}_coverage"])
            positive_fraction = np.asarray(
                out[f"{hemi}_{component}_positive_fraction"]
            )
            candidate = (
                (coverage >= min_subjects)
                & (positive_fraction >= min_positive_fraction)
                & np.isfinite(data)
                & (data >= thresholds[component])
            )
            out[f"{hemi}_{component}_display_mask"] = retain_largest_surface_clusters(
                candidate,
                faces_by_hemi[hemi],
                max_clusters=max_clusters,
                min_vertices=min_cluster_vertices,
            )
    out["display_thresholds"] = np.array(
        [thresholds[name] for name in components], dtype=float
    )
    return out, thresholds


def _as_bool(series: pd.Series) -> pd.Series:
    """Coerce bool-like CSV values without treating ``"False"`` as true."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.astype("string")
        .fillna("false")
        .str.strip()
        .str.lower()
        .isin({"1", "true", "t", "yes", "y"})
    )


def prepare_projection_frame(
    frame: pd.DataFrame,
    *,
    components: Sequence[str] = DEFAULT_COMPONENTS,
    exclude_discovery: bool = True,
    exclude_mixed: bool = True,
    require_resolved_roi: bool = True,
    exclude_rois: Sequence[str] = ("LatV",),
) -> pd.DataFrame:
    """Filter the projection table and add continuous template-expression scores.

    For component ``k``, the primary electrode-level score is

    ``clip(explained_energy, 0, 1) * proportion_k``.

    The mixture proportion alone can look decisive even when the three frozen
    templates reconstruct the electrode poorly.  Multiplication by explained
    energy retains continuous evidence while down-weighting those poor fits.
    """

    components = tuple(components)
    required = {
        "subject",
        "channel",
        "roi",
        "explained_energy",
        *[f"proportion_{name}" for name in components],
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"projection table missing columns: {sorted(missing)}")

    out = frame.copy()
    keep = out["subject"].notna() & out["channel"].notna()
    if exclude_discovery and "in_discovery" in out:
        keep &= ~_as_bool(out["in_discovery"])
    if exclude_mixed and "mix" in out:
        keep &= ~_as_bool(out["mix"])
    if require_resolved_roi:
        roi = out["roi"].astype("string").str.strip()
        keep &= roi.notna() & ~roi.str.lower().isin({"", "unknown", "nan", "none"})
    if exclude_rois:
        keep &= ~out["roi"].astype("string").isin(set(exclude_rois))

    out = out.loc[keep].copy()
    energy = pd.to_numeric(out["explained_energy"], errors="coerce").clip(0.0, 1.0)
    for name in components:
        proportion = pd.to_numeric(
            out[f"proportion_{name}"], errors="coerce"
        ).clip(0.0, 1.0)
        out[f"score_{name}"] = energy * proportion
    score_columns = [f"score_{name}" for name in components]
    out = out.dropna(subset=score_columns)
    return out.reset_index(drop=True)


def subject_roi_scores(
    frame: pd.DataFrame,
    *,
    components: Sequence[str] = DEFAULT_COMPONENTS,
) -> pd.DataFrame:
    """Average contacts within each subject and ROI before group aggregation."""

    components = tuple(components)
    score_columns = [f"score_{name}" for name in components]
    required = {"subject", "channel", "roi", *score_columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prepared table missing columns: {sorted(missing)}")

    # Center relative to each subject's mean sampled-brain expression.  This is
    # descriptive (not an inferential null), but removes a subject-wide tendency
    # to express one template more strongly than another.
    centered = frame.copy()
    subject_means = centered.groupby("subject", observed=True)[score_columns].transform(
        "mean"
    )
    enrichment_columns: list[str] = []
    for name in components:
        score = f"score_{name}"
        enrichment = f"enrichment_{name}"
        centered[enrichment] = centered[score] - subject_means[score]
        enrichment_columns.append(enrichment)

    grouped = centered.groupby(["subject", "roi"], observed=True, sort=True)
    subject_roi = grouped[score_columns + enrichment_columns].mean().reset_index()
    counts = grouped["channel"].nunique().rename("n_contacts_subject_roi").reset_index()
    return subject_roi.merge(counts, on=["subject", "roi"], how="left")


def add_template_specificity(
    frame: pd.DataFrame,
    *,
    components: Sequence[str] = DEFAULT_COMPONENTS,
) -> pd.DataFrame:
    """Add one-vs-best-other template contrasts to an electrode table.

    ``specificity_k = score_k - max(score_j for j != k)``.  Positive values
    identify electrodes where template ``k`` is the best of the three frozen
    motifs, rather than merely having a non-zero NNLS loading.
    """

    components = tuple(components)
    score_columns = [f"score_{name}" for name in components]
    missing = set(score_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"prepared table missing columns: {sorted(missing)}")
    if len(components) < 2:
        raise ValueError("template specificity requires at least two components")

    out = frame.copy()
    scores = out[score_columns].to_numpy(dtype=float)
    for index, name in enumerate(components):
        other = np.delete(scores, index, axis=1)
        out[f"specificity_{name}"] = scores[:, index] - np.max(other, axis=1)
    return out


def summarize_rois(
    subject_roi: pd.DataFrame,
    *,
    components: Sequence[str] = DEFAULT_COMPONENTS,
    min_subjects: int = 10,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a long, subject-balanced ROI summary with bootstrap intervals."""

    components = tuple(components)
    if min_subjects < 1:
        raise ValueError("min_subjects must be at least 1")
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for roi, roi_frame in subject_roi.groupby("roi", observed=True, sort=True):
        n_subjects = int(roi_frame["subject"].nunique())
        if n_subjects < min_subjects:
            continue
        n_contacts = int(roi_frame["n_contacts_subject_roi"].sum())
        for name in components:
            values = roi_frame[f"score_{name}"].to_numpy(dtype=float)
            enrichments = roi_frame[f"enrichment_{name}"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            enrichments = enrichments[np.isfinite(enrichments)]
            if len(values) == 0:
                continue
            if bootstrap_samples > 0:
                draw = rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
                boot_score = values[draw].mean(axis=1)
                draw_e = rng.integers(
                    0,
                    len(enrichments),
                    size=(bootstrap_samples, len(enrichments)),
                )
                boot_enrichment = enrichments[draw_e].mean(axis=1)
                score_low, score_high = np.quantile(boot_score, [0.025, 0.975])
                enrichment_low, enrichment_high = np.quantile(
                    boot_enrichment, [0.025, 0.975]
                )
            else:
                score_low = score_high = np.nan
                enrichment_low = enrichment_high = np.nan
            records.append(
                {
                    "roi": roi,
                    "component": name,
                    "n_subjects": n_subjects,
                    "n_contacts": n_contacts,
                    "mean_score": float(values.mean()),
                    "score_ci_low": float(score_low),
                    "score_ci_high": float(score_high),
                    "mean_enrichment": float(enrichments.mean()),
                    "enrichment_ci_low": float(enrichment_low),
                    "enrichment_ci_high": float(enrichment_high),
                }
            )
    return pd.DataFrame.from_records(records)


def subject_balanced_surface_values(
    frame: pd.DataFrame,
    surface_vertices: np.ndarray,
    value_column: str,
    *,
    bandwidth: float = 8.0,
    max_distance: float = 15.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kernel-smooth a value within subject, then average subjects per vertex.

    Returns the group mean, number of contributing subjects, and fraction of
    those subjects whose local value is positive.  A subject with ten nearby
    contacts has the same group weight as a subject with one nearby contact.
    """

    required = {"subject", "x", "y", "z", value_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prepared table missing columns: {sorted(missing)}")
    if bandwidth <= 0 or max_distance <= 0:
        raise ValueError("bandwidth and max_distance must be positive")

    vertices = np.asarray(surface_vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("surface_vertices must have shape (n_vertices, 3)")
    tree = cKDTree(vertices)
    group_sum = np.zeros(len(vertices), dtype=float)
    coverage = np.zeros(len(vertices), dtype=np.int32)
    positive_count = np.zeros(len(vertices), dtype=np.int32)

    for _, subject_frame in frame.groupby("subject", observed=True, sort=False):
        coords = subject_frame[["x", "y", "z"]].to_numpy(dtype=float)
        values = subject_frame[value_column].to_numpy(dtype=float)
        finite = np.isfinite(coords).all(axis=1) & np.isfinite(values)
        coords = coords[finite]
        values = values[finite]
        numerator = np.zeros(len(vertices), dtype=float)
        denominator = np.zeros(len(vertices), dtype=float)
        for xyz, value in zip(coords, values):
            indices = np.asarray(tree.query_ball_point(xyz, r=max_distance), dtype=int)
            if len(indices) == 0:
                continue
            delta = vertices[indices] - xyz
            distance_sq = np.einsum("ij,ij->i", delta, delta)
            weights = np.exp(-distance_sq / (2.0 * bandwidth**2))
            numerator[indices] += weights * value
            denominator[indices] += weights
        valid = denominator > 0
        subject_values = np.zeros(len(vertices), dtype=float)
        subject_values[valid] = numerator[valid] / denominator[valid]
        group_sum[valid] += subject_values[valid]
        positive_count[valid] += subject_values[valid] > 0
        coverage[valid] += 1

    group_mean = np.full(len(vertices), np.nan, dtype=float)
    positive_fraction = np.full(len(vertices), np.nan, dtype=float)
    valid = coverage > 0
    group_mean[valid] = group_sum[valid] / coverage[valid]
    positive_fraction[valid] = positive_count[valid] / coverage[valid]
    return group_mean, coverage, positive_fraction


def subject_balanced_surface(
    frame: pd.DataFrame,
    surface_vertices: np.ndarray,
    component: str,
    *,
    bandwidth: float = 8.0,
    max_distance: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper for a component's absolute expression score."""

    mean, coverage, _ = subject_balanced_surface_values(
        frame,
        surface_vertices,
        f"score_{component}",
        bandwidth=bandwidth,
        max_distance=max_distance,
    )
    return mean, coverage


def combined_focused_rgba(
    payload: Mapping[str, np.ndarray],
    hemi: str,
    thresholds: Mapping[str, float],
    value_max: float,
    *,
    components: Sequence[str] = DEFAULT_COMPONENTS,
    alpha_scale: float = 1.0,
) -> np.ndarray:
    """Winner-take-all RGBA from the focused display masks of all templates.

    Vertices that survive more than one focused mask keep the higher-specificity
    template.  Alpha scales from the per-template threshold up to ``value_max``.
    ``alpha_scale`` multiplies the final alpha (use <1 to fade the overlay).
    """

    if not components:
        raise ValueError("components must be non-empty")
    reference = np.asarray(payload[f"{hemi}_{components[0]}_specificity"], dtype=float)
    rgba = np.zeros((len(reference), 4), dtype=float)
    best = np.full(len(reference), -np.inf)
    for component in components:
        mask = np.asarray(payload[f"{hemi}_{component}_display_mask"], dtype=bool)
        data = np.asarray(payload[f"{hemi}_{component}_specificity"], dtype=float)
        if len(mask) != len(reference) or len(data) != len(reference):
            raise ValueError(f"{hemi} {component} arrays must match vertex count")
        threshold = float(thresholds[component])
        if not np.isfinite(threshold):
            continue
        denominator = max(value_max - threshold, 1e-12)
        intensity = np.zeros(len(reference), dtype=float)
        valid = mask & np.isfinite(data)
        # Gold washes out on light cortex; keep a higher opacity floor.
        floor = 0.88 if component == "sustain" else 0.70
        intensity[valid] = floor + (1.0 - floor) * np.clip(
            (data[valid] - threshold) / denominator, 0.0, 1.0
        )
        better = valid & (data >= best)
        from src.nmf.waveform_analysis import FUNCTION_COLORS

        rgba[better, :3] = np.asarray(mcolors.to_rgb(FUNCTION_COLORS[component]))
        rgba[better, 3] = intensity[better]
        best[better] = data[better]
    if alpha_scale < 0:
        raise ValueError("alpha_scale must be non-negative")
    rgba[:, 3] *= float(alpha_scale)
    return rgba


def surface_label_boundary_edges(
    faces: np.ndarray,
    vertex_mask: np.ndarray,
) -> np.ndarray:
    """Return ``(n_edges, 2)`` vertex pairs on the boundary of a surface label."""

    faces = np.asarray(faces, dtype=int)
    mask = np.asarray(vertex_mask, dtype=bool)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (n_faces, 3)")
    if mask.ndim != 1:
        raise ValueError("vertex_mask must be one-dimensional")
    inside = mask[faces[:, 0]] & mask[faces[:, 1]] & mask[faces[:, 2]]
    if not inside.any():
        return np.zeros((0, 2), dtype=int)
    interior = faces[inside]
    edges = np.vstack(
        [
            np.sort(interior[:, [0, 1]], axis=1),
            np.sort(interior[:, [1, 2]], axis=1),
            np.sort(interior[:, [2, 0]], axis=1),
        ]
    )
    order = np.lexsort(edges.T[::-1])
    edges = edges[order]
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return unique[counts == 1]


def offset_vertices_along_normals(
    coords: np.ndarray,
    faces: np.ndarray,
    offset_mm: float,
) -> np.ndarray:
    """Move mesh vertices along area-weighted outward normals.

    A small positive offset lifts an overlay off the original cortex so the two
    surfaces do not occupy the same depth buffer samples.
    """

    coords = np.asarray(coords, dtype=float)
    faces = np.asarray(faces, dtype=int)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_vertices, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (n_faces, 3)")
    if not np.isfinite(offset_mm):
        raise ValueError("offset_mm must be finite")
    if offset_mm == 0:
        return coords.copy()
    v0 = coords[faces[:, 0]]
    v1 = coords[faces[:, 1]]
    v2 = coords[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    normals = np.zeros_like(coords)
    np.add.at(normals, faces[:, 0], face_normals)
    np.add.at(normals, faces[:, 1], face_normals)
    np.add.at(normals, faces[:, 2], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = lengths[:, 0] > 1e-12
    normals[valid] /= lengths[valid]
    centered = coords - coords.mean(axis=0)
    inward = np.einsum("ij,ij->i", normals, centered) < 0
    normals[inward] *= -1
    return coords + offset_mm * normals
