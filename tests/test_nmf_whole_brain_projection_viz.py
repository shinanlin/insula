"""Tests for subject-balanced whole-brain motif visualization summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.nmf.whole_brain_projection_viz import (
    add_template_specificity,
    combined_focused_rgba,
    offset_vertices_along_normals,
    prepare_projection_frame,
    refocus_display_masks,
    retain_largest_surface_clusters,
    subject_balanced_surface,
    subject_balanced_surface_values,
    subject_roi_scores,
    summarize_rois,
    surface_label_boundary_edges,
)


def _row(subject: str, channel: str, roi: str, sustain: float, **extra):
    return {
        "subject": subject,
        "channel": channel,
        "roi": roi,
        "in_discovery": False,
        "mix": False,
        "explained_energy": 0.5,
        "proportion_sustain": sustain,
        "proportion_motor": 1 - sustain,
        "proportion_sensory": 0.0,
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        **extra,
    }


def test_prepare_projection_frame_filters_and_energy_weights():
    frame = pd.DataFrame(
        [
            _row("s1", "a", "STG", 0.8),
            _row("s1", "b", "STG", 0.8, in_discovery=True),
            _row("s1", "c", "Unknown", 0.8),
            _row("s1", "d", "MTG", 0.8, mix="True"),
        ]
    )
    prepared = prepare_projection_frame(frame)
    assert prepared["channel"].tolist() == ["a"]
    assert np.isclose(prepared.loc[0, "score_sustain"], 0.4)
    assert np.isclose(prepared.loc[0, "score_motor"], 0.1)


def test_roi_summary_gives_each_subject_equal_weight():
    rows = [_row("s1", f"a{i}", "STG", 1.0) for i in range(10)]
    rows += [_row("s2", "b", "STG", 0.0)]
    prepared = prepare_projection_frame(pd.DataFrame(rows))
    subject_roi = subject_roi_scores(prepared)
    summary = summarize_rois(
        subject_roi, min_subjects=1, bootstrap_samples=0
    )
    sustain = summary.query("roi == 'STG' and component == 'sustain'").iloc[0]
    # Subject-balanced mean: (0.5 + 0.0) / 2, not contact-pooled 5 / 11.
    assert np.isclose(sustain["mean_score"], 0.25)
    assert sustain["n_subjects"] == 2
    assert sustain["n_contacts"] == 11


def test_surface_map_balances_subjects_before_group_average():
    vertices = np.array([[0.0, 0.0, 0.0]])
    rows = [_row("s1", f"a{i}", "STG", 1.0) for i in range(5)]
    rows += [_row("s2", "b", "STG", 0.0)]
    prepared = prepare_projection_frame(pd.DataFrame(rows))
    data, coverage = subject_balanced_surface(
        prepared,
        vertices,
        "sustain",
        bandwidth=1.0,
        max_distance=1.0,
    )
    assert coverage.tolist() == [2]
    assert np.isclose(data[0], 0.25)


def test_template_specificity_is_one_vs_best_competitor():
    prepared = prepare_projection_frame(
        pd.DataFrame([_row("s1", "a", "STG", 0.8)])
    )
    specific = add_template_specificity(prepared)
    assert np.isclose(specific.loc[0, "specificity_sustain"], 0.3)
    assert np.isclose(specific.loc[0, "specificity_motor"], -0.3)
    assert np.isclose(specific.loc[0, "specificity_sensory"], -0.4)


def test_surface_value_summary_reports_positive_subject_fraction():
    vertices = np.array([[0.0, 0.0, 0.0]])
    frame = pd.DataFrame(
        [
            {**_row("s1", "a", "STG", 1.0), "value": 1.0},
            {**_row("s2", "b", "STG", 1.0), "value": -1.0},
            {**_row("s3", "c", "STG", 1.0), "value": 2.0},
        ]
    )
    mean, coverage, positive_fraction = subject_balanced_surface_values(
        frame,
        vertices,
        "value",
        bandwidth=1.0,
        max_distance=1.0,
    )
    assert coverage.tolist() == [3]
    assert np.isclose(mean[0], 2 / 3)
    assert np.isclose(positive_fraction[0], 2 / 3)


def test_surface_cluster_filter_removes_speckles_and_keeps_largest():
    # Two disconnected triangles plus one isolated selected vertex.
    faces = np.array([[0, 1, 2], [3, 4, 5], [5, 6, 3]])
    mask = np.array([True, True, True, True, True, True, True, True])
    keep = retain_largest_surface_clusters(
        mask, faces, max_clusters=1, min_vertices=2
    )
    assert keep.tolist() == [False, False, False, True, True, True, True, False]


def test_refocus_display_masks_keeps_upper_tail():
    faces = {"lh": np.array([[0, 1, 2]]), "rh": np.array([[0, 1, 2]])}
    payload = {}
    for hemi in ("lh", "rh"):
        payload[f"{hemi}_sustain_specificity"] = np.array([0.1, 0.9, 0.4])
        payload[f"{hemi}_sustain_coverage"] = np.array([3, 3, 3])
        payload[f"{hemi}_sustain_positive_fraction"] = np.array([1.0, 1.0, 1.0])
        payload[f"{hemi}_motor_specificity"] = np.array([0.2, 0.3, 0.8])
        payload[f"{hemi}_motor_coverage"] = np.array([3, 3, 3])
        payload[f"{hemi}_motor_positive_fraction"] = np.array([1.0, 1.0, 1.0])
        payload[f"{hemi}_sensory_specificity"] = np.array([0.05, 0.6, 0.7])
        payload[f"{hemi}_sensory_coverage"] = np.array([3, 3, 3])
        payload[f"{hemi}_sensory_positive_fraction"] = np.array([1.0, 1.0, 1.0])
    focused, thresholds = refocus_display_masks(
        payload,
        faces,
        keep_top=0.50,
        min_subjects=3,
        min_positive_fraction=0.50,
        max_clusters=5,
        min_cluster_vertices=1,
    )
    assert thresholds["sustain"] == 0.4
    assert focused["lh_sustain_display_mask"].tolist() == [False, True, True]


def test_combined_focused_rgba_keeps_higher_specificity_template():
    payload = {
        "lh_sustain_specificity": np.array([0.80, 0.10, 0.50, 0.00]),
        "lh_sustain_display_mask": np.array([True, False, True, False]),
        "lh_motor_specificity": np.array([0.20, 0.90, 0.60, 0.00]),
        "lh_motor_display_mask": np.array([False, True, True, False]),
        "lh_sensory_specificity": np.array([0.00, 0.00, 0.00, 0.70]),
        "lh_sensory_display_mask": np.array([False, False, False, True]),
    }
    rgba = combined_focused_rgba(
        payload,
        "lh",
        {"sustain": 0.0, "motor": 0.0, "sensory": 0.0},
        1.0,
    )
    assert np.allclose(rgba[0, :3], [196 / 255, 163 / 255, 90 / 255], atol=1e-6)
    assert np.allclose(rgba[1, :3], [169 / 255, 55 / 255, 59 / 255], atol=1e-6)
    assert np.allclose(rgba[2, :3], [169 / 255, 55 / 255, 59 / 255], atol=1e-6)
    assert np.allclose(rgba[3, :3], [35 / 255, 105 / 255, 189 / 255], atol=1e-6)
    assert np.allclose(rgba[:, 3], [0.976, 0.970, 0.880, 0.910], atol=1e-3)
    faded = combined_focused_rgba(
        payload,
        "lh",
        {"sustain": 0.0, "motor": 0.0, "sensory": 0.0},
        1.0,
        alpha_scale=0.5,
    )
    assert np.allclose(faded[:, 3], 0.5 * rgba[:, 3], atol=1e-6)


def test_surface_label_boundary_keeps_outer_edges_only():
    faces = np.array([[0, 1, 2], [1, 2, 3]])
    mask = np.array([True, True, True, True])
    edges = {tuple(edge) for edge in surface_label_boundary_edges(faces, mask)}
    assert (1, 2) not in edges
    assert edges == {(0, 1), (0, 2), (1, 3), (2, 3)}


def test_offset_vertices_along_normals_lifts_a_triangle():
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    faces = np.array([[0, 1, 2]])
    moved = offset_vertices_along_normals(coords, faces, 0.5)
    assert np.allclose(moved[:, :2], coords[:, :2])
    assert np.allclose(moved[:, 2], 0.5)
