"""Tests for OAEC electrode selection on focused motif territories."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.nmf.oaec_focused_overlay import (
    annotate_motif_pairs,
    assign_focused_membership,
    assign_partner_insula_cluster,
    assign_territory_plot_colors,
    collapse_phase_partners,
    electrodes_for_any_insula_phase,
    electrodes_for_cluster_phase,
    focused_mask_payload,
    group_motif_coupling,
    mark_partner_significance,
    partners_by_motif_cell,
    restrict_overlay_to_focused_partners,
    select_overlay_electrodes,
    subject_motif_coupling,
    subject_within_cross,
    union_cluster_phase_partners,
)


def test_select_overlay_electrodes_keeps_sources_as_seeds():
    pairs = pd.DataFrame(
        {
            "source_channel": ["ins_a", "ins_a", "ins_b"],
            "target_channel": ["stg_1", "ins_b", "stg_1"],
            "pair_key": ["ins_a||stg_1", "ins_a||ins_b", "ins_b||stg_1"],
            "subject": ["s1", "s1", "s1"],
        }
    )
    meta = pd.DataFrame(
        {
            "channel": ["ins_a", "ins_b", "stg_1"],
            "roi": ["AIC", "PIC", "STG"],
            "hemi": ["L", "L", "L"],
            "x": [-30.0, -32.0, -60.0],
            "y": [0.0, 2.0, -10.0],
            "z": [-20.0, -18.0, -5.0],
            "functional_cluster": ["sensory", "motor", pd.NA],
        }
    )
    projection = pd.DataFrame(
        {
            "channel": ["stg_1", "ins_a"],
            "best_component": ["sensory", "sensory"],
            "explained_energy": [0.8, 0.9],
            "in_discovery": [False, True],
        }
    )
    table = select_overlay_electrodes(pairs, meta, projection)
    roles = table.set_index("channel")["role"].to_dict()
    assert roles == {"ins_a": "seed", "ins_b": "seed", "stg_1": "partner"}
    assert table.set_index("channel").loc["ins_a", "seed_cluster"] == "sensory"
    assert pd.isna(table.set_index("channel").loc["stg_1", "seed_cluster"])
    assert table.set_index("channel").loc["stg_1", "best_component"] == "sensory"
    assert table.set_index("channel").loc["stg_1", "n_pairs"] == 2


def test_assign_focused_membership_uses_nearest_masked_vertex():
    electrodes = pd.DataFrame(
        {
            "channel": ["near", "unmasked", "outside"],
            "hemi": ["L", "L", "L"],
            "x": [0.0, 1.0, 40.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
        }
    )
    vertices = {"lh": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)}
    mask = {"lh": np.array([True, False])}
    marked = assign_focused_membership(
        electrodes,
        vertices_by_hemi=vertices,
        mask_by_hemi=mask,
        max_distance=15.0,
    )
    by_channel = marked.set_index("channel")
    assert bool(by_channel.loc["near", "in_focused"])
    assert not bool(by_channel.loc["unmasked", "in_focused"])
    assert not bool(by_channel.loc["outside", "in_focused"])
    assert by_channel.loc["near", "surf_hemi"] == "lh"
    far = assign_focused_membership(
        electrodes,
        vertices_by_hemi=vertices,
        mask_by_hemi=mask,
        max_distance=5.0,
    )
    assert not bool(far.set_index("channel").loc["outside", "in_focused"])


def test_focused_mask_payload_unions_display_masks():
    payload = {
        "lh_sustain_display_mask": np.array([True, False, False]),
        "lh_motor_display_mask": np.array([False, True, False]),
        "lh_sensory_display_mask": np.array([False, False, False]),
        "lh_sustain_specificity": np.array([0.4, 0.0, 0.0]),
        "lh_motor_specificity": np.array([0.0, 0.7, 0.0]),
        "lh_sensory_specificity": np.array([0.0, 0.0, 0.0]),
    }
    mask_by_hemi, spec = focused_mask_payload(payload)
    assert mask_by_hemi["lh"].tolist() == [True, True, False]
    assert "motor" in spec["lh"]


def test_restrict_overlay_drops_extra_cluster_partners_and_orphan_seeds():
    pairs = pd.DataFrame(
        {
            "source_channel": ["ins_keep", "ins_keep", "ins_drop"],
            "target_channel": ["stg_in", "stg_out", "stg_out"],
            "pair_key": ["ins_keep||stg_in", "ins_keep||stg_out", "ins_drop||stg_out"],
            "subject": ["s1", "s1", "s2"],
        }
    )
    electrodes = pd.DataFrame(
        {
            "channel": ["ins_keep", "ins_drop", "stg_in", "stg_out"],
            "role": ["seed", "seed", "partner", "partner"],
            "in_focused": [False, False, True, False],
            "n_pairs": [2, 1, 1, 2],
        }
    )
    keep_pairs, keep_electrodes = restrict_overlay_to_focused_partners(pairs, electrodes)
    assert set(keep_pairs["pair_key"]) == {"ins_keep||stg_in"}
    assert set(keep_electrodes["channel"]) == {"ins_keep", "stg_in"}
    by_channel = keep_electrodes.set_index("channel")
    assert int(by_channel.loc["ins_keep", "n_pairs"]) == 1
    assert int(by_channel.loc["stg_in", "n_pairs"]) == 1


def test_mark_partner_significance_flags_never_sig_partners():
    pairs = pd.DataFrame(
        {
            "target_channel": ["sig_p", "nonsig_p"],
            "sig_within_pair": [True, False],
        }
    )
    electrodes = pd.DataFrame(
        {
            "channel": ["seed_a", "sig_p", "nonsig_p"],
            "role": ["seed", "partner", "partner"],
        }
    )
    marked = mark_partner_significance(pairs, electrodes)
    by_channel = marked.set_index("channel")
    assert bool(by_channel.loc["seed_a", "has_sig"])
    assert bool(by_channel.loc["sig_p", "has_sig"])
    assert not bool(by_channel.loc["nonsig_p", "has_sig"])


def test_partner_insula_cluster_prefers_significant_seed():
    pairs = pd.DataFrame(
        {
            "source_channel": ["ins_s", "ins_m", "ins_s"],
            "target_channel": ["p1", "p1", "p2"],
            "pair_key": ["a", "b", "c"],
            "source_cluster": ["sustain", "motor", "sustain"],
            "sig_within_pair": [False, True, False],
        }
    )
    electrodes = pd.DataFrame(
        {
            "channel": ["ins_s", "ins_m", "p1", "p2"],
            "role": ["seed", "seed", "partner", "partner"],
            "seed_cluster": ["sustain", "motor", pd.NA, pd.NA],
        }
    )
    marked = assign_partner_insula_cluster(pairs, electrodes)
    by_channel = marked.set_index("channel")
    assert by_channel.loc["p1", "insula_cluster"] == "motor"
    assert by_channel.loc["p2", "insula_cluster"] == "sustain"


def test_motif_coupling_is_subject_then_group_averaged():
    pairs = pd.DataFrame(
        {
            "subject": ["s1", "s1", "s1", "s2", "s2"],
            "source_channel": ["ins_s", "ins_s", "ins_m", "ins_s", "ins_s"],
            "target_channel": ["p_s", "p_m", "p_s", "p_s", "p_s"],
            "source_cluster": ["sustain", "sustain", "motor", "sustain", "sustain"],
            "pair_key": ["a", "b", "c", "d", "e"],
            "sig_within_pair": [True, False, True, True, False],
            "stat": [0.4, 0.1, 0.3, 0.8, 0.2],
        }
    )
    electrodes = pd.DataFrame(
        {
            "channel": ["ins_s", "ins_m", "p_s", "p_m"],
            "role": ["seed", "seed", "partner", "partner"],
            "in_focused": [False, False, True, True],
            "seed_cluster": ["sustain", "motor", pd.NA, pd.NA],
            "best_component": [pd.NA, pd.NA, "sensory", "motor"],
        }
    )
    labeled = annotate_motif_pairs(pairs, electrodes)
    assert set(labeled["partner_component"]) <= {"sensory", "motor"}
    cells = subject_motif_coupling(labeled, min_pairs=1)
    sustain_sensory = cells.loc[
        cells["source_cluster"].eq("sustain") & cells["partner_component"].eq("sensory")
    ]
    assert list(sustain_sensory["subject"]) == ["s1", "s2"]
    assert sustain_sensory.set_index("subject").loc["s1", "pair_hit"] == 1.0
    assert sustain_sensory.set_index("subject").loc["s2", "pair_hit"] == 0.5
    grouped = group_motif_coupling(cells)
    cell = grouped.loc[
        grouped["source_cluster"].eq("sustain") & grouped["partner_component"].eq("sensory")
    ].iloc[0]
    assert cell["n_subjects"] == 2
    assert abs(float(cell["mean_pair_hit"]) - 0.75) < 1e-9
    contrast = subject_within_cross(labeled)
    s1 = contrast.set_index("subject").loc["s1"]
    assert int(s1["within_n_pairs"]) == 0
    assert int(s1["cross_n_pairs"]) == 3
    members = partners_by_motif_cell(labeled)
    sustain_sensory_partners = members.loc[
        members["source_cluster"].eq("sustain")
        & members["partner_component"].eq("sensory")
    ]
    assert set(sustain_sensory_partners["target_channel"]) == {"p_s"}
    assert "p_m" not in set(members["target_channel"])  # only non-sig sustain→motor


def test_cluster_phase_keeps_partners_that_also_couple_elsewhere():
    pairs = pd.DataFrame(
        {
            "source_channel": ["ins_m", "ins_s", "ins_m"],
            "target_channel": ["stg", "stg", "smc"],
            "source_cluster": ["motor", "sensory", "motor"],
            "phase_norm": ["stimulus", "stimulus", "delay"],
            "sig_within_pair": [True, True, True],
        }
    )
    electrodes = pd.DataFrame(
        {
            "channel": ["ins_m", "ins_s", "stg", "smc"],
            "role": ["seed", "seed", "partner", "partner"],
            "seed_cluster": ["motor", "sensory", pd.NA, pd.NA],
            "in_focused": [False, False, True, True],
            "focused_component": [pd.NA, pd.NA, "sensory", "motor"],
        }
    )
    stimulus = electrodes_for_cluster_phase(electrodes, pairs, "motor", "stimulus")
    delay = electrodes_for_cluster_phase(electrodes, pairs, "motor", "delay")
    stimulus_ids = set(stimulus["channel"])
    delay_ids = set(delay["channel"])
    assert stimulus_ids == {"ins_m", "stg"}
    assert delay_ids == {"ins_m", "smc"}
    assert "ins_s" not in stimulus_ids
    stg_color = stimulus.set_index("channel").loc["stg", "plot_color"]
    seed_color = stimulus.set_index("channel").loc["ins_m", "plot_color"]
    from src.nmf.waveform_analysis import FUNCTION_COLORS

    assert stg_color == FUNCTION_COLORS["sensory"]
    assert seed_color == FUNCTION_COLORS["motor"]


def test_territory_plot_colors_follow_focused_component():
    electrodes = pd.DataFrame(
        {
            "channel": ["ins_m", "stg"],
            "role": ["seed", "partner"],
            "seed_cluster": ["motor", pd.NA],
            "focused_component": [pd.NA, "sensory"],
            "best_component": [pd.NA, "motor"],
        }
    )
    marked = assign_territory_plot_colors(electrodes)
    from src.nmf.waveform_analysis import FUNCTION_COLORS

    by_channel = marked.set_index("channel")
    assert by_channel.loc["ins_m", "plot_color"] == FUNCTION_COLORS["motor"]
    assert by_channel.loc["stg", "plot_color"] == FUNCTION_COLORS["sensory"]


def test_any_insula_phase_unions_clusters_and_drops_seeds():
    pairs = pd.DataFrame(
        {
            "source_channel": ["ins_m", "ins_s", "ins_m"],
            "target_channel": ["stg", "stg", "smc"],
            "source_cluster": ["motor", "sensory", "motor"],
            "phase_norm": ["stimulus", "stimulus", "delay"],
            "sig_within_pair": [True, True, True],
        }
    )
    electrodes = pd.DataFrame(
        {
            "channel": ["ins_m", "ins_s", "stg", "smc"],
            "role": ["seed", "seed", "partner", "partner"],
            "seed_cluster": ["motor", "sensory", pd.NA, pd.NA],
            "in_focused": [False, False, True, True],
            "focused_component": [pd.NA, pd.NA, "sensory", "motor"],
        }
    )
    stimulus = electrodes_for_any_insula_phase(electrodes, pairs, "stimulus")
    delay = electrodes_for_any_insula_phase(electrodes, pairs, "delay")
    assert set(stimulus["channel"]) == {"stg"}
    assert set(delay["channel"]) == {"smc"}
    from src.nmf.waveform_analysis import FUNCTION_COLORS

    assert stimulus.set_index("channel").loc["stg", "plot_color"] == FUNCTION_COLORS["sensory"]


def test_union_cluster_phase_partners_dedupes_across_clusters():
    motor = pd.DataFrame(
        {
            "channel": ["ins_m", "stg"],
            "role": ["seed", "partner"],
            "phase": ["stimulus", "stimulus"],
            "focused_component": [pd.NA, "sensory"],
        }
    )
    sensory = pd.DataFrame(
        {
            "channel": ["ins_s", "stg", "pTL"],
            "role": ["seed", "partner", "partner"],
            "phase": ["stimulus", "stimulus", "stimulus"],
            "focused_component": [pd.NA, "sensory", "sensory"],
        }
    )
    merged = union_cluster_phase_partners([motor, sensory])
    assert set(merged["channel"]) == {"stg", "pTL"}
    assert "ins_m" not in set(merged["channel"])
    assert len(merged) == 2


def test_collapse_phase_partners_unions_early_and_late():
    table = pd.DataFrame(
        {
            "channel": ["stg", "stg", "smc", "mfg"],
            "role": ["partner", "partner", "partner", "partner"],
            "phase": ["stimulus", "delay", "go", "response"],
            "focused_component": ["sensory", "sensory", "motor", "sustain"],
        }
    )
    collapsed = collapse_phase_partners(table)
    by_phase = collapsed.groupby("phase")["channel"].apply(set).to_dict()
    assert by_phase["stimulus_delay"] == {"stg"}
    assert by_phase["go_response"] == {"smc", "mfg"}
