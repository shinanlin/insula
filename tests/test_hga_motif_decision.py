import numpy as np
import pandas as pd
import pytest

from src.connectivity.hga_motif_decision import (
    MetricColumns,
    add_distance_bins,
    annotate_motif_weights,
    balance_tasks_within_subject,
    classify_lag_clusters,
    stratified_partner_permutation,
    stratified_partner_permutation_multi,
    weighted_cell_summary,
)


def test_lag_classification_strict_unresolved_and_ambiguous():
    pairs = pd.DataFrame(
        {
            "pair_id": ["negative", "positive", "zero", "both", "nonsig"],
            "sig_fdr": [True, True, True, True, False],
        }
    )
    clusters = pd.DataFrame(
        [
            ("negative", -0.10, -0.03, -0.06, 4.0, 0.01),
            ("positive", 0.02, 0.09, 0.05, 3.0, 0.02),
            ("zero", -0.02, 0.03, 0.00, 5.0, 0.01),
            ("both", -0.08, -0.03, -0.05, 5.0, 0.01),
            ("both", 0.04, 0.08, 0.06, 5.0, 0.01),
            ("nonsig", -0.08, -0.03, -0.05, 8.0, 0.01),
        ],
        columns=[
            "pair_id", "lag_start_s", "lag_stop_s", "peak_lag_s",
            "cluster_mass", "p_pair_cluster",
        ],
    )
    result = classify_lag_clusters(pairs, clusters).set_index("pair_id")
    assert result.loc["negative", "lag_direction"] == "insula_leads"
    assert result.loc["positive", "lag_direction"] == "partner_leads"
    assert result.loc["zero", "lag_direction"] == "unresolved"
    assert result.loc["both", "lag_direction"] == "ambiguous"
    assert result.loc["nonsig", "lag_direction"] == "not_fdr"


def _motif_inputs():
    pairs = pd.DataFrame(
        {
            "source": ["S1", "S1"],
            "target": ["T1", "T2"],
            "subject": ["P1", "P1"],
            "task": ["A", "A"],
            "phase": ["Stimulus", "Stimulus"],
            "sig": [1.0, 0.0],
            "fwer": [0.0, 0.0],
            "effect": [2.0, 0.0],
        }
    )
    seeds = pd.DataFrame(
        {
            "channel": ["S1"], "x": [0.0], "y": [0.0], "z": [0.0],
            "loading_sensory": [3.0], "loading_sustain": [1.0],
            "loading_motor": [0.0],
        }
    )
    partners = pd.DataFrame(
        {
            "channel": ["T1", "T2"], "roi": ["A", "A"],
            "hemi": ["L", "L"], "x": [3.0, 6.0], "y": [4.0, 8.0],
            "z": [0.0, 0.0], "explained_energy": [0.8, 0.8],
            "proportion_sensory": [1.0, 0.0],
            "proportion_sustain": [0.0, 0.0],
            "proportion_motor": [0.0, 1.0],
        }
    )
    return pairs, seeds, partners


def test_continuous_motif_weighting_and_projection_qc():
    pairs, seeds, partners = _motif_inputs()
    annotated = annotate_motif_weights(pairs, seeds, partners)
    assert annotated.loc[0, "source_proportion_sensory"] == pytest.approx(0.75)
    assert annotated.loc[0, "motif_match"] == pytest.approx(0.75)
    assert annotated.loc[1, "motif_match"] == pytest.approx(0.0)
    assert annotated.loc[0, "distance_mm"] == pytest.approx(5.0)
    cells = weighted_cell_summary(
        annotated,
        [MetricColumns("m", "sig", "fwer", "effect")],
        group_columns=("subject", "task", "phase"),
    )
    sensory_sensory = cells.query(
        "source_motif == 'sensory' and partner_motif == 'sensory'"
    ).iloc[0]
    sensory_motor = cells.query(
        "source_motif == 'sensory' and partner_motif == 'motor'"
    ).iloc[0]
    assert sensory_sensory["m_fdr_hit_rate"] == pytest.approx(1.0)
    assert sensory_motor["m_fdr_hit_rate"] == pytest.approx(0.0)

    low_quality = partners.copy()
    low_quality.loc[0, "explained_energy"] = 0.4
    sensitivity_ready = annotate_motif_weights(pairs, seeds, low_quality)
    assert not bool(sensitivity_ready.loc[0, "partner_projection_qc"])
    assert sensitivity_ready.loc[0, "motif_match"] == pytest.approx(0.75)


def test_task_balancing_gives_each_task_equal_weight():
    rows = []
    for task, values in (("large", [1.0] * 20), ("small", [0.0])):
        for value in values:
            rows.append(
                {
                    "subject": "P1", "task": task, "phase": "Stimulus",
                    "source_motif": "sensory", "partner_motif": "sensory",
                    "eligible_weight": 1.0, "n_pairs": 1, "value": value,
                }
            )
    entity_like = (
        pd.DataFrame(rows)
        .groupby(["subject", "task", "phase", "source_motif", "partner_motif"])
        .agg(value=("value", "mean"), eligible_weight=("eligible_weight", "sum"),
             n_pairs=("n_pairs", "sum"))
        .reset_index()
    )
    balanced = balance_tasks_within_subject(entity_like, value_columns=["value"])
    assert balanced.loc[0, "value"] == pytest.approx(0.5)
    assert balanced.loc[0, "n_tasks"] == 2


def test_stratified_partner_permutation_is_deterministic():
    pairs, seeds, partners = _motif_inputs()
    pairs = pd.concat([pairs] * 4, ignore_index=True)
    pairs["target"] = ["T1", "T2"] * 4
    pairs["subject"] = "P1"
    annotated = annotate_motif_weights(pairs, seeds, partners)
    annotated["target_roi"] = "A"
    annotated["distance_bin"] = 0
    first = stratified_partner_permutation(
        annotated, outcome_column="sig", n_permutations=100, random_state=11
    )
    second = stratified_partner_permutation(
        annotated, outcome_column="sig", n_permutations=100, random_state=11
    )
    assert first == second
    assert first["n_pairs"] == 8
    multi = stratified_partner_permutation_multi(
        annotated, outcome_columns=["sig", "effect"],
        n_permutations=100, random_state=11,
    ).set_index("outcome")
    assert multi.loc["sig", "stat"] == pytest.approx(first["stat"])
    assert multi.loc["sig", "p"] == pytest.approx(first["p"])


def test_repeated_anatomical_distance_stays_in_same_quintile():
    frame = pd.DataFrame(
        {
            "subject": ["P1"] * 8,
            "target_roi": ["A"] * 8,
            "distance_mm": [10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0],
        }
    )
    result = add_distance_bins(frame, n_bins=5)
    assert result.groupby("distance_mm")["distance_bin"].nunique().eq(1).all()
