from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.decoding.prepare_functional_decoding_dataset import (
    PreparedFeature,
    load_assignments,
    pair_cross_conditions,
    select_assigned_channels,
    significance_union,
    symmetric_condition_qc,
)


def assignment_rows():
    return pd.DataFrame(
        {
            "subject": ["D1", "D2", "D3", "D4", "D5", "D6"],
            "channel": [
                "D1_L1-2",
                "D2_R1-2",
                "D3_L1-2",
                "D4_R1-2",
                "D5_L1-2",
                "D6_R1-2",
            ],
            "hemi": ["L", "R", "L", "R", "L", "R"],
            "functional_cluster": [
                "sensory",
                "sensory",
                "sustain",
                "sustain",
                "motor",
                "motor",
            ],
            "dominance": [0.51, 0.99, 0.50, 0.75, 0.60, 0.80],
        }
    )


def test_load_assignments_maps_all_groups_without_dominance_filter(tmp_path: Path):
    path = tmp_path / "assignments.csv"
    assignment_rows().to_csv(path, index=False)
    loaded = load_assignments(path)
    assert len(loaded) == 6
    assert set(loaded["pseudo_subject"]) == {
        "Sensory",
        "Sustain",
        "Motor",
    }
    assert loaded.loc[loaded["dominance"].eq(0.50), "pseudo_subject"].item() == "Sustain"
    assert (
        loaded.loc[loaded["functional_cluster"].eq("motor") & loaded["hemi"].eq("L"), "pseudo_subject"].item()
        == "Motor"
    )


def test_assignment_validation_rejects_duplicate_key(tmp_path: Path):
    frame = pd.concat([assignment_rows(), assignment_rows().iloc[[0]]], ignore_index=True)
    path = tmp_path / "assignments.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate subject.channel"):
        load_assignments(path)


def test_phase_significance_policy_and_exact_membership():
    assignments = assignment_rows().assign(
        pseudo_subject=[
            "Sensory",
            "Sensory",
            "Sustain",
            "Sustain",
            "Motor",
            "Motor",
        ]
    )
    sig = significance_union(
        "LexicalDelay",
        {"Decision": {"D1_L1-2"}, "Repeat": {"other", "D1_L1-2"}},
    )
    assert sig == {"D1_L1-2", "other"}
    assert select_assigned_channels(
        assignments,
        "Sensory",
        "sub-D1",
        sig,
        {"D1_L1-2", "unavailable"},
    ) == ["D1_L1-2"]
    assert significance_union(
        "PhonemeSequence",
        {"Decision": {"wrong"}, "Repeat": {"D1_L1-2"}},
    ) == {"D1_L1-2"}


def test_symmetric_qc_uses_identical_order_and_rejects_one_sided_channel():
    decision = xr.DataArray(
        np.ones((4, 2, 3)),
        dims=("trial", "channel", "time"),
        coords={"trial": ["a", "b", "c", "d"], "channel": ["z", "shared"], "time": range(3)},
    )
    repeat_values = np.ones((4, 2, 3))
    repeat_values[:3, 0, :] = np.nan
    repeat = xr.DataArray(
        repeat_values,
        dims=("trial", "channel", "time"),
        coords={"trial": ["a", "b", "c", "d"], "channel": ["z", "shared"], "time": range(3)},
    )
    filtered, candidates, retained, _, removed = symmetric_condition_qc(
        {"Decision": decision, "Repeat": repeat}
    )
    assert candidates == ["shared", "z"]
    assert retained == ["shared"]
    assert list(filtered["Decision"].channel.values) == ["shared"]
    assert list(filtered["Repeat"].channel.values) == ["shared"]
    assert "z" in removed


def make_prepared(trials, labels, offset=0):
    X = xr.DataArray(
        np.arange(len(trials) * 2 * 3, dtype=float).reshape(len(trials), 2, 3) + offset,
        dims=("trial", "channel", "time"),
        coords={"trial": trials, "channel": ["c1", "c2"], "time": [0.0, 0.1, 0.2]},
    )
    event_id = {"Word": 0, "Nonword": 1}
    return PreparedFeature(
        X=X,
        y=np.asarray([event_id[label] for label in labels]),
        trials=list(trials),
        conditions=[trial.rsplit("_", 1)[0] for trial in trials],
        labels=list(labels),
        event_id=event_id,
    )


def test_cross_pairing_uses_trial_ids_not_length_truncation():
    repeat = make_prepared(
        ["cat_1", "dog_1", "pim_1"], ["Word", "Word", "Nonword"]
    )
    decision = make_prepared(
        ["pim_1", "cat_1", "extra_1"], ["Nonword", "Word", "Nonword"], offset=100
    )
    paired = pair_cross_conditions(repeat, decision)
    assert paired["Repeat"].trials == ["cat_1", "pim_1"]
    assert paired["Decision"].trials == ["cat_1", "pim_1"]
    assert paired["Repeat"].labels == paired["Decision"].labels == ["Word", "Nonword"]


def test_cross_pairing_rejects_label_mismatch():
    repeat = make_prepared(["cat_1", "pim_1"], ["Word", "Nonword"])
    decision = make_prepared(["cat_1", "pim_1"], ["Nonword", "Nonword"])
    with pytest.raises(ValueError, match="labels differ"):
        pair_cross_conditions(repeat, decision)
