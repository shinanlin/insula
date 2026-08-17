import numpy as np

from src.nmf.waveform_analysis import (
    concatenated_phase_matrix,
    held_out_contrasts,
    names_aligned_to_reference,
    normalize_nmf_factors,
    orient_components,
    orient_components_on_phase_segment,
    orient_three_components,
    orient_two_components,
    split_concat_components,
    within_subject_permutation_p,
)
import pandas as pd


def test_normalize_nmf_factors_preserves_reconstruction_and_unit_h():
    W = np.array([[1.0, 2.0], [3.0, 4.0]])
    H = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]])

    W_normalized, H_normalized = normalize_nmf_factors(W, H)

    np.testing.assert_allclose(W @ H, W_normalized @ H_normalized)
    np.testing.assert_allclose(np.linalg.norm(H_normalized, axis=1), 1.0)


def test_orient_two_components_uses_waveform_not_anatomy():
    times = np.linspace(-0.5, 1.0, 151)
    early = np.exp(-np.square((times - 0.15) / 0.12))
    late = np.exp(-np.square((times - 0.65) / 0.20))

    mapping = orient_two_components(np.vstack([late, early]), times)

    assert mapping == {0: "sustain", 1: "sensory"}


def test_orient_three_components_orders_by_transient_score():
    times = np.linspace(-0.5, 1.0, 151)
    early = np.exp(-np.square((times - 0.15) / 0.12))
    mid = 0.55 * early + 0.45 * np.exp(-np.square((times - 0.55) / 0.22))
    late = np.exp(-np.square((times - 0.65) / 0.20))

    mapping = orient_three_components(np.vstack([mid, late, early]), times)

    assert mapping == {
        1: "sustain",
        0: "motor",
        2: "sensory",
    }
    assert orient_components(np.vstack([late, early]), times) == {
        0: "sustain",
        1: "sensory",
    }


def test_names_aligned_to_reference_recovers_permuted_components():
    times = np.linspace(-0.2, 1.0, 80)
    H_ref = np.vstack(
        [
            np.exp(-np.square((times - 0.65) / 0.22)),
            np.exp(-np.square((times - 0.15) / 0.10)),
            0.5 * np.exp(-np.square((times - 0.40) / 0.18)),
        ]
    )
    ref_names = {
        0: "sustain",
        1: "sensory",
        2: "motor",
    }
    perm = [2, 0, 1]
    H_phase = H_ref[perm]
    names, mapping, corr = names_aligned_to_reference(
        H_phase, times, H_ref, times, ref_names
    )

    assert mapping == {0: 2, 1: 0, 2: 1}
    assert names == {
        0: "motor",
        1: "sustain",
        2: "sensory",
    }
    assert corr[0, 2] > 0.95


def test_within_subject_permutation_returns_bounded_reproducible_p_value():
    labels = np.array(
        ["sustain", "sensory"] * 4,
        dtype=object,
    )
    rois = np.array(["AIC", "PIC"] * 4)
    subjects = np.repeat(["S1", "S2"], 4)

    observed, p_value = within_subject_permutation_p(
        labels, rois, subjects, n_permutations=100, random_state=7
    )

    assert observed == 1.0
    assert 0.0 < p_value <= 1.0


def test_concatenated_phase_matrix_keeps_intersection_and_slices():
    rows = []
    for channel, phases in (
        ("ch_all", ("stimulus", "delay", "go", "response")),
        ("ch_missing_delay", ("stimulus", "go", "response")),
    ):
        for phase in phases:
            t0, t1 = {"stimulus": (-0.2, 0.2), "delay": (0.1, 0.3), "go": (-0.1, 0.2), "response": (-0.1, 0.1)}[phase]
            for time in np.linspace(t0 + 0.01, t1 - 0.01, 5):
                rows.append(
                    {
                        "channel": channel,
                        "phase": phase,
                        "time": float(time),
                        "value": 1.0,
                        "subject": "S1",
                        "roi": "AIC",
                        "hemi": "L",
                        "x": 0.0,
                        "y": 1.0,
                        "z": 2.0,
                        "modality": "sound",
                        "label": 1,
                        "mask": True,
                        "mix": False,
                    }
                )
    frame = pd.DataFrame(rows)
    concat, meta, slices = concatenated_phase_matrix(
        frame, ("stimulus", "delay", "go", "response"), min_coverage=0.95
    )
    assert list(concat.index) == ["ch_all"]
    assert list(meta.index) == ["ch_all"]
    assert set(slices) == {"stimulus", "delay", "go", "response"}
    assert concat.shape[1] == sum(sl.stop - sl.start for sl in slices.values())

    H = np.ones((2, concat.shape[1]))
    H[0, slices["stimulus"]] = np.linspace(1.0, 0.1, slices["stimulus"].stop - slices["stimulus"].start)
    H[1, slices["stimulus"]] = np.linspace(0.1, 1.0, slices["stimulus"].stop - slices["stimulus"].start)
    names = orient_components_on_phase_segment(
        H, concat.columns, slices, name_phase="stimulus"
    )
    split = split_concat_components(H, concat.columns, slices)
    assert set(split) == set(slices)
    assert "sensory" in names.values()
    assert "sustain" in names.values()


def test_held_out_contrasts_use_paired_subjects_only():
    rows = []
    assignments = []
    for subject in ("S1", "S2"):
        for cluster, delay, response in (
            ("sustain", 1.0, 0.1),
            ("sensory", 0.1, 1.0),
        ):
            channel = f"{subject}_{cluster}"
            assignments.append(
                {
                    "channel": channel,
                    "subject": subject,
                    "functional_cluster": cluster,
                }
            )
            for phase, value in (("delay", delay), ("response", response)):
                for time in (0.1, 0.25, 0.5):
                    rows.append(
                        {
                            "channel": channel,
                            "phase": phase,
                            "time": time,
                            "value": value,
                        }
                    )

    contrasts = held_out_contrasts(
        pd.DataFrame(rows), pd.DataFrame(assignments)
    ).set_index("contrast")

    assert contrasts.loc["delay_plateau", "n_subjects"] == 2
    assert contrasts.loc["response_peak", "n_subjects"] == 2
    assert contrasts.loc["delay_plateau", "mean_predicted_difference"] > 0
    assert contrasts.loc["response_peak", "mean_predicted_difference"] > 0
