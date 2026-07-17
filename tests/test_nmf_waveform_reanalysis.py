import numpy as np

from src.nmf.waveform_analysis import (
    held_out_contrasts,
    normalize_nmf_factors,
    orient_two_components,
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

    assert mapping == {0: "sustained_ramping", 1: "sensory_transient"}


def test_within_subject_permutation_returns_bounded_reproducible_p_value():
    labels = np.array(
        ["sustained_ramping", "sensory_transient"] * 4,
        dtype=object,
    )
    rois = np.array(["AIC", "PIC"] * 4)
    subjects = np.repeat(["S1", "S2"], 4)

    observed, p_value = within_subject_permutation_p(
        labels, rois, subjects, n_permutations=100, random_state=7
    )

    assert observed == 1.0
    assert 0.0 < p_value <= 1.0


def test_held_out_contrasts_use_paired_subjects_only():
    rows = []
    assignments = []
    for subject in ("S1", "S2"):
        for cluster, delay, response in (
            ("sustained_ramping", 1.0, 0.1),
            ("sensory_transient", 0.1, 1.0),
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
