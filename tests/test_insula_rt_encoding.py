"""Unit tests for Go-aligned HGA–RT encoding helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.reaction_time.summarize_insula_rt_encoding import (
    aligned_fast_slow_traces,
    collapse_electrode_r_to_subject,
    median_split_traces,
    window_mean_hga,
)


def test_window_mean_hga_averages_selected_samples():
    times = np.array([0.0, 0.1, 0.2, 0.3])
    # trials x channels x time
    X = np.arange(24, dtype=float).reshape(2, 3, 4)
    amp = window_mean_hga(
        X, times, channel_index=1, window_start=0.1, window_end=0.2
    )
    # channel 1 trial 0 samples [5, 6]; trial 1 samples [17, 18]
    np.testing.assert_allclose(amp, [5.5, 17.5])


def test_median_split_traces_separates_fast_and_slow():
    # 4 trials, 3 timepoints. Faster RT → lower amplitude here.
    trial_hga = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [10.0, 10.0, 10.0],
            [20.0, 20.0, 20.0],
        ]
    )
    rt_log = np.array([-1.0, -0.5, 0.5, 1.0])
    fast, slow, n_fast, n_slow = median_split_traces(trial_hga, rt_log)
    assert n_fast == 2
    assert n_slow == 2
    np.testing.assert_allclose(fast, [1.5, 1.5, 1.5])
    np.testing.assert_allclose(slow, [15.0, 15.0, 15.0])


def test_median_split_traces_drops_nonfinite_trials():
    trial_hga = np.array(
        [
            [1.0, 1.0],
            [np.nan, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ]
    )
    rt_log = np.array([0.0, 1.0, 2.0, np.nan])
    fast, slow, n_fast, n_slow = median_split_traces(trial_hga, rt_log)
    # Only trials 0 and 2 remain (finite HGA and RT); median of [0, 2] is 1.
    assert n_fast + n_slow == 2
    assert n_fast == 1
    assert n_slow == 1
    np.testing.assert_allclose(fast, [1.0, 1.0])
    np.testing.assert_allclose(slow, [3.0, 3.0])


def test_collapse_electrode_r_to_subject_uses_median():
    electrode = pd.DataFrame(
        {
            "task": ["LexicalDelay", "LexicalDelay", "LexicalDelay"],
            "subject": ["D0001", "D0001", "D0002"],
            "functional_cluster": ["motor", "motor", "motor"],
            "mean_hga_log_rt_r": [-0.2, -0.4, 0.1],
            "n_trials": [40, 40, 30],
        }
    )
    subjects = collapse_electrode_r_to_subject(electrode)
    assert len(subjects) == 2
    d0001 = subjects.loc[subjects.subject == "D0001"].iloc[0]
    assert d0001["n_electrodes"] == 2
    assert d0001["mean_hga_log_rt_r"] == pytest.approx(-0.3)


def test_aligned_fast_slow_traces_median_splits_one_channel():
    class _Phase:
        trial_meta = pd.DataFrame({"rt_log": [-1.0, -0.5, 0.5, 1.0]})
        channel_meta = pd.DataFrame({"channel": ["ch1"]})
        X = np.array(
            [
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[10.0, 10.0]],
                [[20.0, 20.0]],
            ]
        )
        times = np.array([0.0, 0.1])

    rows = aligned_fast_slow_traces(
        _Phase(),
        task="LexicalDelay",
        subject="D0001",
        retained_clusters=["motor"],
        by_cluster={"motor": ["ch1"]},
        min_trials=4,
    )
    frame = pd.DataFrame(rows)
    assert len(frame) == 2
    np.testing.assert_allclose(frame["hga_fast"], [1.5, 1.5])
    np.testing.assert_allclose(frame["hga_slow"], [15.0, 15.0])


def test_subject_encoding_r_medians_across_tasks():
    from src.reaction_time.summarize_insula_rt_encoding import subject_encoding_r

    rng = np.random.default_rng(0)
    rows = []
    for task, slope in (("LexicalDelay", -1.0), ("PictureNaming", -0.5)):
        hga = rng.normal(size=40)
        rt = slope * hga + rng.normal(scale=0.05, size=40)
        for hga_z, rt_log in zip(hga, rt):
            rows.append(
                {
                    "task": task,
                    "subject": "D0001",
                    "hga_z": hga_z,
                    "rt_log": rt_log,
                }
            )
    values = subject_encoding_r(pd.DataFrame(rows))
    assert values.shape == (1,)
    assert values[0] < -0.7


def test_permute_subject_encoding_t_left_tail_on_negative_association():
    from src.reaction_time.summarize_insula_rt_encoding import (
        permute_subject_encoding_t,
    )

    rng = np.random.default_rng(1)
    rows = []
    for i, subject in enumerate(("D0001", "D0002", "D0003", "D0004")):
        hga = rng.normal(size=50)
        rt = -0.8 * hga + rng.normal(scale=0.2, size=50)
        for hga_z, rt_log in zip(hga, rt):
            rows.append(
                {
                    "task": "LexicalDelay",
                    "subject": subject,
                    "hga_z": float(hga_z),
                    "rt_log": float(rt_log),
                }
            )
    result = permute_subject_encoding_t(
        pd.DataFrame(rows), n_perm=200, random_state=0
    )
    assert result["n_subjects"] == 4
    assert result["observed_t"] < 0
    assert result["p_left"] < 0.05
    assert result["null_t"].shape == (200,)
    # Null should straddle 0 more than the strongly negative observation.
    assert float(np.median(result["null_t"])) > result["observed_t"]
