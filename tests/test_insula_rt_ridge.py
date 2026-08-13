from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.reaction_time.insula_ridge import (
    ClusterResult,
    fit_window_scores,
    joint_cluster_correction,
    make_group_splits,
    make_permutation_seeds,
    make_shuffled_targets,
    parse_item_id,
)
from src.reaction_time.insula_rt_data import (
    PhaseData,
    load_strict_insula_parcellation,
    match_target_go_response,
)
from src.reaction_time.insula_rt_io import (
    PhaseModelResult,
    write_phase_result,
)
from src.reaction_time.summarize_insula_rt_direction import mean_hga_rt_direction
from src.reaction_time.run_insula_rt_ridge import sliding_windows
from src.reaction_time.summarize_insula_rt_ridge import summarize_results


def test_task_aware_item_parser():
    assert (
        parse_item_id("LexicalDelay", "Delay/Repeat/Word/hazel/CORRECT")
        == "Word/hazel"
    )
    assert (
        parse_item_id("PhonemeSequence", "Go/Repeat/LS/vug/CORRECT")
        == "LS/vug"
    )
    assert (
        parse_item_id("PictureNaming", "GoCue/apple.png/image/ListenSpeak")
        == "apple"
    )
    assert (
        parse_item_id("PictureNaming", "GoCue/apple.wav/sound/ListenSpeak")
        == "apple"
    )


def test_grouped_outer_cv_never_splits_an_item():
    groups = np.repeat(["apple", "duck", "star", "umbrella"], 3)
    splits, fold_id = make_group_splits(groups, n_splits=10, random_state=42)
    assert len(splits) == 4
    assert set(fold_id) == {0, 1, 2, 3}
    for train, test in splits:
        assert set(groups[train]).isdisjoint(groups[test])


def test_permutation_is_unrestricted_across_items():
    # The first three positions can be imagined as item A and the last three
    # as item B. Seed 0 moves values across that boundary; no group is supplied.
    shuffled = make_shuffled_targets(np.arange(6, dtype=float), np.array([0]))
    assert set(shuffled[:, 0]) == set(np.arange(6, dtype=float))
    assert any(value >= 3 for value in shuffled[:3, 0])


def test_time_windows_save_exact_sample_coordinates():
    times = np.arange(-1.0, 1.5, 1 / 128)
    starts, stops, t0, t1, center = sliding_windows(
        times, 128, window_s=0.2, step_s=0.02, max_windows=2
    )
    np.testing.assert_array_equal(starts, [0, 3])
    np.testing.assert_array_equal(stops, [26, 29])
    np.testing.assert_allclose(t0, times[starts])
    np.testing.assert_allclose(t1, times[stops - 1])
    np.testing.assert_allclose(center, (t0 + t1) / 2)


def test_strict_insula_selection_excludes_mixed_contacts(tmp_path: Path):
    root = tmp_path / "derivatives" / "parcellation" / "sub-DTEST" / "bipolar"
    root.mkdir(parents=True)
    pd.DataFrame(
        {
            "name": ["a", "p", "mixed", "motor"],
            "roi": ["AIC", "PIC", "AIC–IFG", "PrG"],
            "hemi": ["L", "R", "L", "R"],
            "x": [1, 2, 3, 4],
            "y": [1, 2, 3, 4],
            "z": [1, 2, 3, 4],
            "x_t": [11, 12, 13, 14],
            "y_t": [11, 12, 13, 14],
            "z_t": [11, 12, 13, 14],
        }
    ).to_csv(root / "sub-DTEST_hammers.csv", index=False)
    selected = load_strict_insula_parcellation(tmp_path, subject="DTEST")
    assert selected["channel"].tolist() == ["a", "p"]
    np.testing.assert_array_equal(selected["x_template"], [11, 12])


def test_event_sample_alignment_does_not_use_next_trials_response():
    target = pd.DataFrame(
        {
            "trial_index": [0, 1],
            "target_event_sample": [100.0, 300.0],
            "item_id": ["a", "b"],
            "source_row": [0, 1],
        }
    )
    go = pd.DataFrame(
        {
            "event_sample": [200.0, 400.0],
            "onset": [2.0, 4.0],
            "event_name": ["go/a", "go/b"],
            "item_id": ["a", "b"],
        }
    )
    # Trial a's response epoch is absent. Trial b's response must not be used
    # for a merely because it is the next available Response row.
    response = pd.DataFrame(
        {
            "event_sample": [450.0],
            "onset": [4.5],
            "event_name": ["resp/b"],
            "item_id": ["b"],
        }
    )
    matched = match_target_go_response(target, go, response, phase="Delay")
    assert matched["trial_index"].tolist() == [1]
    assert matched["response_onset"].tolist() == [4.5]


def test_oof_ridge_outputs_one_prediction_per_trial():
    rng = np.random.RandomState(2)
    groups = np.repeat(["a", "b", "c", "d"], 3)
    latent = rng.randn(len(groups))
    X = np.stack(
        [
            np.column_stack([latent, latent + 0.05 * rng.randn(len(groups))]),
            rng.randn(len(groups), 2),
        ],
        axis=1,
    )
    y = latent + 0.05 * rng.randn(len(groups))
    splits, _ = make_group_splits(groups, n_splits=4, random_state=1)
    seeds = make_permutation_seeds(len(splits), 2, random_state=3)
    result = fit_window_scores(
        X,
        y,
        groups,
        splits,
        seeds,
        alphas=np.array([0.1, 1.0]),
        inner_splits=3,
        n_jobs=1,
    )
    assert result.oof_prediction.shape == (2, len(groups))
    assert np.isfinite(result.oof_prediction).all()
    assert result.perm_score_r.shape == (2, 2)
    assert result.score_r[0] > result.score_r[1]


def test_mean_hga_direction_negative_means_higher_hga_shorter_rt():
    amplitude = np.linspace(-1, 1, 20)
    X = np.stack([amplitude, amplitude], axis=1)[:, None, :]
    log_rt = -0.5 * amplitude
    correlation, _, raw_slope, n_trials = mean_hga_rt_direction(
        X,
        np.array([0.0, 0.1]),
        log_rt,
        channel_index=0,
        window_start=0.0,
        window_end=0.1,
    )
    assert n_trials == 20
    assert correlation < -0.99
    assert raw_slope < 0


def test_joint_cluster_correction_detects_run_not_isolated_point():
    rng = np.random.RandomState(4)
    permutations = rng.normal(0, 0.05, size=(1, 14, 199))
    observed = np.zeros((1, 14))
    observed[0, 0] = 0.8
    observed[0, 3:12] = 0.8
    corrected = joint_cluster_correction(
        {
            "Delay": (observed, permutations),
            "Go": (np.zeros_like(observed), permutations.copy()),
        }
    )["Delay"]
    assert not corrected.sig_mask_fwer[0, 0]
    assert corrected.sig_mask_fwer[0, 3:12].all()


def test_joint_cluster_correction_never_marks_negative_prediction():
    rng = np.random.RandomState(8)
    permutations = rng.normal(-0.5, 0.03, size=(1, 14, 199))
    observed = np.full((1, 14), -0.5)
    # This run exceeds its deliberately negative permutation null and would
    # pass the cluster test without an explicit positive-OOF-r requirement.
    observed[0, 3:12] = -0.2
    corrected = joint_cluster_correction(
        {
            "Delay": (observed, permutations),
            "Go": (np.full_like(observed, -0.5), permutations.copy()),
        }
    )["Delay"]
    assert not corrected.sig_mask_fwer.any()


def _dummy_phase_data() -> PhaseData:
    trial_meta = pd.DataFrame(
        {
            "rt_raw": [0.4, 0.5, 0.6, 0.7],
            "rt_log": np.log([0.4, 0.5, 0.6, 0.7]),
            "trial_index": [0, 1, 2, 3],
            "source_row": [0, 1, 2, 3],
            "target_event_sample": [10, 20, 30, 40],
            "go_event_sample": [11, 21, 31, 41],
            "response_event_sample": [15, 26, 37, 48],
            "target_onset": [1.0, 2.0, 3.0, 4.0],
            "go_onset": [1.1, 2.1, 3.1, 4.1],
            "response_onset": [1.5, 2.6, 3.7, 4.8],
            "trial_uid": ["d:0", "d:1", "d:2", "d:3"],
            "item_id": ["a", "a", "b", "b"],
            "recording": ["d"] * 4,
            "target_event_name": ["event"] * 4,
            "go_event_name": ["go"] * 4,
            "response_event_name": ["response"] * 4,
            "source_file": ["source.h5"] * 4,
        }
    )
    channel_meta = pd.DataFrame(
        {
            "channel": ["DTEST_A1-2"],
            "roi": ["AIC"],
            "hemi": ["L"],
            "label": ["insula"],
            "center": ["insula short gyrus L"],
            "mix": [False],
            "x_template": [1.0],
            "y_template": [2.0],
            "z_template": [3.0],
            "x_native": [4.0],
            "y_native": [5.0],
            "z_native": [6.0],
            "x_mni": [7.0],
            "y_mni": [8.0],
            "z_mni": [9.0],
        }
    )
    return PhaseData(
        X=np.zeros((4, 1, 3)),
        times=np.array([-0.1, 0.0, 0.1]),
        sfreq=10.0,
        trial_meta=trial_meta,
        channel_meta=channel_meta,
        task="LexicalDelay",
        subject="DTEST",
        phase="Delay",
        description="Repeat",
    )


def test_h5_round_trip_and_summary_schema(tmp_path: Path):
    data = _dummy_phase_data()
    cluster = ClusterResult(
        point_p=np.array([[0.01, 0.01]]),
        cluster_p_fwer=np.array([[0.04, 0.04]]),
        sig_mask_fwer=np.array([[True, True]]),
    )
    result = PhaseModelResult(
        score_r=np.array([[-0.2, 0.3]]),
        score_r2=np.array([[-0.1, 0.02]]),
        score_mae=np.array([[0.2, 0.1]]),
        perm_score_r=np.zeros((1, 2, 5)),
        oof_prediction=np.zeros((1, 2, 4)),
        fold_id=np.array([0, 0, 1, 1]),
        window_start=np.array([-0.2, -0.1]),
        window_end=np.array([-0.1, 0.0]),
        window_center=np.array([-0.15, -0.05]),
        cluster=cluster,
    )
    path = (
        tmp_path
        / "task-LexicalDelay"
        / "sub-DTEST"
        / "sub-DTEST_task-LexicalDelay_proc-Delay_desc-Repeat_rt-ridge.h5"
    )
    write_phase_result(path, data=data, result=result, config={"n_perm": 5})
    with h5py.File(path, "r") as h5:
        assert h5.attrs["permutation"] == "unrestricted_training_rt_shuffle"
        assert h5["scores/oof_prediction"].shape == (1, 2, 4)
        assert h5["channels/roi"].asstr()[0] == "AIC"

    coverage, electrodes, clusters = summarize_results(
        tmp_path, assignments_path=None
    )
    assert coverage.loc[0, "n_electrodes"] == 1
    assert bool(electrodes.loc[0, "significant"])
    assert electrodes.loc[0, "n_significant_windows"] == 1
    assert len(clusters) == 1
    assert (tmp_path / "summaries" / "significant_clusters.csv").is_file()
