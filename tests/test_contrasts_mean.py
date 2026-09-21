from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.univariate import contrasts_mean as cm


def test_window_reduce_mean_and_p75():
    trial = np.array([[0.0, 1.0, 2.0, 4.0], [10.0, 10.0, 10.0, 10.0]])
    np.testing.assert_allclose(cm.window_reduce(trial, "mean"), [1.75, 10.0])
    np.testing.assert_allclose(cm.window_reduce(trial, "p75"), [2.5, 10.0])
    with pytest.raises(ValueError, match="window_stat"):
        cm.window_reduce(trial, "max")


def test_resolve_contrast_description_mean_and_p75():
    assert cm.resolve_contrast_description("RepeatVsPassive") == "RepeatVsPassiveMean"
    assert cm.resolve_contrast_description("RepeatVsPassive", "p75") == "RepeatVsPassiveP75"
    assert cm.resolve_contrast_description("DecisionVsRepeat", "p75") == "DecisionVsRepeatP75"


def test_run_phase_contrast_repeat_vs_passive_p75_description():
    phase_df = pd.concat(
        [
            _trial_rows("Repeat", "D0001_A1-2", [1.0, 1.1, 1.2, 0.9]),
            _trial_rows("Passive", "D0001_A1-2", [0.1, 0.0, 0.2, -0.1]),
        ],
        ignore_index=True,
    )
    result = cm.run_phase_contrast(
        phase_df,
        "RepeatVsPassive",
        subject="D0001",
        phase="Delay",
        n_perm=50,
        alpha=0.05,
        window_stat="p75",
    )
    assert result.loc[0, "contrast"] == "RepeatVsPassiveP75"


def test_parse_contrast_keys_default_is_lexical_trio():
    assert cm.parse_contrast_keys(None) == list(cm.DEFAULT_CONTRASTS)
    assert "RepeatVsPassive" not in cm.DEFAULT_CONTRASTS


def test_parse_contrast_keys_repeatable_and_comma():
    assert cm.parse_contrast_keys(["RepeatVsPassive"]) == ["RepeatVsPassive"]
    assert cm.parse_contrast_keys(["RepeatVsPassive,DecisionVsRepeat"]) == [
        "RepeatVsPassive",
        "DecisionVsRepeat",
    ]
    assert cm.parse_contrast_keys(["RepeatVsPassive", "DecisionVsRepeat"]) == [
        "RepeatVsPassive",
        "DecisionVsRepeat",
    ]


def test_parse_contrast_keys_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown contrast"):
        cm.parse_contrast_keys(["NotAContrast"])


def test_descriptions_for_contrasts_skips_unused_arms():
    assert cm.descriptions_for_contrasts(["RepeatVsPassive"]) == ("Repeat", "Passive")
    assert cm.descriptions_for_contrasts(["DecisionVsRepeat"]) == ("Decision", "Repeat")
    assert "Passive" not in cm.descriptions_for_contrasts(cm.DEFAULT_CONTRASTS)
    assert "Decision" not in cm.descriptions_for_contrasts(["RepeatVsPassive"])


def test_select_epoch_path_filters_recording():
    sound = SimpleNamespace(recording="sound")
    picture = SimpleNamespace(recording="picture")
    assert cm.select_epoch_path([sound, picture], recording="sound") is sound
    assert cm.select_epoch_path([sound, picture], recording="picture") is picture
    assert cm.select_epoch_path([sound, picture], recording="video") is None


def test_select_epoch_path_requires_unique_without_recording():
    sound = SimpleNamespace(recording="sound")
    picture = SimpleNamespace(recording="picture")
    with pytest.raises(ValueError, match="--recording"):
        cm.select_epoch_path([sound, picture], recording=None)
    assert cm.select_epoch_path([sound], recording=None) is sound
    assert cm.select_epoch_path([], recording=None) is None


def test_keep_correct_trials_uses_event_name():
    df = pd.DataFrame(
        {
            "event_name": [
                "Sound/Stimulus/Word/item/CORRECT",
                "Sound/Stimulus/Word/item/INCORRECT",
                "Picture/Stimulus/item/CORRECT",
                "Sound/Stimulus/Word/item/CORRECT/extra",
            ],
                "remark": ["", "", "", ""],
        }
    )
    kept = cm.keep_correct_trials(df)
    assert list(kept["event_name"]) == [
        "Sound/Stimulus/Word/item/CORRECT",
        "Picture/Stimulus/item/CORRECT",
        "Sound/Stimulus/Word/item/CORRECT/extra",
    ]


def test_keep_correct_trials_keeps_unlabeled_picture_naming():
    df = pd.DataFrame(
        {
            "event_name": [
                "Stimulus/apple.wav/sound/ListenSpeak",
                "Stimulus/duck.wav/sound/JustListen",
            ],
            "remark": ["", ""],
        }
    )
    kept = cm.keep_correct_trials(df)
    assert list(kept["event_name"]) == list(df["event_name"])


def test_keep_correct_trials_falls_back_to_remark():
    df = pd.DataFrame({"remark": ["CORRECT", "INCORRECT", "CORRECT"]})
    kept = cm.keep_correct_trials(df)
    assert list(kept["remark"]) == ["CORRECT", "CORRECT"]


def _trial_rows(description: str, channel: str, values: list[float], **extra) -> pd.DataFrame:
    rows = []
    for idx, value in enumerate(values):
        row = {
            "description": description,
            "channel": channel,
            "mean_hga": value,
            "lexicality": extra.get("lexicality", "Word"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def test_run_phase_contrast_repeat_vs_passive_masks():
    phase_df = pd.concat(
        [
            _trial_rows("Repeat", "D0001_A1-2", [1.0, 1.1, 1.2, 0.9]),
            _trial_rows("Passive", "D0001_A1-2", [0.1, 0.0, 0.2, -0.1]),
            _trial_rows("Decision", "D0001_A1-2", [9.0, 9.1, 9.2, 9.3]),
        ],
        ignore_index=True,
    )
    result = cm.run_phase_contrast(
        phase_df,
        "RepeatVsPassive",
        subject="D0001",
        phase="Stimulus",
        n_perm=200,
        alpha=0.05,
    )
    assert len(result) == 1
    assert result.loc[0, "contrast"] == "RepeatVsPassiveMean"
    assert result.loc[0, "mean_a"] == pytest.approx(1.05)
    assert result.loc[0, "mean_b"] == pytest.approx(0.05)
    assert result.loc[0, "mean_diff"] == pytest.approx(1.0)
    assert result.loc[0, "n_a"] == 4
    assert result.loc[0, "n_b"] == 4


def test_run_phase_contrast_repeat_vs_passive_empty_without_passive():
    phase_df = _trial_rows("Repeat", "D0001_A1-2", [1.0, 1.1, 1.2])
    result = cm.run_phase_contrast(
        phase_df,
        "RepeatVsPassive",
        subject="D0001",
        phase="Stimulus",
        n_perm=50,
        alpha=0.05,
    )
    assert result.empty
