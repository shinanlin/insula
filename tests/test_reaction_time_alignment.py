from __future__ import annotations

import numpy as np
import pandas as pd

from src.reaction_time.run_reaction_time import _prepare_data


def _value(trial: int, channel: str, time: float) -> float:
    channel_offset = {"ch2": 20.0, "ch10": 10.0}[channel]
    time_offset = {-0.1: 1.0, 0.1: 2.0}[time]
    return trial * 100.0 + channel_offset + time_offset


def test_prepare_data_preserves_trial_rt_and_channel_alignment():
    rows = []
    rt_by_trial = {2: 0.8, 1: 0.4, 3: np.nan}

    # Deliberately use non-sorted trial, channel, and time appearance order.
    for trial in [2, 1, 3]:
        for channel in ["ch2", "ch10"]:
            for time in [0.1, -0.1]:
                rows.append(
                    {
                        "phase": "Go",
                        "description": "Repeat",
                        "trial": trial,
                        "channel": channel,
                        "time": time,
                        "value": _value(trial, channel, time),
                        "rt": rt_by_trial[trial],
                    }
                )

    X, rt, times, channels = _prepare_data(
        pd.DataFrame(rows),
        phase="Go",
        description="Repeat",
    )

    np.testing.assert_array_equal(channels, ["ch2", "ch10"])
    np.testing.assert_array_equal(times, [-0.1, 0.1])
    np.testing.assert_array_equal(rt, [0.8, 0.4])
    assert X.shape == (2, 2, 2)

    np.testing.assert_array_equal(
        X[0, 0],
        [_value(2, "ch2", -0.1), _value(2, "ch2", 0.1)],
    )
    np.testing.assert_array_equal(
        X[0, 1],
        [_value(2, "ch10", -0.1), _value(2, "ch10", 0.1)],
    )
    np.testing.assert_array_equal(
        X[1, 0],
        [_value(1, "ch2", -0.1), _value(1, "ch2", 0.1)],
    )
    np.testing.assert_array_equal(
        X[1, 1],
        [_value(1, "ch10", -0.1), _value(1, "ch10", 0.1)],
    )
