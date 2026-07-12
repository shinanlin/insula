"""Tests for src/generate_xcorr_viewer.py"""

import json
import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.xcorr.generate_xcorr_viewer import (
    bids_to_recon_id,
    get_shank_prefix,
    classify_channels,
    filter_same_hemisphere,
    get_shank_channels,
    compute_xcorr_matrix,
    build_html,
    color_vertices,
)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
class TestHelpers:
    def test_bids_to_recon_id(self):
        assert bids_to_recon_id("D0094") == "D94"
        assert bids_to_recon_id("D0106") == "D106"
        assert bids_to_recon_id("D0007") == "D7"

    def test_get_shank_prefix(self):
        assert get_shank_prefix("D0040_L1IF2-3") == "D0040_L1IF"
        assert get_shank_prefix("D0094_LFAI3-4") == "D0094_LFAI"
        assert get_shank_prefix("D0094_LIA10-11") == "D0094_LIA"


class TestClassifyChannels:
    def _make_parc(self, rois):
        return pd.DataFrame({
            "channel": [f"ch{i}" for i in range(len(rois))],
            "roi": rois,
            "label": ["lbl"] * len(rois),
            "hemi": ["L"] * len(rois),
        })

    def test_basic(self):
        parc = self._make_parc(["INS", "IFG", "IFGs", "STG", "INS"])
        ins, ifg = classify_channels(parc)
        assert set(ins) == {"ch0", "ch4"}
        assert set(ifg) == {"ch1", "ch2"}

    def test_no_ifg(self):
        parc = self._make_parc(["INS", "STG"])
        ins, ifg = classify_channels(parc)
        assert len(ins) == 1
        assert len(ifg) == 0


class TestFilterSameHemisphere:
    def test_shared(self):
        parc = pd.DataFrame({
            "channel": ["a", "b", "c"],
            "hemi": ["L", "L", "R"],
            "roi": ["INS", "IFG", "INS"],
        })
        ins, ifg = filter_same_hemisphere(["a", "c"], ["b"], parc)
        assert ins == ["a"]  # only L hemisphere
        assert ifg == ["b"]

    def test_no_shared(self):
        parc = pd.DataFrame({
            "channel": ["a", "b"],
            "hemi": ["L", "R"],
            "roi": ["INS", "IFG"],
        })
        ins, ifg = filter_same_hemisphere(["a"], ["b"], parc)
        assert ins == []
        assert ifg == []


class TestGetShankChannels:
    def test_basic(self):
        roi = ["D0094_LFAI3-4", "D0094_LIA1-2"]
        all_chs = [
            "D0094_LFAI1-2", "D0094_LFAI2-3", "D0094_LFAI3-4",
            "D0094_LIA1-2", "D0094_LIA2-3",
            "D0094_ROG1-2",  # different shank
        ]
        result = get_shank_channels(roi, all_chs)
        assert "D0094_ROG1-2" not in result
        assert len(result) == 5


class TestComputeXcorr:
    def test_shape(self):
        rng = np.random.default_rng(42)
        n_trials, n_chan, n_time = 5, 3, 128
        xdata = rng.standard_normal((n_trials, n_chan, n_time)).astype(np.float32)
        sfreq = 128.0
        xcorr, lag_times = compute_xcorr_matrix(xdata, sfreq, max_lag_s=0.5)
        expected_lags = int(0.5 * sfreq) * 2 + 1
        assert xcorr.shape == (n_trials, n_chan, n_chan, expected_lags)
        assert lag_times.shape == (expected_lags,)
        assert lag_times[0] == pytest.approx(-0.5)
        assert lag_times[-1] == pytest.approx(0.5)

    def test_autocorrelation_peak_at_zero(self):
        rng = np.random.default_rng(0)
        n_trials, n_chan, n_time = 3, 2, 256
        xdata = rng.standard_normal((n_trials, n_chan, n_time)).astype(np.float32)
        sfreq = 256.0
        xcorr, lag_times = compute_xcorr_matrix(xdata, sfreq, max_lag_s=1.0)
        mean_xcorr = xcorr.mean(axis=0)
        zero_idx = np.argmin(np.abs(lag_times))
        # Auto-correlation peak should be at or near lag=0
        for c in range(n_chan):
            peak_idx = np.argmax(mean_xcorr[c, c])
            assert abs(peak_idx - zero_idx) <= 2

    def test_values_non_negative(self):
        rng = np.random.default_rng(1)
        xdata = rng.standard_normal((4, 2, 100)).astype(np.float32)
        xcorr, _ = compute_xcorr_matrix(xdata, 100.0, max_lag_s=0.5)
        assert np.all(xcorr >= 0), "Squared xcorr should be non-negative"


class TestBuildHtml:
    def test_produces_valid_html(self):
        n_v = 100
        n_f = 50
        rng = np.random.default_rng(42)
        verts = rng.standard_normal((n_v, 3)).astype(np.float32)
        faces = rng.integers(0, n_v, (n_f, 3)).astype(np.int32)
        colors = np.full((n_v, 3), 200, dtype=np.uint8)
        electrodes = [
            {"name": "ch1", "x": 0, "y": 0, "z": 0, "roi": "INS", "label": "lbl", "type": "insula"},
            {"name": "ch2", "x": 1, "y": 1, "z": 1, "roi": "IFG", "label": "lbl", "type": "ifg"},
        ]
        xcorr = rng.random((2, 2, 51)).astype(np.float32)
        lag_times = np.linspace(-1, 1, 51).astype(np.float32)
        meta = {"subject": "D0094", "task": "Test", "phase": "Response", "desc": "Repeat"}

        html = build_html(
            verts, faces, colors,
            verts, faces, colors,
            electrodes, ["ch1", "ch2"], xcorr, lag_times, meta,
        )
        assert "<!DOCTYPE html>" in html
        assert "Three" in html or "three" in html
        assert "Plotly" in html or "plotly" in html
        assert "D0094" in html
        assert "__DATA_JSON__" not in html  # should be replaced


class TestColorVertices:
    def test_default_color(self):
        verts = np.zeros((10, 3))
        colors = color_vertices(verts, [], "lh")
        assert colors.shape == (10, 3)
        assert np.all(colors == [220, 220, 220])
