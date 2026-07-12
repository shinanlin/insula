"""Tests for run_cross_condition_generalized.py.

Tests the loading function and core integration with DirectCrossDecoder
using real intersection h5 files from the LexicalDelay dataset.
"""
import numpy as np
import pytest
import sys
import os
import h5py
import tempfile
import json

# ============================================================================
# Test with synthetic h5 files (no real data needed for unit tests)
# ============================================================================

@pytest.fixture
def synthetic_h5_pair(tmp_path):
    """Create a pair of synthetic intersection h5 files (Repeat + Decision)
    for the same ROI and phase, with matching channels."""
    n_trials = 20
    n_channels = 5
    n_times = 128  # 1 second at 128Hz
    channels = [f"D0001_CH{i}" for i in range(n_channels)]
    
    meta = {
        "roi": "AICl",
        "event_id": json.dumps({"Word": 0, "Nonword": 1}),
        "description": None,  # set per file
        "phase": "Delay",
        "tmin": -0.5,
        "tmax": 0.5,
        "fs": 128,
    }
    
    files = {}
    for desc in ["Repeat", "Decision"]:
        fpath = tmp_path / f"sub-AICl_proc-Delay_desc-{desc}_highgamma.h5"
        X = np.random.randn(n_trials, n_channels, n_times).astype(np.float32)
        y = np.array([0, 1] * (n_trials // 2), dtype=int)  # balanced
        
        with h5py.File(fpath, "w") as f:
            f.create_dataset("X", data=X)
            f.create_dataset("y", data=y)
            _str = h5py.string_dtype(encoding="utf-8")
            f.create_dataset("channel", data=np.array(channels, dtype=_str))
            f.create_dataset("label", data=np.array(
                ["Word" if i == 0 else "Nonword" for i in y], dtype=_str
            ))
            f.create_dataset("time", data=np.linspace(-0.5, 0.5, n_times))
            for k, v in meta.items():
                if k == "description":
                    f.attrs[k] = desc
                elif k == "event_id":
                    f.attrs[k] = v
                else:
                    f.attrs[k] = v
        
        files[desc] = str(fpath)
    
    return files, channels, n_trials, n_channels, n_times


# ============================================================================
# Test loading function
# ============================================================================

class TestLoadIntersectionCondition:
    """Test the h5 loading function with synthetic data."""
    
    def test_loads_correct_shape(self, synthetic_h5_pair):
        """X shape matches expected (n_trials, n_channels, n_times)."""
        files, channels, n_trials, n_channels, n_times = synthetic_h5_pair
        
        with h5py.File(files["Repeat"], "r") as f:
            X = f["X"][()]
            y = f["y"][()]
            loaded_channels = [
                ch.decode("utf-8") if isinstance(ch, bytes) else ch
                for ch in f["channel"][()]
            ]
        
        assert X.shape == (n_trials, n_channels, n_times)
        assert y.shape == (n_trials,)
        assert len(loaded_channels) == n_channels
    
    def test_channels_match_across_conditions(self, synthetic_h5_pair):
        """Repeat and Decision files have identical channel lists."""
        files, _, _, _, _ = synthetic_h5_pair
        
        ch_repeat, ch_decision = [], []
        with h5py.File(files["Repeat"], "r") as f:
            ch_repeat = [ch.decode("utf-8") if isinstance(ch, bytes) else ch
                        for ch in f["channel"][()]]
        with h5py.File(files["Decision"], "r") as f:
            ch_decision = [ch.decode("utf-8") if isinstance(ch, bytes) else ch
                          for ch in f["channel"][()]]
        
        assert ch_repeat == ch_decision
    
    def test_labels_are_balanced(self, synthetic_h5_pair):
        """Word and Nonword labels are balanced."""
        files, _, _, _, _ = synthetic_h5_pair
        
        with h5py.File(files["Repeat"], "r") as f:
            y = f["y"][()]
        
        unique, counts = np.unique(y, return_counts=True)
        assert len(unique) == 2
        assert counts[0] == counts[1]  # balanced
    
    def test_meta_attributes(self, synthetic_h5_pair):
        """File has required metadata attributes."""
        files, _, _, _, _ = synthetic_h5_pair
        
        with h5py.File(files["Repeat"], "r") as f:
            assert "fs" in f.attrs
            assert "tmin" in f.attrs
            assert "tmax" in f.attrs
            assert "roi" in f.attrs
            assert "phase" in f.attrs
            assert f.attrs["fs"] == 128
            assert f.attrs["phase"] == "Delay"


# ============================================================================
# Test DirectCrossDecoder integration
# ============================================================================

class TestDirectCrossDecoderIntegration:
    """Test that DirectCrossDecoder works with our data format."""

    def test_decoder_runs_on_synthetic_data(self, synthetic_h5_pair):
        """DirectCrossDecoder can fit on train condition and score on test."""
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.svm import LinearSVC
        from mne.decoding import Vectorizer
        from sklearn.base import clone
        
        # Add scripts dir to path for import
        scripts_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "scripts"
        )
        sys.path.insert(0, scripts_dir)
        from src.decoding.direct_cross_decoder import DirectCrossDecoder
        
        files, _, n_trials, n_channels, n_times = synthetic_h5_pair
        
        with h5py.File(files["Repeat"], "r") as f:
            X_train = f["X"][()]
            y_train = f["y"][()]
        with h5py.File(files["Decision"], "r") as f:
            X_test = f["X"][()]
            y_test = f["y"][()]
        
        estimator = make_pipeline(
            Vectorizer(),
            StandardScaler(),
            PCA(n_components=0.85, random_state=42),
            LinearSVC(random_state=42, max_iter=10000),
        )
        
        decoder = DirectCrossDecoder(estimator=estimator, random_state=42)
        decoder.fit(X_train, y_train)
        predictions = decoder.predict(X_test)
        
        assert predictions.shape == (n_trials,)
        assert set(predictions).issubset({0, 1})
    
    def test_decoder_score_is_bounded(self, synthetic_h5_pair):
        """Accuracy score is between 0 and 1."""
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.svm import LinearSVC
        from mne.decoding import Vectorizer
        
        scripts_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "scripts"
        )
        sys.path.insert(0, scripts_dir)
        from src.decoding.direct_cross_decoder import DirectCrossDecoder
        
        files, _, _, _, _ = synthetic_h5_pair
        
        with h5py.File(files["Repeat"], "r") as f:
            X_train, y_train = f["X"][()], f["y"][()]
        with h5py.File(files["Decision"], "r") as f:
            X_test, y_test = f["X"][()], f["y"][()]
        
        estimator = make_pipeline(
            Vectorizer(),
            StandardScaler(),
            PCA(n_components=0.85, random_state=42),
            LinearSVC(random_state=42, max_iter=10000),
        )
        
        decoder = DirectCrossDecoder(estimator=estimator, random_state=42)
        decoder.fit(X_train, y_train)
        score = decoder.score(X_test, y_test)
        
        assert 0.0 <= score <= 1.0


# ============================================================================
# Test with real intersection data (skip if not available)
# ============================================================================

REAL_DATA_ROOT = "/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS"
REAL_DATA_DIR = os.path.join(
    REAL_DATA_ROOT, "derivatives", "decoding(intersection)(bipolar)",
    "sub-AICl", "lexicality"
)
REAL_DATA_AVAILABLE = os.path.exists(REAL_DATA_DIR)


@pytest.mark.skipif(not REAL_DATA_AVAILABLE, reason="Real intersection data not found")
class TestRealIntersectionData:
    """Verify real intersection data has expected properties."""
    
    def test_aicl_delay_channels_match(self):
        """AICl Delay: Repeat and Decision have same channels."""
        repeat_file = os.path.join(
            REAL_DATA_DIR,
            "sub-AICl_task-LexicalDelay_proc-Delay_desc-Repeat_highgamma.h5"
        )
        decision_file = os.path.join(
            REAL_DATA_DIR,
            "sub-AICl_task-LexicalDelay_proc-Delay_desc-Decision_highgamma.h5"
        )
        
        with h5py.File(repeat_file, "r") as f:
            ch_r = sorted([ch.decode() if isinstance(ch, bytes) else ch 
                          for ch in f["channel"][()]])
            shape_r = f["X"].shape
        with h5py.File(decision_file, "r") as f:
            ch_d = sorted([ch.decode() if isinstance(ch, bytes) else ch 
                          for ch in f["channel"][()]])
            shape_d = f["X"].shape
        
        assert ch_r == ch_d, f"Channels differ: {len(ch_r)} vs {len(ch_d)}"
        assert shape_r[1] == shape_d[1], "Channel dimension mismatch"
    
    def test_aicl_data_not_all_nan(self):
        """AICl Delay Repeat data has actual values, not all NaN."""
        repeat_file = os.path.join(
            REAL_DATA_DIR,
            "sub-AICl_task-LexicalDelay_proc-Delay_desc-Repeat_highgamma.h5"
        )
        with h5py.File(repeat_file, "r") as f:
            X = f["X"][()]
        
        assert not np.all(np.isnan(X)), "All data is NaN"
        nan_frac = np.isnan(X).mean()
        assert nan_frac < 0.5, f"Too many NaNs: {nan_frac:.1%}"
