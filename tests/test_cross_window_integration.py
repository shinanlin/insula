"""Integration test: run the window decoding on real AICl data.

Run: python -m pytest tests/test_cross_window_integration.py -v -s
"""
import numpy as np
import pytest
import sys
import os

# Add project root for imports
scripts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
sys.path.insert(0, scripts_dir)

REAL_DATA_DIR = "/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/derivatives/decoding(intersection)(bipolar)/sub-AICl/lexicality"
REAL_DATA_AVAILABLE = os.path.exists(REAL_DATA_DIR)


@pytest.mark.skipif(not REAL_DATA_AVAILABLE, reason="Real intersection data not found")
class TestWindowDecodingIntegration:

    @pytest.fixture
    def load_aicl_delay(self):
        """Load AICl Delay Repeat/Decision data."""
        import h5py
        
        repeat_f = os.path.join(
            REAL_DATA_DIR,
            "sub-AICl_task-LexicalDelay_proc-Delay_desc-Repeat_highgamma.h5"
        )
        decision_f = os.path.join(
            REAL_DATA_DIR,
            "sub-AICl_task-LexicalDelay_proc-Delay_desc-Decision_highgamma.h5"
        )
        
        with h5py.File(repeat_f, "r") as f:
            X1 = f["X"][()]
            y1 = f["y"][()]
            fs = int(f.attrs["fs"])
            tmin = float(f.attrs["tmin"])
        
        with h5py.File(decision_f, "r") as f:
            X2 = f["X"][()]
            y2 = f["y"][()]
        
        # Align trial counts
        n_min = min(len(y1), len(y2))
        X1, y1 = X1[:n_min], y1[:n_min]
        X2, y2 = X2[:n_min], y2[:n_min]
        
        return X1, y1, X2, y2, fs, tmin

    def test_window_runs_without_error(self, load_aicl_delay):
        """Full window decode pipeline completes without crashing."""
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.svm import LinearSVC
        from mne.decoding import Vectorizer
        from ieeg.calc.oversample import MinimumNaNSplit
        from src.decoding.direct_cross_decoder import (
            DirectCrossDecoder,
            direct_cross_domain_permutation_scores,
        )
        
        X1, y1, X2, y2, fs, data_tmin = load_aicl_delay
        
        # Manually extract the Delay window (0 to 0.5)
        window_tmin = 0.0
        window_tmax = 0.5
        
        start_sample = int(round((window_tmin - data_tmin) * fs))
        end_sample = int(round((window_tmax - data_tmin) * fs))
        
        X1 = X1[..., start_sample:end_sample]
        X2 = X2[..., start_sample:end_sample]
        
        estimator = make_pipeline(
            Vectorizer(),
            StandardScaler(),
            PCA(n_components=0.85, random_state=42),
            LinearSVC(random_state=42, max_iter=10000),
        )
        
        decoder = DirectCrossDecoder(estimator=estimator, random_state=42)
        cv = MinimumNaNSplit(n_splits=2, n_repeats=1)
        
        obs_scores, perm_scores, p_value = direct_cross_domain_permutation_scores(
            X1=X1, y1=y1, X2=X2, y2=y2,
            cv=cv,
            cross_decoder=decoder,
            scoring="accuracy",
            n_permutations=2,
            n_jobs=2,
            random_state=42,
        )
        
        assert len(obs_scores) == 2  # 2 folds
        assert perm_scores.shape == (2, 2)  # (n_folds, n_perm)
        assert 0.0 <= p_value <= 1.0
        
        print(f"\n  obs_scores: {obs_scores}")
        print(f"  Mean accuracy: {np.mean(obs_scores):.3f}")
        print(f"  p-value: {p_value:.4f}")
