"""Integration test: run the generalized decoding on real AICl data.

This test loads real intersection h5 files and runs a tiny 
cross-condition generalized decode (1 fold, 1 perm, coarse window)
to verify the full pipeline doesn't crash — specifically checking
that OOB window slicing is handled correctly.

Run: python -m pytest tests/test_cross_generalized_integration.py -v -s
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
class TestGeneralizedDecodingIntegration:

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
        
        # Effective tmax from data dimensions
        tmax = tmin + X1.shape[-1] / fs
        
        return X1, y1, X2, y2, fs, tmin, tmax

    def test_generalized_runs_without_error(self, load_aicl_delay):
        """Full generalized decode pipeline completes without crashing.
        
        Uses coarse params (window=0.5, step=0.5, 1 fold, 1 perm)
        to keep test fast. The key check is that OOB windows don't crash.
        """
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.svm import LinearSVC
        from mne.decoding import Vectorizer
        from ieeg.calc.oversample import MinimumNaNSplit
        from direct_cross_decoder import (
            DirectCrossDecoder,
            direct_cross_domain_generalized_permutation_scores,
        )
        
        X1, y1, X2, y2, fs, tmin, tmax = load_aicl_delay
        
        estimator = make_pipeline(
            Vectorizer(),
            StandardScaler(),
            PCA(n_components=0.85, random_state=42),
            LinearSVC(random_state=42, max_iter=10000),
        )
        
        decoder = DirectCrossDecoder(estimator=estimator, random_state=42)
        cv = MinimumNaNSplit(n_splits=2, n_repeats=1)
        
        obs_scores, perm_scores, pvals_fdr = direct_cross_domain_generalized_permutation_scores(
            X1=X1, y1=y1, X2=X2, y2=y2,
            cv=cv,
            cross_decoder=decoder,
            scoring="accuracy",
            n_permutations=1,
            n_jobs=1,
            random_state=42,
            window=0.5,
            step=0.5,
            fs=fs,
            train_tmin=tmin,
            train_tmax=tmax,
            test_tmin=tmin,
            test_tmax=tmax,
        )
        
        # Check output shapes
        assert obs_scores.ndim == 3  # (T_train, T_test, n_folds)
        assert perm_scores.ndim == 4  # (T_train, T_test, n_perm, n_folds)
        assert pvals_fdr.ndim == 2  # (T_train, T_test)
        
        # Valid cells should have reasonable accuracy values
        valid = ~np.isnan(obs_scores)
        assert valid.any(), "All cells are NaN — no valid windows at all"
        assert obs_scores[valid].min() >= 0.0
        assert obs_scores[valid].max() <= 1.0
        
        # p-values should be in [0, 1]
        assert pvals_fdr.min() >= 0.0
        assert pvals_fdr.max() <= 1.0
        
        print(f"\n  obs_scores shape: {obs_scores.shape}")
        print(f"  Valid cells: {valid.sum()} / {valid.size}")
        print(f"  Mean accuracy (valid): {np.nanmean(obs_scores):.3f}")
        print(f"  Sig cells (p<0.05): {(pvals_fdr < 0.05).sum()}")

    def test_fine_step_also_works(self, load_aicl_delay):
        """Finer step (0.3s) that generates more time points — more OOB risk."""
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.svm import LinearSVC
        from mne.decoding import Vectorizer
        from ieeg.calc.oversample import MinimumNaNSplit
        from direct_cross_decoder import (
            DirectCrossDecoder,
            direct_cross_domain_generalized_permutation_scores,
        )
        
        X1, y1, X2, y2, fs, tmin, tmax = load_aicl_delay
        
        estimator = make_pipeline(
            Vectorizer(),
            StandardScaler(),
            PCA(n_components=0.85, random_state=42),
            LinearSVC(random_state=42, max_iter=10000),
        )
        
        decoder = DirectCrossDecoder(estimator=estimator, random_state=42)
        cv = MinimumNaNSplit(n_splits=2, n_repeats=1)
        
        # window=0.5, step=0.3 (same params as the smoke test that was failing)
        obs_scores, perm_scores, pvals_fdr = direct_cross_domain_generalized_permutation_scores(
            X1=X1, y1=y1, X2=X2, y2=y2,
            cv=cv,
            cross_decoder=decoder,
            scoring="accuracy",
            n_permutations=1,
            n_jobs=1,
            random_state=42,
            window=0.5,
            step=0.3,
            fs=fs,
            train_tmin=tmin,
            train_tmax=tmax,
            test_tmin=tmin,
            test_tmax=tmax,
        )
        
        assert obs_scores.ndim == 3
        valid = ~np.isnan(obs_scores)
        assert valid.any()
        
        print(f"\n  obs_scores shape: {obs_scores.shape}")
        print(f"  Valid cells: {valid.sum()} / {valid.size}")
        print(f"  Mean accuracy (valid): {np.nanmean(obs_scores):.3f}")
