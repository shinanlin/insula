import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.run_xcorr_pair_permutation import (
    compute_pair_xcorr_trials,
    build_trial_shuffle_null,
    build_circular_shift_null,
    cluster_permutation_test,
)


class TestComputePairXcorrTrials:
    def test_shape_and_lag_axis(self):
        rng = np.random.default_rng(42)
        n_trials, n_time = 6, 200
        sfreq = 100.0
        max_lag_s = 0.5

        src = rng.standard_normal((n_trials, n_time)).astype(np.float32)
        tgt = rng.standard_normal((n_trials, n_time)).astype(np.float32)

        xcorr_trials, lag_times = compute_pair_xcorr_trials(
            src,
            tgt,
            sfreq=sfreq,
            max_lag_s=max_lag_s,
        )

        expected_lags = int(max_lag_s * sfreq) * 2 + 1
        assert xcorr_trials.shape == (n_trials, expected_lags)
        assert lag_times.shape == (expected_lags,)
        assert lag_times[0] == -max_lag_s
        assert lag_times[-1] == max_lag_s
        assert np.all(xcorr_trials >= 0)


class TestTrialShuffleNull:
    def test_null_shape(self):
        rng = np.random.default_rng(0)
        n_trials, n_time = 8, 160
        sfreq = 80.0

        src = rng.standard_normal((n_trials, n_time)).astype(np.float32)
        tgt = rng.standard_normal((n_trials, n_time)).astype(np.float32)

        n_perm = 20
        null_curves, lag_times = build_trial_shuffle_null(
            src,
            tgt,
            sfreq=sfreq,
            max_lag_s=0.5,
            n_perm=n_perm,
            random_state=42,
        )

        expected_lags = int(0.5 * sfreq) * 2 + 1
        assert null_curves.shape == (n_perm, expected_lags)
        assert lag_times.shape == (expected_lags,)


class TestCircularShiftNull:
    def test_null_shape_and_reproducibility(self):
        rng = np.random.default_rng(7)
        n_trials, n_time = 7, 180
        sfreq = 90.0

        src = rng.standard_normal((n_trials, n_time)).astype(np.float32)
        tgt = rng.standard_normal((n_trials, n_time)).astype(np.float32)

        n_perm = 24
        null_a, lag_a = build_circular_shift_null(
            src,
            tgt,
            sfreq=sfreq,
            max_lag_s=0.4,
            n_perm=n_perm,
            random_state=123,
        )
        null_b, lag_b = build_circular_shift_null(
            src,
            tgt,
            sfreq=sfreq,
            max_lag_s=0.4,
            n_perm=n_perm,
            random_state=123,
        )

        expected_lags = int(0.4 * sfreq) * 2 + 1
        assert null_a.shape == (n_perm, expected_lags)
        assert lag_a.shape == (expected_lags,)
        assert np.allclose(lag_a, lag_b)
        assert np.allclose(null_a, null_b)


class TestClusterPermutation:
    def test_detects_synthetic_cluster(self):
        rng = np.random.default_rng(123)
        n_lag = 101
        lag_times = np.linspace(-0.5, 0.5, n_lag)

        # Null curves around small baseline
        null_curves = 0.01 + 0.002 * rng.standard_normal((400, n_lag)).astype(np.float32)

        # Observed with a clear positive bump near 0.12s
        observed = 0.01 + 0.002 * rng.standard_normal(n_lag).astype(np.float32)
        bump = (lag_times > 0.08) & (lag_times < 0.16)
        observed[bump] += 0.03

        res = cluster_permutation_test(
            observed_curve=observed,
            null_curves=null_curves,
            alpha=0.05,
        )

        sig = [c for c in res['clusters'] if c['significant']]
        assert len(sig) >= 1

        # At least one significant cluster overlaps synthetic bump window
        overlaps = []
        for cl in sig:
            s, e = cl['start_idx'], cl['end_idx']
            cl_mask = np.zeros(n_lag, dtype=bool)
            cl_mask[s:e] = True
            overlaps.append(np.any(cl_mask & bump))

        assert any(overlaps)
