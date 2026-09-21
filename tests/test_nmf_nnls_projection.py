"""Unit tests for fixed-W NNLS projection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.nmf.nnls_projection import (
    COMPONENT_NAMES,
    load_group_basis,
    project_nnls,
    write_trace_h5,
    read_trace_h5,
)


def test_project_nnls_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n_ch, k, n_times, n_trials = 12, 3, 20, 5
    W = rng.random((n_ch, k))
    H_true = rng.random((n_trials, k, n_times))
    X = np.einsum("ck,nkt->nct", W, H_true)
    H_hat = project_nnls(X, W)
    assert H_hat.shape == (n_trials, k, n_times)
    assert np.isfinite(H_hat).all()
    # Exact non-negative factorization should recover H closely.
    assert np.allclose(H_hat, H_true, atol=1e-5, rtol=1e-4)


def test_project_nnls_handles_nan_channels():
    rng = np.random.default_rng(1)
    W = rng.random((6, 3))
    H_true = rng.random((2, 3, 8))
    X = np.einsum("ck,nkt->nct", W, H_true)
    X[:, 0, :] = np.nan  # drop channel 0 for all trials
    H_hat = project_nnls(X, W)
    assert H_hat.shape == X.shape[:1] + (3, X.shape[2])
    assert np.isfinite(H_hat).all()
    # Remaining channels still identify a non-negative solution.
    assert (H_hat >= -1e-10).all()


def test_write_read_trace_h5_roundtrip(tmp_path):
    H = np.random.default_rng(2).random((4, 3, 10)).astype(np.float32)
    times = np.linspace(-0.5, 1.0, 10)
    trials = pd.DataFrame(
        {
            "subject": ["D0001"] * 4,
            "trial_index": np.arange(4),
            "n_shared_channels": [5] * 4,
        }
    )
    path = tmp_path / "toy_nnls.h5"
    write_trace_h5(
        path,
        H=H,
        times=times,
        trials=trials,
        attrs={"method": "nnls", "task": "LexicalDelay", "phase": "Stimulus"},
    )
    loaded = read_trace_h5(path)
    assert loaded["H"].shape == H.shape
    assert np.allclose(loaded["H"], H)
    assert loaded["component_names"] == list(COMPONENT_NAMES)
    assert loaded["attrs"]["method"] == "nnls"
    assert list(loaded["trials"]["subject"]) == ["D0001"] * 4


def test_load_group_basis_from_canonical_csv():
    pytest.importorskip("pandas")
    from src.paths import nmf_assignments_path

    path = nmf_assignments_path()
    if not path.is_file():
        pytest.skip("canonical assignments missing")
    basis = load_group_basis(path)
    assert basis.W.shape[1] == 3
    assert basis.W.shape[0] == len(basis.channels)
    assert set(COMPONENT_NAMES) <= {
        c.removeprefix("loading_")
        for c in ("loading_sustain", "loading_motor", "loading_sensory")
    }
