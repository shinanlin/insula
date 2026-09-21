"""Unit tests for fixed-H whole-brain motif projection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.nmf.whole_brain_projection import (
    build_projection_table,
    project_fixed_temporal_basis,
)


def test_project_fixed_temporal_basis_recovers_known_loadings():
    rng = np.random.default_rng(0)
    k, n_times, n_elec = 3, 40, 8
    H = rng.random((k, n_times))
    H = H / np.linalg.norm(H, axis=1, keepdims=True)
    weights_true = rng.random((n_elec, k))
    X = np.clip(weights_true @ H, 0.0, None)
    weights_hat, reconstructed = project_fixed_temporal_basis(X, H)
    assert weights_hat.shape == (n_elec, k)
    assert reconstructed.shape == X.shape
    assert np.allclose(weights_hat, weights_true, atol=1e-5, rtol=1e-4)


def test_build_projection_table_marks_unmatched():
    rng = np.random.default_rng(1)
    names = ("sustain", "motor", "sensory")
    n_elec, n_times = 5, 12
    metadata = pd.DataFrame(
        {"roi": ["STG"] * n_elec, "subject": ["D0001"] * n_elec},
        index=[f"ch{i}" for i in range(n_elec)],
    )
    metadata.index.name = "channel"
    X = rng.random((n_elec, n_times))
    weights = np.zeros((n_elec, 3))
    weights[:, 0] = 1.0
    reconstructed = np.zeros_like(X)
    reconstructed[0] = X[0]
    table = build_projection_table(
        metadata,
        X,
        weights,
        reconstructed,
        names,
        pilot_min_explained_energy=0.5,
    )
    assert "loading_sustain" in table.columns
    assert table.loc[0, "pilot_class"] == "sustain"
    assert (table.loc[1:, "pilot_class"] == "unmatched").all()
