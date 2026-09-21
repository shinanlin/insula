#!/usr/bin/env python3
"""Export the exact stimulus NMF input matrix X (electrode x time)."""
from __future__ import annotations
import sys; sys.path.insert(0, ".")
import numpy as np
from src.nmf.waveform_analysis import (
    TASKS, discover_paths, load_hga_rows, restrict_windows,
    stimulus_matrix, prepare_shape_matrix,
)
from src.paths import RESULTS_ROOT
import os

all_paths = discover_paths(RESULTS_ROOT, tuple(TASKS))
rows = load_hga_rows(all_paths, phases={"stimulus"}, exclude_subjects={"D0121"})
rows = restrict_windows(rows)
raw, meta = stimulus_matrix(rows)
X, keep = prepare_shape_matrix(raw.to_numpy())
raw = raw.iloc[np.flatnonzero(keep)]; meta = meta.iloc[np.flatnonzero(keep)]
times = raw.columns.to_numpy(float)
os.makedirs("results/nmf/rankdiag", exist_ok=True)
np.save("results/nmf/rankdiag/X.npy", X)
np.save("results/nmf/rankdiag/times.npy", times)
meta.reset_index().to_csv("results/nmf/rankdiag/meta.csv", index=False)
print("X shape", X.shape, "times", times.shape, "meta", meta.shape, flush=True)
