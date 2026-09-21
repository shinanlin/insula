#!/usr/bin/env python3
"""Export the CANONICAL concat-NMF input matrix: crop_postonset (matches published k=3)."""
from __future__ import annotations
import sys; sys.path.insert(0, ".")
import json, os
import numpy as np
from src.nmf.waveform_analysis import (
    TASKS, PHASE_WINDOWS_POSTONSET, discover_paths, load_hga_rows, restrict_windows,
    concatenated_phase_matrix, prepare_shape_matrix,
)
from src.paths import RESULTS_ROOT

phases = tuple(PHASE_WINDOWS_POSTONSET)  # stimulus, delay, go, response
all_paths = discover_paths(RESULTS_ROOT, tuple(TASKS))
rows = load_hga_rows(all_paths, phases=set(phases), exclude_subjects={"D0121"})
rows = restrict_windows(rows, PHASE_WINDOWS_POSTONSET)   # <-- canonical crop
concat, meta, phase_slices = concatenated_phase_matrix(rows, phases=phases)
Xc, keep = prepare_shape_matrix(concat.to_numpy())
concat = concat.iloc[np.flatnonzero(keep)]; meta = meta.iloc[np.flatnonzero(keep)]

os.makedirs("results/nmf/rankdiag", exist_ok=True)
np.save("results/nmf/rankdiag/Xconcat_postonset.npy", Xc)
meta.reset_index().to_csv("results/nmf/rankdiag/meta_concat_postonset.csv", index=False)
sl = {p: [int(s.start), int(s.stop)] for p, s in phase_slices.items()}
json.dump({"phases": list(phases), "slices": sl, "n_feat": int(Xc.shape[1]), "crop": "postonset"},
          open("results/nmf/rankdiag/concat_phase_slices_postonset.json", "w"))
print("Xconcat_postonset", Xc.shape, "| slices", sl, "| meta", meta.shape, flush=True)
