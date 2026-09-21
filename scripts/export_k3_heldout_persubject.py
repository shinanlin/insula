#!/usr/bin/env python3
"""Export per-subject held-out cluster waveforms for k=3 third-cluster test.

Reuses src.nmf machinery EXACTLY (same load_hga_rows, same restrict_windows,
same subject-mean aggregation as summarize_held_out) so the third-cluster test
uses an identical standard to the existing sustained/sensory contrasts.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, ".")
from src.nmf.waveform_analysis import (
    TASKS, discover_paths, load_hga_rows, restrict_windows, PHASE_WINDOWS,
)
from src.paths import RESULTS_ROOT

OUT = Path("results/nmf/k3/heldout_persubject.csv")
assign = pd.read_csv("results/nmf/k3/channel_assignments.csv")
print("assignments:", assign.shape, assign["functional_cluster"].value_counts().to_dict(), flush=True)

held_phases = {"delay", "go", "response"}
paths = discover_paths(RESULTS_ROOT, tuple(TASKS))
chans = set(assign["channel"])
rows = load_hga_rows(paths, phases=held_phases,
                     exclude_subjects={"D0121"}, channels=chans)
print("held-out rows:", rows.shape, flush=True)

# subject-mean per cluster/phase/time  -- identical to summarize_held_out internals
values = (
    restrict_windows(rows)
    .groupby(["channel", "phase", "time"], as_index=False)["value"].mean()
    .merge(assign[["channel", "subject", "functional_cluster"]],
           on="channel", how="inner", validate="many_to_one")
)
subject_means = (
    values.groupby(["subject", "functional_cluster", "phase", "time"],
                   as_index=False)["value"].mean()
)
subject_means.to_csv(OUT, index=False)
print("wrote", OUT, subject_means.shape, flush=True)
print(subject_means.groupby(["functional_cluster","phase"])["subject"].nunique())
