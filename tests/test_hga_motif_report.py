from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_hga_motif_decision import (
    _decision,
    _lagged_concordance,
    _plot_anatomy,
    _plot_lag_direction,
    _plot_matched_cross,
    _plot_method_concordance,
    _plot_motif_cells,
)


MOTIFS = ("sensory", "sustain", "motor")
PHASES = ("Stimulus", "Delay", "Go", "Response")
METHODS = ("xcorr", "xcorr_resid", "oaec")


def _group_cells() -> pd.DataFrame:
    rows = []
    for phase in PHASES:
        for source in MOTIFS:
            for target in MOTIFS:
                row = {
                    "phase": phase,
                    "source_motif": source,
                    "partner_motif": target,
                    "weighted_lag_s_mean": -0.03 if source == target else 0.01,
                    "weighted_lag_s_ci_low": -0.05 if source == target else -0.01,
                    "weighted_lag_s_ci_high": -0.01 if source == target else 0.03,
                }
                for method in METHODS:
                    row[f"{method}_fdr_hit_rate_mean"] = 0.1 + 0.02 * (source == target)
                    row[f"{method}_fwer_hit_rate_mean"] = 0.04
                rows.append(row)
    return pd.DataFrame(rows)


def _group_contrasts() -> pd.DataFrame:
    rows = []
    for phase in PHASES:
        row = {"phase": phase}
        for method in METHODS:
            for suffix in ("fdr_hit_rate", "effect"):
                stem = f"{method}_matched_minus_cross_{suffix}"
                row[f"{stem}_mean"] = 0.02
                row[f"{stem}_ci_low"] = 0.01
                row[f"{stem}_ci_high"] = 0.03
                row[f"{stem}_q_fdr"] = 0.01
        rows.append(row)
    return pd.DataFrame(rows)


def test_report_plotters_and_decision_smoke(tmp_path: Path):
    cells = _group_cells()
    contrasts = _group_contrasts()
    pairs = pd.DataFrame(
        {
            "phase": list(PHASES) * 2,
            "xcorr_lag_direction": ["insula_leads"] * 4 + ["partner_leads"] * 4,
        }
    )
    roi_rows = []
    for phase in PHASES:
        for roi_index in range(3):
            row = {"phase": phase, "target_roi": f"ROI{roi_index}", "n_pairs": 10}
            for method in METHODS:
                row[f"{method}_sig_fdr_mean"] = 0.1
            roi_rows.append(row)
    concordance = pd.DataFrame(
        [
            {
                "phase": phase, "method_a": left, "method_b": right,
                "fdr_jaccard": 0.3, "effect_spearman": 0.5,
            }
            for phase in PHASES
            for left, right in (("xcorr", "xcorr_resid"), ("xcorr", "oaec"))
        ]
    )
    _plot_motif_cells(cells, tmp_path)
    _plot_matched_cross(contrasts, tmp_path)
    _plot_lag_direction(cells, pairs, tmp_path)
    _plot_anatomy(pd.DataFrame(roi_rows), tmp_path)
    _plot_method_concordance(concordance, tmp_path)
    expected = (
        "main_phase_resolved_motif_coupling", "main_matched_vs_cross",
        "main_lag_direction", "main_anatomical_distribution",
        "main_method_concordance",
    )
    for stem in expected:
        assert (tmp_path / f"{stem}.png").exists()
        assert (tmp_path / f"{stem}.svg").exists()
    assert _decision(contrasts)[0] == (
        "preferentially motif-concordant / parallel embedding"
    )
    unsupported = pd.DataFrame(
        {
            "phase": ["Delay", "Delay"],
            "outcome": ["xcorr_effect", "xcorr_resid_effect"],
            "stat": [0.01, 0.01], "q_fdr": [0.5, 0.5],
        }
    )
    assert _decision(contrasts, unsupported)[0] == (
        "evidence insufficient due to coverage or instability"
    )
    supported = unsupported.assign(q_fdr=0.01)
    assert _decision(contrasts, supported)[0] == (
        "preferentially motif-concordant / parallel embedding"
    )


def test_lagged_concordance_is_subject_and_task_balanced():
    rows = []
    for subject in ("S1", "S2"):
        for task in ("A", "B"):
            for lag in (-0.08, -0.04, -0.02):
                rows.append(
                    {
                        "subject": subject, "task": task, "phase": "Response",
                        "peak_lag_s": lag, "consensus_lag_s": lag,
                        "xcorr_lag_direction": "insula_leads",
                    }
                )
    pair, group = _lagged_concordance(
        pd.DataFrame(rows), n_resamples=100, seed=3
    )
    assert pair["direction_agree"].all()
    assert group.loc[0, "direction_agreement_mean"] == 1.0
    assert group.loc[0, "lag_error_s_mean"] == 0.0
    assert np.isclose(group.loc[0, "lag_correlation_mean"], 1.0)
