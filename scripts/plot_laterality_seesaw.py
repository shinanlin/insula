#!/usr/bin/env python3
"""Sampling-compensated laterality seesaw from home-phase HGA.

Reads results/nmf/laterality_home_phase_electrodes.csv (no HGA reload).

Index: subject-equalized LI = (μ_L − μ_R) / (μ_L + μ_R)
  1. mean HGA within subject × hemisphere
  2. unweighted mean across subjects in each hemisphere
  3. classic laterality index on those two means

This stops extra left contacts (and multi-contact subjects) from pulling
the index left via electrode counts. Bootstrap CIs resample subjects.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.nmf.waveform_analysis import FUNCTION_COLORS
from src.paths import img_dir, nmf_results_dir, save_svg

cm = 1 / 2.54
fontsize = 7
HOME_PHASE = {
    "sensory": "Stimulus",
    "sustain": "Delay",
    "motor": "Response",
}
PHASE_CLUSTER_ORDER = ("sensory", "sustain", "motor")
N_BOOT = 5000
SEED = 0


def style_ax(ax, *, trim=True):
    ax.tick_params(labelsize=fontsize, width=0.75, length=2, which="both")
    plt.setp(ax.spines.values(), linewidth=0.75)
    sns.despine(ax=ax, offset=1, trim=trim)


def laterality_index(mu_l: float, mu_r: float) -> float:
    denom = mu_l + mu_r
    if abs(denom) < 1e-12:
        return float("nan")
    return (mu_l - mu_r) / denom


def subject_means(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    subj = df.groupby(["subject", "hemi"], as_index=False)["HGA"].mean()
    left = subj.loc[subj["hemi"] == "L", "HGA"].to_numpy(float)
    right = subj.loc[subj["hemi"] == "R", "HGA"].to_numpy(float)
    return left, right


def bootstrap_li(left: np.ndarray, right: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    boots = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        boots[i] = laterality_index(
            rng.choice(left, size=left.size, replace=True).mean(),
            rng.choice(right, size=right.size, replace=True).mean(),
        )
    return boots


def summarize(matched: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for cluster in PHASE_CLUSTER_ORDER:
        sub = matched[matched["home_cluster"] == cluster]
        left, right = subject_means(sub)
        li = laterality_index(left.mean(), right.mean())
        boots = bootstrap_li(left, right, rng)
        p = float(2 * min((boots > 0).mean(), (boots < 0).mean()))
        rows.append(
            {
                "cluster": cluster,
                "phase": HOME_PHASE[cluster].lower(),
                "n_elec_L": int((sub["hemi"] == "L").sum()),
                "n_elec_R": int((sub["hemi"] == "R").sum()),
                "n_subj_L": int(left.size),
                "n_subj_R": int(right.size),
                "mean_L": float(left.mean()),
                "mean_R": float(right.mean()),
                "LI": float(li),
                "ci_lo": float(np.quantile(boots, 0.025)),
                "ci_hi": float(np.quantile(boots, 0.975)),
                "boot_p": p,
            }
        )
    return pd.DataFrame(rows)


def plot_seesaw(stats: pd.DataFrame, out_path: Path, *, close: bool = True) -> Path:
    """Horizontal seesaw: x = LI (left-dominant on the left), y = component."""
    plt.rcParams["svg.fonttype"] = "none"
    ordered = stats.set_index("cluster").loc[list(PHASE_CLUSTER_ORDER)]
    n = len(PHASE_CLUSTER_ORDER)
    ys = np.arange(n - 1, -1, -1, dtype=float)

    fig, ax = plt.subplots(figsize=(8.5 * cm, 5.2 * cm))
    ax.axvline(0.0, color="0.35", lw=0.9, zorder=1)
    x_abs = max(
        0.45,
        float(np.nanmax(np.abs(ordered[["LI", "ci_lo", "ci_hi"]].to_numpy()))) + 0.08,
    )
    for y, cluster in zip(ys, PHASE_CLUSTER_ORDER):
        row = ordered.loc[cluster]
        li = float(row["LI"])
        lo = float(row["ci_lo"])
        hi = float(row["ci_hi"])
        color = FUNCTION_COLORS[cluster]
        ax.plot([0.0, li], [y, y], color=color, lw=1.0, zorder=2)
        ax.errorbar(
            li,
            y,
            xerr=[[li - lo], [hi - li]],
            fmt="none",
            ecolor=color,
            elinewidth=0.75,
            capsize=2.0,
            capthick=0.75,
            zorder=3,
        )
        ax.scatter(
            [li],
            [y],
            s=22,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            zorder=4,
        )

    ax.set_yticks(ys)
    ax.set_yticklabels(
        [f"{c}\n{HOME_PHASE[c]}" for c in PHASE_CLUSTER_ORDER],
        fontsize=fontsize,
    )
    ax.set_ylim(-0.55, n - 0.45)
    ax.set_xlim(-x_abs, x_abs)
    ax.invert_xaxis()
    ax.set_xlabel("(L − R) / (L + R)", fontsize=fontsize)
    ax.set_ylabel("")
    ax.text(
        0.0,
        1.02,
        "Left",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        color="0.25",
    )
    ax.text(
        1.0,
        1.02,
        "Right",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=fontsize,
        color="0.25",
    )
    ax.set_title("Home-phase laterality", fontsize=fontsize)
    style_ax(ax, trim=False)
    return save_svg(fig, out_path, close=close)


def main() -> None:
    elec_path = nmf_results_dir() / "laterality_home_phase_electrodes.csv"
    matched = pd.read_csv(elec_path)
    matched = matched[matched["cluster_label"] == matched["home_cluster"]].copy()
    stats = summarize(matched)
    print(stats.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    stats_path = nmf_results_dir() / "laterality_home_phase_seesaw.csv"
    stats.to_csv(stats_path, index=False)
    img = img_dir("nmf")
    out_svg = plot_seesaw(stats, img / "laterality_home_phase_hga.svg")
    plot_seesaw(stats, img / "laterality_home_phase_seesaw.svg")
    print("Wrote", stats_path)
    print("Wrote", out_svg)


if __name__ == "__main__":
    main()
