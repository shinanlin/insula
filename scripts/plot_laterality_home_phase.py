#!/usr/bin/env python3
"""Phase-matched laterality: home-window Repeat HGA by cluster and hemisphere.

sensory ← stimulus 0–0.5 s
sustain ← delay 0–1.0 s (NMF delay length; also prints delay 0–0.5)
motor   ← response 0–0.5 s

Hemisphere comes from channel_assignments.csv, not HGA x-sign.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from plot_functional_hga_heatmap import load_subset
from src.nmf.waveform_analysis import FUNCTION_COLORS
from src.paths import img_dir, nmf_assignments_path, nmf_results_dir, save_svg

NMF_K = 3
cm = 1 / 2.54
fontsize = 7
HEMI_COLORS = {"L": "#333333", "R": "#B0B0B0"}
HEMI_LABELS = {"L": "Left", "R": "Right"}
HOME_PHASE = {
    "sensory": ("stimulus", 0.0, 0.5),
    "sustain": ("delay", 0.0, 1.0),
    "motor": ("response", 0.0, 0.5),
}
PHASE_CLUSTER_ORDER = ("sensory", "sustain", "motor")
ALT_WINDOWS = {
    "sustain": ("delay", 0.0, 0.5),
}


def style_ax(ax, *, trim=True):
    ax.tick_params(labelsize=fontsize, width=0.75, length=2, which="both")
    plt.setp(ax.spines.values(), linewidth=0.75)
    sns.despine(ax=ax, offset=1, trim=trim)


def window_means(hga: pd.DataFrame, phase: str, t0: float, t1: float) -> pd.DataFrame:
    win = hga[(hga["phase"] == phase) & (hga["time"] > t0) & (hga["time"] < t1)]
    return (
        win.groupby(["channel", "hemi", "cluster_label", "subject"], as_index=False)
        .agg(HGA=("value", "mean"))
        .assign(home_phase=phase, t0=t0, t1=t1)
    )


def report(label: str, left: pd.Series, right: pd.Series) -> dict:
    li = (left.mean() - right.mean()) / (left.mean() + right.mean())
    _, p = mannwhitneyu(left, right, alternative="two-sided")
    row = {
        "contrast": label,
        "n_L": int(left.size),
        "n_R": int(right.size),
        "mean_L": float(left.mean()),
        "mean_R": float(right.mean()),
        "LI": float(li),
        "mw_p": float(p),
    }
    print(
        f"  {label:28s}  n={left.size}/{right.size}  "
        f"mean L/R={left.mean():.3f}/{right.mean():.3f}  "
        f"LI={li:+.3f}  MW p={p:.3f}"
    )
    return row


def main() -> None:
    plt.rcParams["svg.fonttype"] = "none"
    assign = pd.read_csv(nmf_assignments_path())
    hemi_map = assign.set_index("channel")["hemi"].astype(str).str.upper().str[0]

    hga, _ = load_subset(NMF_K)
    hga["cluster_label"] = hga["functional_cluster"].astype(str)
    hga["hemi"] = hga["channel"].map(hemi_map)
    hga = hga[hga["hemi"].isin(["L", "R"])].copy()

    rows = []
    stats = []
    print("phase-matched window-mean HGA (that cluster only)")
    for cluster in PHASE_CLUSTER_ORDER:
        phase, t0, t1 = HOME_PHASE[cluster]
        elec = window_means(hga, phase, t0, t1)
        elec["home_cluster"] = cluster
        rows.append(elec)
        matched = elec[elec["cluster_label"] == cluster]
        left = matched.loc[matched["hemi"] == "L", "HGA"]
        right = matched.loc[matched["hemi"] == "R", "HGA"]
        stats.append(
            report(f"{cluster} {phase} ({t0:g},{t1:g})", left, right)
            | {"cluster": cluster, "phase": phase, "t0": t0, "t1": t1, "which": "matched"}
        )

    print("\nsame window, other clusters (control)")
    home_hga = pd.concat(rows, ignore_index=True)
    for cluster in PHASE_CLUSTER_ORDER:
        phase, t0, t1 = HOME_PHASE[cluster]
        sub = home_hga[
            (home_hga["home_cluster"] == cluster)
            & (home_hga["cluster_label"] != cluster)
        ]
        left = sub.loc[sub["hemi"] == "L", "HGA"]
        right = sub.loc[sub["hemi"] == "R", "HGA"]
        stats.append(
            report(f"not-{cluster} {phase}", left, right)
            | {"cluster": f"not-{cluster}", "phase": phase, "t0": t0, "t1": t1, "which": "control"}
        )

    print("\nalternate delay window (spatial-plot match)")
    phase, t0, t1 = ALT_WINDOWS["sustain"]
    elec = window_means(hga, phase, t0, t1)
    matched = elec[elec["cluster_label"] == "sustain"]
    left = matched.loc[matched["hemi"] == "L", "HGA"]
    right = matched.loc[matched["hemi"] == "R", "HGA"]
    stats.append(
        report(f"sustain {phase} ({t0:g},{t1:g})", left, right)
        | {"cluster": "sustain", "phase": phase, "t0": t0, "t1": t1, "which": "alt-delay"}
    )

    matched = home_hga[home_hga["cluster_label"] == home_hga["home_cluster"]].copy()
    fig, ax = plt.subplots(figsize=(8.5 * cm, 5.2 * cm))
    sns.violinplot(
        data=matched,
        x="home_cluster",
        y="HGA",
        hue="hemi",
        order=list(PHASE_CLUSTER_ORDER),
        hue_order=["L", "R"],
        split=True,
        inner=None,
        palette=HEMI_COLORS,
        cut=0,
        linewidth=0.5,
        ax=ax,
        legend=False,
    )
    x_index = {name: i for i, name in enumerate(PHASE_CLUSTER_ORDER)}
    dx = {"L": -0.06, "R": 0.06}
    for cluster in PHASE_CLUSTER_ORDER:
        sub = matched[matched["home_cluster"] == cluster]
        for hemi in ("L", "R"):
            vals = sub.loc[sub["hemi"] == hemi, "HGA"]
            ax.scatter(
                [x_index[cluster] + dx[hemi]],
                [vals.mean()],
                s=14,
                marker="o",
                color=FUNCTION_COLORS[cluster],
                edgecolor="white",
                linewidth=0.3,
                zorder=3,
            )
    ax.set_xticks(range(len(PHASE_CLUSTER_ORDER)))
    ax.set_xticklabels(
        [f"{c}\n{HOME_PHASE[c][0].capitalize()}" for c in PHASE_CLUSTER_ORDER],
        fontsize=fontsize,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Window-mean HGA (z)", fontsize=fontsize)
    ax.set_title("Home-phase expression", fontsize=fontsize)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor=HEMI_COLORS[h],
                markersize=5,
                label=HEMI_LABELS[h],
            )
            for h in ("L", "R")
        ],
        frameon=False,
        fontsize=fontsize,
    )
    style_ax(ax, trim=False)

    img = img_dir("nmf")
    out_svg = save_svg(fig, img / "laterality_home_phase_hga.svg", close=True)
    stats_path = nmf_results_dir() / "laterality_home_phase_hga.csv"
    pd.DataFrame(stats).to_csv(stats_path, index=False)
    elec_path = nmf_results_dir() / "laterality_home_phase_electrodes.csv"
    matched.to_csv(elec_path, index=False)
    print("Wrote", out_svg)
    print("Wrote", stats_path)
    print("Wrote", elec_path)


if __name__ == "__main__":
    main()
