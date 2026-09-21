#!/usr/bin/env python3
"""Collect the shrinkage-LDA time-resolved decoding grid into figures and a table.

Reads every ``(decode)(resolved)(lda){datatype}`` result written by
``run_decoding_resolved_lda.py`` and produces one SVG per description, with
phases as columns and pooled ROIs overlaid, plus a JSON summary of the
significant windows.

Figure style follows the canonical ``plot_phase_accuracy`` in
``notebooks/decode_functional.ipynb`` (see ``docs/PLOTTING_STYLE.md``), so these
panels drop in next to the production time-resolved figures: same ROI colours,
same 7 pt type, same stacked significance strips, same centimetre sizing.
"""

import rootutils

path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import argparse
import json
import logging
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter1d

from src.paths import RESULTS_ROOT, save_svg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

cm = 1 / 2.54
plt.rcParams["svg.fonttype"] = "none"
fontsize = 7

red = "#A9373B"
blue = "#2369BD"
gold = "#C4A35A"
stg_color = "#20B2AA"

ROI_COLOR_MAP = {
    "Sensory": blue,
    "Sustain": red,
    "Motor": gold,
    "STG": stg_color,
}
HUE_ORDER = ["Sensory", "Sustain", "Motor", "STG"]

CHANCE_AUC = 0.5
SMOOTH_SIGMA = 2
# Lexicality strip geometry, from phase_accuracy_bar_params in decode_functional.
BAR_PAD = 0.01
BAR_HEIGHT = 0.02
YLIM_PAD = 0.1


def roi_display_name(subject):
    """Match ``roi_hemi_from_subject``: anatomical pools drop the hemi suffix."""
    subject = str(subject)
    if subject.endswith(("l", "r")) and subject not in ROI_COLOR_MAP:
        return subject[:-1]
    return subject


def result_path(task, roi, datatype, phase, description, band):
    """Locate one result, tolerating the optional ``recording`` entity.

    PhonemeSequence inputs carry ``recording-1``, which the driver copies onto
    its output name; LexicalDelay outputs have no such entity. Returns ``None``
    when the cell has not been run.
    """
    folder = (
        RESULTS_ROOT / "decoding" / task / f"sub-{roi}"
        / f"(decode)(resolved)(lda){datatype}"
    )
    matches = sorted(
        folder.glob(f"sub-{roi}_proc-{phase}_*desc-{description}_{band}.h5")
    )
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Expected one result, found {[str(m) for m in matches]}")
    return matches[0]


def read_result(target, roi, phase, description):
    with h5py.File(target, "r") as stream:
        time = stream["time"][:]
        auc = stream["auc"][:]
        frame = pd.DataFrame({
            "time": time,
            "auc": gaussian_filter1d(auc, sigma=SMOOTH_SIGMA, mode="nearest"),
            "auc_raw": auc,
            "mask": np.asarray(stream["mask"][:]).astype(bool).ravel(),
            "p_value": stream["p_values"][:],
        })
        frame["roi"] = roi_display_name(roi)
        frame["phase"] = phase
        frame["description"] = description
        frame["n_channels"] = int(stream.attrs["n_channels"])
        frame["n_items"] = int(stream.attrs["n_items"])
        frame["n_perm"] = int(stream.attrs["n_perm"])
        # Lexicality results predate the multiclass path and carry neither key.
        frame["n_classes"] = int(stream.attrs.get("n_classes", 2))
        frame["scoring"] = str(stream.attrs.get("scoring", "roc_auc_pooled_oof"))
    return frame


def significant_windows(time_points, mask):
    """Contiguous significant stretches as (onset, offset) pairs."""
    if not mask.any():
        return []
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return [
        (round(float(time_points[start]), 3), round(float(time_points[stop - 1]), 3))
        for start, stop in zip(edges[::2], edges[1::2])
    ]


def plot_phase_auc(scores, phases, hue_order, description, xlim=(-0.5, 1.5)):
    """Time-resolved AUC panel per phase, in the canonical fig3 style."""
    roi_colors = [ROI_COLOR_MAP[roi] for roi in hue_order]
    values = scores["auc"].dropna()
    ymin = max(0.0, float(values.min()) - YLIM_PAD)
    ymax = float(values.max()) + YLIM_PAD
    y_top = ymax + BAR_PAD + len(hue_order) * (BAR_HEIGHT + 0.005) + 0.02

    n_phases = len(phases)
    legend_panel = n_phases - 1
    fig, axes = plt.subplots(1, n_phases, figsize=(2.5 * n_phases * cm, 3 * cm))
    if n_phases == 1:
        axes = [axes]
    plt.subplots_adjust(wspace=-0.1)

    for j, phase in enumerate(phases):
        ax = axes[j]
        subset = scores[scores.phase == phase]

        sns.lineplot(
            data=subset,
            x="time", y="auc", hue="roi",
            palette=roi_colors, hue_order=hue_order,
            lw=1, ax=ax, legend=(j == legend_panel),
        )
        ax.set_autoscale_on(False)
        ax.margins(y=0)

        for k, roi in enumerate(hue_order):
            roi_data = subset[subset.roi == roi].sort_values("time")
            if roi_data.empty or not roi_data["mask"].any():
                continue
            offset = k * (BAR_HEIGHT + 0.005)
            ax.fill_between(
                roi_data["time"].values,
                ymax + BAR_PAD + offset,
                ymax + BAR_PAD + BAR_HEIGHT + offset,
                where=roi_data["mask"].values,
                color=roi_colors[k],
                alpha=1,
                step="mid",
                linewidth=0,
            )

    for j, phase in enumerate(phases):
        ax = axes[j]
        ax.tick_params(labelsize=fontsize, width=0.5, length=2)
        plt.setp(ax.spines.values(), linewidth=0.75)
        ax.axvline(x=0, color="gray", linestyle="--", lw=0.5)
        ax.axhline(y=CHANCE_AUC, color="gray", linestyle="--", lw=0.5)
        ax.set_xlim(*xlim)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(phase.capitalize(), fontsize=fontsize)
        if j == 0:
            ax.set_ylabel("AUC", fontsize=fontsize)

    axes[n_phases // 2].set_xlabel("Time (s)", fontsize=fontsize)
    if axes[legend_panel].get_legend() is not None:
        # Two phases leave no free corner for the canonical in-panel legend, so
        # it sits outside the last axis instead of covering the Delay trace.
        axes[legend_panel].legend(
            fontsize=fontsize, frameon=False,
            loc="center left", bbox_to_anchor=(1.02, 0.5),
        )

    for ax in axes:
        ax.set_ylim(ymin, y_top)
        ax.set_ybound(ymin, y_top)

    # Ticks stop at the data ceiling; the band above ymax holds the significance
    # strips, and a tick up there would read as an AUC the traces never reach.
    # This has to precede despine, which trims the spine to the outermost tick.
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=3))
    ticks = axes[0].get_yticks()
    axes[0].set_yticks(ticks[(ticks >= ymin) & (ticks <= ymax)])

    for ax in axes[1:]:
        ax.set_yticks([])

    for ax in axes:
        sns.despine(ax=ax, trim=True, offset=0.1)

    for ax in axes[1:]:
        ax.spines["left"].set_visible(False)
        ax.set_ylabel("")

    axes[0].yaxis.set_tick_params(labelleft=True)
    axes[0].yaxis.set_visible(True)
    return fig, axes


def main(task, datatype, band, rois, phases, descriptions, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for description in descriptions:
        frames, missing = [], []
        for roi in rois:
            for phase in phases:
                target = result_path(task, roi, datatype, phase, description, band)
                if target is None:
                    missing.append(f"{roi}/{phase}")
                    continue
                frames.append(read_result(target, roi, phase, description))

        if missing:
            logger.warning("%s: missing %s", description, ", ".join(missing))
        if not frames:
            continue

        scores = pd.concat(frames, ignore_index=True)
        hue_order = [r for r in HUE_ORDER if r in set(scores["roi"])]

        fig, _ = plot_phase_auc(scores, phases, hue_order, description)
        svg = save_svg(fig, out_dir / f"resolved_lda_{task}_{datatype}_{description}",
                       close=True)
        logger.info("Wrote %s", svg)

        for (roi, phase), group in scores.groupby(["roi", "phase"], sort=False):
            group = group.sort_values("time")
            windows = significant_windows(
                group["time"].values, group["mask"].values.astype(bool)
            )
            raw = group["auc_raw"].values
            entry = {
                "n_channels": int(group["n_channels"].iloc[0]),
                "n_items": int(group["n_items"].iloc[0]),
                "n_perm": int(group["n_perm"].iloc[0]),
                "n_classes": int(group["n_classes"].iloc[0]),
                "scoring": str(group["scoring"].iloc[0]),
                "peak_auc": round(float(raw.max()), 4),
                "peak_time": round(float(group["time"].values[int(raw.argmax())]), 3),
                "n_significant": int(group["mask"].sum()),
                "significant_windows": windows,
                "first_significant": windows[0][0] if windows else None,
            }
            summary[f"{roi}_{phase}_{description}"] = entry
            logger.info(
                "%-8s %-9s %-9s %3d ch  peak AUC %.3f at %+.2fs, %2d sig windows %s",
                roi, phase, description, entry["n_channels"],
                entry["peak_auc"], entry["peak_time"], entry["n_significant"],
                windows or "-",
            )

    summary_path = out_dir / f"resolved_lda_{datatype}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", summary_path)
    return summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="LexicalDelay")
    parser.add_argument("--datatype", type=str, default="lexicality")
    parser.add_argument("--band", type=str, default="highgamma")
    parser.add_argument("--rois", nargs="+",
                        default=["Sensory", "Sustain", "Motor", "STGl"])
    parser.add_argument("--phases", nargs="+", default=["Stimulus", "Delay"])
    parser.add_argument("--descriptions", nargs="+", default=["Repeat", "Decision"])
    parser.add_argument("--out_dir", type=str, default=None)

    args = parser.parse_args()
    out_dir = (
        RESULTS_ROOT / "decoding" / args.task / "(summary)(resolved)(lda)"
        if args.out_dir is None
        else Path(args.out_dir)
    )
    main(
        args.task, args.datatype, args.band,
        args.rois, args.phases, args.descriptions, out_dir,
    )
