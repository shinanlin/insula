"""One insula electrode, available alignments. SVGs under img/tfr/{task}/."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

from src.paths import save_svg

DEFAULT_PHASES = ("Stimulus", "Delay", "Go", "Response")
XLIM = (-0.5, 1.0)
XTICKS = (-0.5, 0.0, 1.0)
YLIM = (4, 250)
P_THRESHOLD = 0.05


def plot_channel_phases(
    panels,
    dest,
    *,
    subject,
    channel,
    task,
    condition="Repeat",
    cluster="",
    p_threshold=P_THRESHOLD,
    xlim=XLIM,
    phases=None,
):
    """Spectrograms for one bipolar pair; columns follow ``phases``."""
    phases = tuple(phases) if phases is not None else tuple(
        phase for phase in DEFAULT_PHASES if phase in panels
    )
    if not phases:
        raise ValueError("no phases to plot")
    cm = 1 / 2.54
    fontsize = 7
    plt.rcParams["svg.fonttype"] = "none"
    stacked = np.concatenate([np.asarray(panels[phase]["data"]).ravel() for phase in phases])
    vmin, vmax = np.nanpercentile(stacked, [5, 95])
    limit = max(abs(float(vmin)), abs(float(vmax)))
    if not np.isfinite(limit) or limit == 0:
        limit = 1.0
    vmin, vmax = -limit, limit
    cmap = sns.color_palette("vlag", as_cmap=True)
    fig = plt.figure(figsize=(max(4, 4 * len(phases)) * cm, 4.2 * cm))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=len(phases),
        wspace=0.28,
        left=0.10,
        right=0.86,
        bottom=0.22,
        top=0.72,
    )
    axes = [fig.add_subplot(gs[0, j]) for j in range(len(phases))]
    mesh = None
    for ax, phase in zip(axes, phases):
        panel = panels[phase]
        times = np.asarray(panel["times"], float)
        freqs = np.asarray(panel["freqs"], float)
        data = np.asarray(panel["data"], float)
        mask = np.asarray(panel["mask"], float)
        pvals = np.asarray(panel["pvals"], float)
        mesh = ax.pcolormesh(
            times,
            freqs,
            data,
            shading="gouraud",
            cmap=cmap,
            rasterized=True,
            vmin=vmin,
            vmax=vmax,
        )
        significant = (pvals < p_threshold) & (mask > 0)
        if np.any(significant):
            sig = np.zeros_like(mask)
            sig[significant] = 1
            ax.contour(
                times,
                freqs,
                sig,
                levels=[0.8],
                colors="gray",
                linewidths=0.75,
                linestyles="-",
            )
        ax.axvline(0, color="gray", linestyle="-.", linewidth=0.5)
        ax.set_xlim(*xlim)
        ax.set_ylim(*YLIM)
        ax.set_xticks(XTICKS)
        ax.set_title(phase, fontsize=fontsize, pad=3)
        ax.tick_params(labelsize=fontsize, width=0.75, length=2, pad=1.5)
        plt.setp(ax.spines.values(), linewidth=0.75)
        ax.set_xlabel("")
        ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
        if ax is axes[0]:
            ax.set_ylabel("Frequency (Hz)", fontsize=fontsize)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False, length=0)
    cbar = fig.colorbar(mesh, ax=axes, fraction=0.03, pad=0.03)
    cbar.set_label("dB", fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize, width=0.5, length=2)
    cbar.outline.set_linewidth(0.75)
    label = channel.replace(f"{subject}_", "", 1)
    if cluster:
        label = f"{label} · {cluster}"
    fig.suptitle(f"{subject} {label} · {task} {condition}", fontsize=fontsize, y=0.98)
    fig.text(0.48, 0.04, "Time (s)", ha="center", va="top", fontsize=fontsize)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_svg(fig, dest, dpi=300, close=True)
    return dest
