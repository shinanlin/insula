#!/usr/bin/env python3
"""Plot functional-cluster HGA heatmap for NMF k=2 or k=3."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from mne_bids import BIDSPath
from tqdm.auto import tqdm

from src.nmf.waveform_pca import load_exclude_channels
from src.paths import (
    RESULTS_ROOT,
    hga_results_dir,
    img_dir,
    nmf_assignments_path,
    nmf_exclude_channels_path,
)


TASKS = [
    "PhonemeSequence",
    "LexicalDelay",
    "PictureNaming",
    "SentenceRep",
]
REF, ATLAS = "bipolar", "hammers"
PHASES = ["stimulus", "delay", "go", "response"]
DESCRIPTION = "Repeat"
MODALITY = "sound"
EXCLUDE_SUBJECTS = {"D0121"}

RED = "#A9373B"
GOLD = "#C4A35A"
BLUE = "#2369BD"

CLUSTER_META = {
    2: {
        "clusters": ["sustain", "sensory"],
        "colors": {
            "sustain": RED,
            "sensory": BLUE,
        },
        "labels": {
            "sustain": "sustain",
            "sensory": "sensory",
        },
    },
    3: {
        "clusters": ["sustain", "motor", "sensory"],
        "colors": {
            "sustain": RED,
            "motor": GOLD,
            "sensory": BLUE,
        },
        "labels": {
            "sustain": "sustain",
            "motor": "motor",
            "sensory": "sensory",
        },
    },
}

_FALLBACK_COLORS = (RED, GOLD, BLUE, "#6B8F71", "#8B6B9E", "#C47A3A")


def _meta_for_assignments(k: int, assignments: pd.DataFrame) -> dict:
    if k in CLUSTER_META:
        return CLUSTER_META[k]
    present = list(dict.fromkeys(assignments["functional_cluster"].astype(str)))
    # Prefer canonical extremes first when present.
    preferred = ["sustain", "sensory"]
    ordered = [c for c in preferred if c in present]
    ordered.extend(sorted(c for c in present if c not in ordered))
    colors = {
        name: _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)]
        for i, name in enumerate(ordered)
    }
    labels = {name: name.replace("_", " ") for name in ordered}
    return {"clusters": ordered, "colors": colors, "labels": labels}


def load_subset(
    k: int, *, assignments_path: Path | None = None
) -> tuple[pd.DataFrame, dict]:
    assign_path = (
        Path(assignments_path)
        if assignments_path is not None
        else nmf_assignments_path()
    )
    assignments = pd.read_csv(assign_path)
    drop = load_exclude_channels(nmf_exclude_channels_path())
    n_before = len(assignments)
    assignments = assignments[~assignments["channel"].astype(str).isin(drop)].copy()
    n_dropped = n_before - len(assignments)
    if n_dropped:
        print(f"Dropped {n_dropped} channels listed in exclude_channels.txt")
    meta = _meta_for_assignments(k, assignments)
    clusters = meta["clusters"]
    assign_map = assignments.set_index("channel")["functional_cluster"]
    keep_channels = set(assignments["channel"].astype(str))

    hga_paths: list[Path] = []
    for task in TASKS:
        hga_paths.extend(
            BIDSPath(
                root=str(hga_results_dir(task)),
                datatype="HGA",
                suffix="time",
                check=False,
            ).match()
        )

    frames = []
    for path in tqdm(hga_paths, desc="load HGA"):
        df = pd.read_csv(path)
        if "channel" not in df.columns:
            continue
        df = df[df["channel"].isin(keep_channels)]
        if not df.empty:
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No HGA rows for NMF channels under {RESULTS_ROOT}")

    hgas = pd.concat(frames, ignore_index=True)
    hgas.loc[hgas.phase == "Resp", "phase"] = "Response"
    hgas.loc[hgas.phase == "Audio", "phase"] = "Stimulus"
    hgas["phase"] = hgas["phase"].astype(str).str.lower()
    hgas["functional_cluster"] = hgas["channel"].map(assign_map)

    subset = hgas[
        (hgas.description == DESCRIPTION)
        & (hgas.modality == MODALITY)
        & hgas["functional_cluster"].isin(clusters)
        & ~hgas["subject"].isin(EXCLUDE_SUBJECTS)
    ].copy()
    print(subset.groupby("functional_cluster")["channel"].nunique())
    return subset, meta


def cluster_phase_channel_order(
    agg: pd.DataFrame, phase: str, cluster: str
) -> list[str]:
    phase_data = agg[(agg.phase == phase) & (agg.functional_cluster == cluster)]
    channels = sorted(phase_data["channel"].unique())
    first_sig = []
    for ch in channels:
        ch_data = phase_data[phase_data.channel == ch]
        sig_times = ch_data.loc[ch_data["mask"], "time"].to_numpy()
        first_sig.append((ch, float(sig_times.min()) if len(sig_times) else np.inf))
    return [ch for ch, _ in sorted(first_sig, key=lambda x: x[1])]


def plot_heatmap(subset: pd.DataFrame, meta: dict, out_path: Path) -> Path:
    clusters = meta["clusters"]
    labels = meta["labels"]
    cmaps = {
        name: sns.light_palette(color, as_cmap=True)
        for name, color in meta["colors"].items()
    }

    agg = (
        subset.groupby(
            ["channel", "time", "phase", "functional_cluster"], as_index=False
        )
        .agg(value=("value", "mean"), mask=("mask", "any"))
    )
    agg.loc[~agg["mask"], "value"] = np.nan

    n_by_cluster = {
        c: int(subset.loc[subset.functional_cluster == c, "channel"].nunique())
        for c in clusters
    }
    # Row height ∝ electrode count so larger clusters occupy more vertical space.
    height_ratios = [max(n_by_cluster.get(c, 0), 1) for c in clusters]
    total_channels = int(sum(height_ratios))
    cm = 1 / 2.54
    fontsize = 7

    fig, axes = plt.subplots(
        nrows=len(clusters),
        ncols=len(PHASES),
        figsize=(2.5 * len(PHASES) * cm, 0.03 * total_channels * cm),
        sharex="col",
        gridspec_kw={"height_ratios": height_ratios},
    )
    if len(clusters) == 1:
        axes = np.asarray([axes])

    for i, cluster in enumerate(clusters):
        cmap = cmaps[cluster]
        for j, phase in enumerate(PHASES):
            ax = axes[i, j]
            ordered = cluster_phase_channel_order(agg, phase, cluster)
            phase_data = agg[
                (agg.phase == phase)
                & (agg.functional_cluster == cluster)
                & agg.channel.isin(ordered)
            ]
            if not ordered:
                ax.text(
                    0.5,
                    0.5,
                    "No channels",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=fontsize,
                )
                ax.set_yticks([])
                continue

            pivot = (
                phase_data.pivot(index="channel", columns="time", values="value")
                .reindex(index=ordered)
            )
            da = xr.DataArray(
                pivot.to_numpy(dtype=float),
                coords={
                    "channel": pivot.index,
                    "time": pivot.columns.astype(float),
                },
                dims=["channel", "time"],
            )
            time_values = da.coords["time"].values.astype(float)
            step = np.diff(time_values).mean() if len(time_values) > 1 else 0.01
            time_edges = np.concatenate(
                (
                    [time_values[0] - step / 2],
                    time_values[:-1] + step / 2,
                    [time_values[-1] + step / 2],
                )
            )
            channel_edges = np.arange(len(ordered) + 1)
            t_mesh, c_mesh = np.meshgrid(time_edges, channel_edges)

            ax.pcolormesh(
                t_mesh,
                c_mesh,
                da.values,
                cmap=cmap,
                vmin=0,
                vmax=1,
                shading="auto",
                rasterized=True,
            )
            ax.axvline(x=0, color="k", linestyle="--", linewidth=0.5)
            ax.set_ylim(0, len(ordered))
            ax.set_yticks([])
            sns.despine(ax=ax, offset=0, trim=False)
            ax.spines["left"].set_visible(False)

            if i == len(clusters) - 1:
                ax.set_xlabel(f"{phase.capitalize()} (s)", fontsize=fontsize)
                ax.set_xticks(
                    [-0.5, 0, 0.5, 1.0] if phase != "response" else [-0.5, 0, 0.5]
                )
                ax.tick_params(labelsize=fontsize)
            else:
                ax.set_xlabel("")
                ax.set_xticks([])

            if j == 0:
                ax.set_ylabel(
                    f"{labels[cluster]}\n(n={len(ordered)})",
                    fontsize=fontsize,
                )
            if i == 0:
                ax.set_title(phase.capitalize(), fontsize=fontsize)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path.resolve()}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=3, choices=(2, 3, 4, 5, 6))
    parser.add_argument(
        "--assignments",
        type=Path,
        default=None,
        help="Optional channel_assignments.csv (default: results/nmf/channel_assignments.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: img/functional_hga/functional_hga_heatmap[_k3].svg",
    )
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    plt.rcParams["svg.fonttype"] = "none"
    args = parse_args()
    out = args.out
    if out is None:
        suffix = "" if args.k == 2 else f"_k{args.k}"
        out = img_dir("functional_hga") / f"functional_hga_heatmap{suffix}.svg"
    subset, meta = load_subset(args.k, assignments_path=args.assignments)
    plot_heatmap(subset, meta, out)


if __name__ == "__main__":
    main()
