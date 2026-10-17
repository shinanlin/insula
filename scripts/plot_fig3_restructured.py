#!/usr/bin/env python3
"""Build the restructured Figure 3 requested in the Week 82 review.

The source notebook remains untouched.  This script reuses its loaders, writes
epoch-specific panels, and assembles a self-contained SVG with three parallel
sections: input, delay, and output.
"""

from __future__ import annotations

import base64
import html
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from mne_bids import BIDSPath


REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "vizpub" / "fig3.ipynb"
FIG3_DIR = REPO / "img" / "fig3"
PUB_DIR = REPO / "img" / "pub"
FIG3_DIR.mkdir(parents=True, exist_ok=True)
PUB_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("PYVISTA_USE_PANEL", "false")
os.environ.setdefault("MNE_3D_BACKEND", "notebook")
os.environ.setdefault("MPLBACKEND", "Agg")

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def execute_notebook_cells(cell_ids: tuple[str, ...]) -> dict:
    """Execute selected, dependency-ordered notebook cells in one namespace."""
    notebook = json.loads(NOTEBOOK.read_text())
    cells = {cell.get("id"): cell for cell in notebook["cells"]}
    missing = [cell_id for cell_id in cell_ids if cell_id not in cells]
    if missing:
        raise RuntimeError(f"Missing Figure 3 notebook cells: {missing}")

    namespace = {"__name__": "__main__"}
    old_cwd = Path.cwd()
    os.chdir(REPO / "vizpub")
    try:
        for cell_id in cell_ids:
            print(f"=== exec {cell_id} ===", flush=True)
            source = "".join(cells[cell_id]["source"])
            source = "\n".join(
                line for line in source.splitlines() if not line.startswith("%")
            )
            exec(compile(source, f"{NOTEBOOK.name}:{cell_id}", "exec"), namespace)

            # Only the two tasks used in the restructured figure are needed.
            if cell_id == "fig3-helpers":
                namespace["TASKS"] = ["LexicalDelay", "PictureNaming"]
                namespace["CONDITION_TASKS"] = {
                    "Passive": ["PictureNaming"],
                    "Repeat": ["LexicalDelay", "PictureNaming"],
                    "Decision": ["LexicalDelay"],
                }
    finally:
        os.chdir(old_cwd)
    return namespace


def add_packaged_mask(
    df: pd.DataFrame,
    *,
    task: str,
    hga_results_dir,
) -> pd.DataFrame:
    """Attach the packaged time-resolved significance mask to HGA traces."""
    if df.empty:
        return df.assign(mask=False)

    wanted_subjects = set(df["subject"].astype(str).str.removeprefix("sub-"))
    wanted_descriptions = set(df["description"].astype(str))
    wanted_modalities = set(df["modality"].astype(str))
    wanted_phases = set(df["phase"].astype(str).str.lower())
    paths = BIDSPath(
        root=str(hga_results_dir(task)),
        datatype="HGA",
        suffix="time",
        check=False,
    ).match()

    frames: list[pd.DataFrame] = []
    for path in paths:
        subject = str(path.subject).removeprefix("sub-")
        description = str(path.description)
        modality = path.recording if path.recording is not None else "sound"
        phase = str(path.processing).lower()
        if phase == "resp":
            phase = "response"
        elif phase == "audio":
            phase = "stimulus"
        if (
            subject not in wanted_subjects
            or description not in wanted_descriptions
            or modality not in wanted_modalities
            or phase not in wanted_phases
        ):
            continue
        frame = pd.read_csv(
            path,
            usecols=[
                "channel",
                "time",
                "mask",
                "subject",
                "phase",
                "modality",
                "description",
            ],
        )
        if frame.empty:
            continue
        frame["subject"] = frame["subject"].astype(str).str.removeprefix("sub-")
        frame["phase"] = frame["phase"].astype(str).str.lower().replace(
            {"resp": "response", "audio": "stimulus"}
        )
        frame["time"] = np.round(frame["time"].to_numpy(dtype=float), 4)
        frames.append(frame)

    out = df.drop(columns=["mask"], errors="ignore").copy()
    out["subject"] = out["subject"].astype(str).str.removeprefix("sub-")
    out["time"] = np.round(out["time"].to_numpy(dtype=float), 4)
    if not frames:
        print(f"WARNING: no packaged masks found for {task}; showing raw HGA")
        out["mask"] = True
        return out

    mask_df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=[
            "subject",
            "channel",
            "time",
            "phase",
            "modality",
            "description",
        ],
        keep="first",
    )
    out = out.merge(
        mask_df,
        on=[
            "subject",
            "channel",
            "time",
            "phase",
            "modality",
            "description",
        ],
        how="left",
    )
    out["mask"] = out["mask"].fillna(False).astype(bool)
    print(
        f"{task}: mask fraction={out['mask'].mean():.3f}; "
        f"significant electrodes={out.loc[out['mask'], 'channel'].nunique()}",
        flush=True,
    )
    return out


def _electrode_key(df: pd.DataFrame) -> pd.Series:
    return df["subject"].astype(str) + "|" + df["channel"].astype(str)


def plot_epoch_raster(
    df: pd.DataFrame,
    *,
    phase: str,
    compare_field: str,
    levels: tuple[str, str],
    xlim: tuple[float, float],
    colors: dict[str, str],
    out_path: Path,
) -> None:
    """Three cluster rows; two conditions stacked within each row."""
    clusters = ("sensory", "sustain", "motor")
    labels = {"sensory": "sensory", "sustain": "sustained", "motor": "motor"}
    sub = df.loc[
        (df["phase"] == phase)
        & df[compare_field].isin(levels)
        & df["functional_cluster"].isin(clusters)
        & df["time"].between(*xlim)
    ].copy()
    sub["electrode"] = _electrode_key(sub)
    sub.loc[~sub["mask"], "value"] = np.nan

    fig, axes = plt.subplots(3, 1, figsize=(3.15, 4.25), sharex=True)
    for ax, cluster in zip(axes, clusters):
        chunk = sub.loc[sub["functional_cluster"] == cluster].copy()
        any_sig = chunk.groupby("electrode")["mask"].any()
        electrodes = any_sig.index[any_sig].tolist()
        if electrodes:
            peak = (
                chunk.loc[chunk["electrode"].isin(electrodes) & chunk["mask"]]
                .groupby("electrode")["time"]
                .min()
                .sort_values()
            )
            electrodes = peak.index.tolist()

        pivots = []
        all_times = sorted(chunk["time"].dropna().unique())
        for level in reversed(levels):
            pivot = (
                chunk.loc[chunk[compare_field] == level]
                .pivot_table(
                    index="electrode", columns="time", values="value", aggfunc="mean"
                )
                .reindex(index=electrodes, columns=all_times)
            )
            pivots.append(pivot)

        matrix = pd.concat(pivots, axis=0).to_numpy(dtype=float)
        positive = matrix[np.isfinite(matrix) & (matrix > 0)]
        vmax = max(float(np.quantile(positive, 0.95)), 0.15) if positive.size else 1.0
        cmap = sns.light_palette(colors[cluster], as_cmap=True)
        cmap.set_bad("white")
        if matrix.size:
            ax.imshow(
                matrix,
                extent=(xlim[0], xlim[1], 0, max(matrix.shape[0], 1)),
                aspect="auto",
                origin="lower",
                cmap=cmap,
                vmin=0,
                vmax=vmax,
                interpolation="nearest",
                rasterized=True,
            )
        n = len(electrodes)
        ax.axhline(n, color="0.25", ls="--", lw=0.8)
        ax.axvline(0, color="0.2", ls="--", lw=0.8)
        ax.set_xlim(*xlim)
        ax.set_yticks([])
        ax.set_ylabel(labels[cluster], color=colors[cluster], fontsize=9)
        if n:
            ax.text(0.02, 0.75, levels[0], transform=ax.transAxes, fontsize=8)
            ax.text(0.02, 0.25, levels[1], transform=ax.transAxes, fontsize=8)
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[-1].set_xlabel("Time (s)", fontsize=9)
    axes[-1].set_xticks([xlim[0], 0, xlim[1]])
    fig.tight_layout(h_pad=0.6)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_epoch_traces(
    df: pd.DataFrame,
    *,
    phase: str,
    compare_field: str,
    levels: tuple[str, str],
    xlim: tuple[float, float],
    colors: dict[str, str],
    out_path: Path,
) -> None:
    clusters = ("sensory", "sustain", "motor")
    labels = {"sensory": "sensory", "sustain": "sustained", "motor": "motor"}
    sub = df.loc[
        (df["phase"] == phase)
        & df[compare_field].isin(levels)
        & df["functional_cluster"].isin(clusters)
        & df["time"].between(*xlim)
    ].copy()
    sub["electrode"] = _electrode_key(sub)

    fig, axes = plt.subplots(3, 1, figsize=(3.15, 3.75), sharex=True)
    for ax, cluster in zip(axes, clusters):
        chunk = sub.loc[sub["functional_cluster"] == cluster]
        for idx, level in enumerate(levels):
            level_df = chunk.loc[chunk[compare_field] == level]
            electrode_mean = (
                level_df.groupby(["electrode", "time"], as_index=False)["value"].mean()
            )
            summary = electrode_mean.groupby("time")["value"].agg(["mean", "sem"])
            if summary.empty:
                continue
            line_color = colors[cluster] if idx == 0 else "#333333"
            linestyle = "-" if idx == 0 else "--"
            ax.plot(
                summary.index,
                summary["mean"],
                color=line_color,
                ls=linestyle,
                lw=1.7,
                label=level,
            )
            ax.fill_between(
                summary.index,
                summary["mean"] - summary["sem"],
                summary["mean"] + summary["sem"],
                color=line_color,
                alpha=0.18,
                linewidth=0,
            )
        ax.axhline(0, color="0.2", ls="--", lw=0.7)
        ax.axvline(0, color="0.2", ls="--", lw=0.7)
        ax.set_xlim(*xlim)
        ax.set_ylabel(labels[cluster], color=colors[cluster], fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    handles = [
        Line2D([0], [0], color="black", lw=1.7, ls="-", label=levels[0]),
        Line2D([0], [0], color="black", lw=1.7, ls="--", label=levels[1]),
    ]
    axes[0].legend(handles=handles, loc="upper right", frameon=False, fontsize=8)
    axes[-1].set_xlabel("Time (s)", fontsize=9)
    axes[-1].set_xticks([xlim[0], 0, xlim[1]])
    fig.tight_layout(h_pad=0.55)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_contrast_spatial(
    assignments: pd.DataFrame,
    *,
    task: str,
    phase: str,
    contrast: str,
    direction: str,
    out_path: Path,
) -> None:
    """Render significant insula electrodes for one epoch contrast."""
    from src.decoding.viz_insula_patterns import render_insula_hemisphere_cluster
    from src.univariate.viz_mean import (
        BrainSurfaceContext,
        attach_metadata,
        filter_insula_electrodes,
        filter_qc,
        load_coord_metadata,
        load_mean_contrasts,
        resolve_mean_results_root,
        select_significant,
    )

    root = resolve_mean_results_root(task=task, contrast_desc=contrast)
    table = load_mean_contrasts(
        root, tasks=(task,), phase=phase, contrast_desc=contrast
    )
    coords = load_coord_metadata(root, tasks=(task,))
    table = filter_insula_electrodes(filter_qc(attach_metadata(table, coords)))
    table = select_significant(
        table, phase=phase, contrast=contrast, tasks=(task,)
    )
    table = table.loc[table["direction"] == direction].copy()
    table = table.merge(
        assignments[["channel", "functional_cluster"]].drop_duplicates("channel"),
        on="channel",
        how="inner",
    )
    table["pattern"] = table["mean_diff"].abs().astype(float)
    table["significant"] = True

    context = BrainSurfaceContext()
    fig, axes = plt.subplots(1, 2, figsize=(5.3, 2.35))
    for ax, (hemi, label) in zip(axes, (("lh", "Left"), ("rh", "Right"))):
        image = render_insula_hemisphere_cluster(
            table, hemi, context, size_by_pattern=False
        )
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(label, fontsize=9)
    counts = table["functional_cluster"].value_counts().to_dict()
    handles = []
    from src.nmf.waveform_analysis import FUNCTION_COLORS

    for cluster in ("sensory", "sustain", "motor"):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=FUNCTION_COLORS[cluster],
                markeredgecolor="none",
                markersize=5,
                label=f"{cluster if cluster != 'sustain' else 'sustained'} (n={counts.get(cluster, 0)})",
            )
        )
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def svg_data_uri(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{payload}"


def image_tag(path: Path, x: int, y: int, width: int, height: int) -> str:
    return (
        f'<image x="{x}" y="{y}" width="{width}" height="{height}" '
        f'preserveAspectRatio="xMidYMid meet" href="{svg_data_uri(path)}"/>'
    )


def task_schematic() -> str:
    rows = (
        ("Passive", "Sound", "No response", "#333333"),
        ("Repeat", "Sound or image", "Spoken response", "#333333"),
        ("Decision", "Sound", "Yes / No", "#333333"),
    )
    parts = [
        '<text x="96" y="118" class="panel">A</text>',
        '<text x="190" y="118" class="section">Task structure</text>',
        '<text x="680" y="118" class="event">Stimulus</text>',
        '<text x="1510" y="118" class="event">Delay</text>',
        '<text x="2370" y="118" class="event">Response</text>',
    ]
    for idx, (name, stimulus, response, color) in enumerate(rows):
        y = 168 + idx * 70
        parts.extend(
            [
                f'<text x="225" y="{y + 12}" class="condition" fill="{color}">{html.escape(name)}</text>',
                f'<line x1="610" y1="{y}" x2="3000" y2="{y}" stroke="#333" stroke-width="3"/>',
                f'<circle cx="780" cy="{y}" r="8" fill="{color}"/>',
                f'<circle cx="1640" cy="{y}" r="8" fill="{color}"/>',
                f'<circle cx="2500" cy="{y}" r="8" fill="{color}"/>',
                f'<text x="840" y="{y - 14}" class="small">{html.escape(stimulus)}</text>',
                f'<text x="1700" y="{y - 14}" class="small">Maintain task goal</text>',
                f'<text x="2560" y="{y - 14}" class="small">{html.escape(response)}</text>',
            ]
        )
    return "\n".join(parts)


def assemble_master(panel_paths: dict[str, Path], out_path: Path) -> None:
    columns = (
        ("B", "INPUT", "Sound vs Image", 80, "input"),
        ("C", "DELAY", "Decision vs Repeat", 1240, "delay"),
        ("D", "OUTPUT", "Repeat vs Passive", 2400, "output"),
    )
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="3600" height="2750" viewBox="0 0 3600 2750">',
        '<rect width="3600" height="2750" fill="white"/>',
        """<style>
        text { font-family: Arial, Helvetica, sans-serif; fill: #111; }
        .title { font-size: 58px; font-weight: 500; }
        .panel { font-size: 48px; font-weight: 700; }
        .section { font-size: 42px; font-weight: 700; }
        .subtitle { font-size: 32px; }
        .event { font-size: 30px; font-weight: 600; }
        .condition { font-size: 27px; font-weight: 600; }
        .small { font-size: 23px; }
        .rowlabel { font-size: 25px; fill: #555; }
        </style>""",
        '<text x="80" y="68" class="title">Fig. 3 | Insular dynamics dissociate input modality, upcoming goal, and speech output</text>',
        task_schematic(),
        '<line x1="80" y1="390" x2="3520" y2="390" stroke="#D0D0D0" stroke-width="3"/>',
    ]

    for panel, heading, subtitle, x, key in columns:
        parts.extend(
            [
                f'<text x="{x}" y="460" class="panel">{panel}</text>',
                f'<text x="{x + 90}" y="460" class="section">{heading}</text>',
                f'<text x="{x + 90}" y="505" class="subtitle">{subtitle}</text>',
                f'<text x="{x}" y="600" class="rowlabel">Spatial distribution</text>',
                image_tag(panel_paths[f"{key}_spatial"], x, 610, 1080, 560),
                f'<text x="{x}" y="1220" class="rowlabel">Electrode × time</text>',
                image_tag(panel_paths[f"{key}_raster"], x, 1230, 1080, 700),
                f'<text x="{x}" y="1990" class="rowlabel">Mean HGA</text>',
                image_tag(panel_paths[f"{key}_traces"], x, 2000, 1080, 690),
            ]
        )
    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def main() -> None:
    namespace = execute_notebook_cells(
        ("fig3-imports", "fig3-style", "fig3-helpers", "fig3-load", "fig3-pn-load")
    )
    working: pd.DataFrame = namespace["working"].copy()
    pn_repeat: pd.DataFrame = namespace["repeat"].copy()
    assignments: pd.DataFrame = namespace["assignments"].copy()
    colors = namespace["FUNCTION_COLORS"]
    hga_results_dir = namespace["hga_results_dir"]

    working = working.loc[~working["channel"].isin(namespace["DROP_CHANNELS"])].copy()

    input_df = pn_repeat.loc[
        (pn_repeat["phase"] == "stimulus")
        & pn_repeat["modality"].isin(["sound", "image"])
    ].copy()
    delay_df = working.loc[
        (working["task"] == "LexicalDelay")
        & (working["phase"] == "delay")
        & working["description"].isin(["Decision", "Repeat"])
    ].copy()
    output_df = working.loc[
        (working["task"] == "PictureNaming")
        & (working["phase"] == "response")
        & working["description"].isin(["Repeat", "Passive"])
    ].copy()

    input_df = add_packaged_mask(
        input_df, task="PictureNaming", hga_results_dir=hga_results_dir
    )
    delay_df = add_packaged_mask(
        delay_df, task="LexicalDelay", hga_results_dir=hga_results_dir
    )
    output_df = add_packaged_mask(
        output_df, task="PictureNaming", hga_results_dir=hga_results_dir
    )

    paths = {
        "input_spatial": FIG3_DIR / "fig3_pn_repeat_sound_vs_image_spatial.svg",
        "delay_spatial": FIG3_DIR / "fig3_delay_decision_gt_repeat_spatial.svg",
        "output_spatial": FIG3_DIR / "fig3_output_repeat_gt_passive_spatial.svg",
        "input_raster": FIG3_DIR / "fig3_epoch_input_raster.svg",
        "delay_raster": FIG3_DIR / "fig3_epoch_delay_raster.svg",
        "output_raster": FIG3_DIR / "fig3_epoch_output_raster.svg",
        "input_traces": FIG3_DIR / "fig3_epoch_input_traces.svg",
        "delay_traces": FIG3_DIR / "fig3_epoch_delay_traces.svg",
        "output_traces": FIG3_DIR / "fig3_epoch_output_traces.svg",
    }

    plot_epoch_raster(
        input_df,
        phase="stimulus",
        compare_field="modality",
        levels=("sound", "image"),
        xlim=(-0.5, 0.6),
        colors=colors,
        out_path=paths["input_raster"],
    )
    plot_epoch_raster(
        delay_df,
        phase="delay",
        compare_field="description",
        levels=("Decision", "Repeat"),
        xlim=(0.0, 1.0),
        colors=colors,
        out_path=paths["delay_raster"],
    )
    plot_epoch_raster(
        output_df,
        phase="response",
        compare_field="description",
        levels=("Repeat", "Passive"),
        xlim=(-0.5, 1.0),
        colors=colors,
        out_path=paths["output_raster"],
    )

    plot_epoch_traces(
        input_df,
        phase="stimulus",
        compare_field="modality",
        levels=("sound", "image"),
        xlim=(-0.5, 0.6),
        colors=colors,
        out_path=paths["input_traces"],
    )
    plot_epoch_traces(
        delay_df,
        phase="delay",
        compare_field="description",
        levels=("Decision", "Repeat"),
        xlim=(0.0, 1.0),
        colors=colors,
        out_path=paths["delay_traces"],
    )
    plot_epoch_traces(
        output_df,
        phase="response",
        compare_field="description",
        levels=("Repeat", "Passive"),
        xlim=(-0.5, 1.0),
        colors=colors,
        out_path=paths["output_traces"],
    )
    plot_contrast_spatial(
        assignments,
        task="LexicalDelay",
        phase="Delay",
        contrast="DecisionVsRepeatMean",
        direction="Decision",
        out_path=paths["delay_spatial"],
    )
    plot_contrast_spatial(
        assignments,
        task="PictureNaming",
        phase="Response",
        contrast="RepeatVsPassiveMean",
        direction="Repeat",
        out_path=paths["output_spatial"],
    )

    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Figure 3 panels: {missing}")

    master = PUB_DIR / "pub-new_fig3-restructured.svg"
    assemble_master(paths, master)
    print(f"Wrote {master}", flush=True)


if __name__ == "__main__":
    main()
