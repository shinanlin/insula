"""Actual-data Figure 3 draft: approved task comparisons, frozen NMF labels.

This module only reads validated native epochs/statistics. It neither refits
NMF nor performs new condition tests. The publication notebook supplies the
existing condition palettes, SEM/ylim recipe and masked-heatmap conventions.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import warnings
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_hex, to_rgb
from matplotlib.font_manager import fontManager

from src.nmf.waveform_analysis import FUNCTION_COLORS

ROOT = Path(__file__).resolve().parents[1]
CLUSTERS = ("sensory", "sustain", "motor")
LABELS = dict(sensory="Sensory", sustain="Sustained", motor="Motor")
SPECS = (
    dict(key="input", letter="A", title="Input modality", task="PictureNaming",
         task_label="Picture Naming", phase="Stimulus", field="modality",
         levels=("sound", "image"), xlim=(-.5, .6),
         instructions=("Sound → ‘apple’", "Image → ‘apple’")),
    dict(key="speech", letter="B", title="Speech requirement", task="PictureNaming",
         task_label="Picture Naming · sound", phase="Delay", field="condition",
         levels=("Repeat", "Passive"), xlim=(0., 1.),
         instructions=("Repeat → speak later", "Passive → no speech")),
    dict(key="rule", letter="C", title="Response rule", task="LexicalDelay",
         task_label="Lexical Delay · sound", phase="Delay", field="condition",
         levels=("Decision", "Repeat"), xlim=(0., 1.),
         instructions=("Repeat → word", "Decision → yes / no")),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def source_helpers():
    """Reuse definitions verbatim, without executing the main notebook."""
    from src import figure3_legacy_hga
    names = {"rfig3_blend_rgb", "rfig3_darken", "rfig3_lighten",
             "rfig3_condition_color", "rfig3_condition_cmap",
             "rfig3_compute_shared_ylim", "rfig3_electrode_key"}
    return {name:getattr(figure3_legacy_hga,name) for name in names}


def protected_snapshot():
    registry = ROOT / "results/audit/figure4_formal_2026-10-20/protected_main_inputs.json"
    entries = json.loads(registry.read_text())
    # Capture the current author-approved display version as well as all data.
    if isinstance(entries, dict) and "files" in entries:
        entries = entries["files"]
    assert isinstance(entries, dict)
    return {str(p): digest(p) for p in entries}


def _names(dataset):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in dataset[:]]


def _fingerprint(path):
    p = Path(path).resolve()
    stat = p.stat()
    return dict(path=str(p), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def _trial_mean(product, requested, cache_dir, input_rows, missing_rows):
    """Original PN streaming nanmean, keeping only requested NMF channels."""
    epoch_path = Path(product["zscore_path"])
    stats_path = Path(product["statistics_path"])
    identity = dict(epoch=_fingerprint(epoch_path), stats=_fingerprint(stats_path),
                    channels=sorted(requested), statistics_sha256=digest(stats_path))
    cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache_path = cache_dir / (cache_key + ".npz")
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            channels = cached["channels"].tolist()
            times, mean, mask = cached["times"], cached["mean"], cached["mask"]
            n_trials = int(cached["n_trials"])
    else:
        epochs = mne.read_epochs(epoch_path, preload=False, verbose="error")
        channels = [ch for ch in sorted(requested) if ch in epochs.ch_names]
        times = epochs.times.copy()
        n_trials = len(epochs.events)
        assert n_trials == product["trials"] and n_trials > 0
        if not channels:
            mean = np.empty((0, len(times)))
            mask = np.empty((0, len(times)), dtype=bool)
        else:
            total = np.zeros((len(channels), len(times)), dtype=float)
            count = np.zeros_like(total)
            for i in range(n_trials):
                epoch = epochs.get_data(picks=channels, item=[i], copy=False, verbose="error")[0]
                assert not np.isinf(epoch).any(), "Native QC unexpectedly retained infinity"
                good = np.isfinite(epoch)
                total += np.where(good, epoch, 0.)
                count += good
            mean = np.full_like(total, np.nan)
            np.divide(total, count, out=mean, where=count > 0)
            with h5py.File(stats_path, "r") as stats:
                stat_names = _names(stats["ch_names"])
                assert stat_names == epochs.ch_names
                assert stats["mask"].shape == (len(stat_names), len(times))
                indices = [stat_names.index(ch) for ch in channels]
                mask = stats["mask"][:][indices].astype(bool)
        del epochs
        np.savez_compressed(cache_path, channels=np.asarray(channels), times=times,
                            mean=mean, mask=mask, n_trials=n_trials)
    identity.update(n_trials=n_trials, n_selected_channels=len(channels), cache=str(cache_path))
    input_rows.append(identity)
    for ch in sorted(set(requested) - set(channels)):
        missing_rows.append(dict(channel=ch, source=str(epoch_path),
                                 reason="absent from restored native-QC epoch"))
    return channels, times, mean, mask, n_trials


def load_data(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / "epoch_means"
    cache_dir.mkdir(exist_ok=True)
    assignment_path = ROOT / "results/nmf/channel_assignments.csv"
    assignments = pd.read_csv(assignment_path)
    assert len(assignments) == 231 and not assignments.duplicated(["subject", "channel"]).any()
    manifest = json.loads((ROOT / "results/nmf/nmf_manifest.json").read_text())
    assert manifest["phases"] == ["stimulus", "delay", "response"]
    assert manifest["modalities"] == ["sound"]
    assert assignments.functional_cluster.isin(CLUSTERS).all()
    metadata = assignments.set_index(["subject", "channel"])
    input_rows, missing_rows, frames, gate_rows = [], [], [], []
    coverage = pd.read_csv(ROOT / "results/paper_trial_inputs/subject_task_coverage.csv")
    for task in ("PictureNaming", "LexicalDelay"):
        for subject, group in assignments.groupby("subject", sort=True):
            if subject == "D0121":
                continue
            status = coverage.loc[coverage.task.eq(task) & coverage.subject.eq(subject)]
            if status.empty:
                continue
            validation_path = ROOT / "results/native_preprocessing_restore_2026-10-18" / task / subject / "output_validation.json"
            if not validation_path.exists():
                assert status.iloc[0].status == "native_QC_zero_usable", str(validation_path)
                for ch in group.channel:
                    missing_rows.append(dict(task=task, subject=subject, channel=ch,
                                             reason="native QC: no usable task sites"))
                continue
            validation = json.loads(validation_path.read_text())
            assert validation["validation_passed"]
            products = validation["products"]
            # Original context loader: sound-only, all-condition/all-phase
            # significance union, intersected with the frozen NMF assignments.
            sig_union = set()
            for product in products:
                if product["modality"] == "sound" and product["condition"] in {"Repeat", "Passive", "Decision"}:
                    with h5py.File(product["statistics_path"], "r") as stats:
                        sig_union.update(_names(stats["sig_ch_names"]))
            assigned = set(group.channel)
            context = assigned & sig_union
            gate_rows.append(dict(task=task, subject=subject, assigned_sites=len(assigned),
                                  original_context_union_sites=len(context)))
            for spec in SPECS:
                if spec["task"] != task:
                    continue
                # Original modality panel uses all available NMF sites; the
                # two context comparisons retain their original task sig union.
                requested = assigned if spec["key"] == "input" else context
                if not requested:
                    continue
                for level in spec["levels"]:
                    condition = "Repeat" if spec["field"] == "modality" else level
                    modality = level if spec["field"] == "modality" else "sound"
                    matches = [p for p in products if (p["condition"], p["modality"], p["phase"]) == (condition, modality, spec["phase"])]
                    assert len(matches) == 1, (task, subject, spec["key"], level)
                    print(task, subject, spec["key"], level, "sites", len(requested), flush=True)
                    channels, times, mean, mask, n_trials = _trial_mean(matches[0], requested, cache_dir, input_rows, missing_rows)
                    select = (times >= spec["xlim"][0]) & (times <= spec["xlim"][1])
                    assert select.any()
                    n_t = int(select.sum())
                    if not channels:
                        continue
                    frame = pd.DataFrame(dict(
                        channel=np.repeat(channels, n_t), time=np.tile(times[select], len(channels)),
                        value=mean[:, select].reshape(-1), mask=mask[:, select].reshape(-1)))
                    frame["subject"], frame["task"], frame["phase"] = subject, task, spec["phase"].lower()
                    frame["row"], frame["level"], frame["n_trials"] = spec["key"], level, n_trials
                    frame["functional_cluster"] = frame.channel.map(group.set_index("channel").functional_cluster)
                    frame["electrode"] = subject + "|" + frame.channel
                    frame["mask_observed"] = True
                    frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    actual = data[["subject", "channel", "functional_cluster"]].drop_duplicates()
    check = actual.set_index(["subject", "channel"]).functional_cluster
    assert check.eq(metadata.loc[check.index, "functional_cluster"]).all()
    assert data.groupby(["row", "electrode", "time"])["level"].nunique().eq(2).all(), "Unpaired native waveforms; inspect instead of silently dropping"
    data.to_pickle(out / "waveforms.pkl")
    data.to_csv(out / "waveforms.csv.gz", index=False)
    pd.DataFrame(missing_rows).to_csv(out / "missing_sites.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(out / "candidate_gates.csv", index=False)
    write_json(out / "native_input_manifest.json", dict(
        assignments_sha256=digest(assignment_path), n_frozen_sites=len(assignments),
        cluster_counts=assignments.functional_cluster.value_counts().to_dict(),
        selected_native_files=input_rows,
        native_mask_recipe="unchanged saved time_perm_cluster mask, mapped by native channel and epoch time; same package_highgamma recipe",
        cohort_recipe="original PN modality loader and task-specific sound sig-union context loader intersect frozen NMF labels",
        paired_waveforms_checked=True, new_inferential_tests=False))
    return data


def _matrices(data, spec):
    row = data.loc[data.row.eq(spec["key"])].copy()
    panels = []
    for cluster in CLUSTERS:
        sub = row.loc[row.functional_cluster.eq(cluster)].copy()
        first = sub.loc[sub["mask"]].groupby("electrode").time.min()
        electrodes = sorted(sub.electrode.unique(), key=lambda e: (first.get(e, np.inf), e))
        times = np.sort(sub.time.unique())
        matrices = {}
        for level in spec["levels"]:
            masked = sub.loc[sub["level"].eq(level)].copy()
            masked.loc[~masked["mask"], "value"] = np.nan
            matrices[level] = masked.pivot(index="electrode", columns="time", values="value").reindex(index=electrodes, columns=times).to_numpy()
        all_values = np.concatenate([v.ravel() for v in matrices.values()])
        positive = all_values[np.isfinite(all_values) & (all_values > 0)]
        vmax = max(float(np.quantile(positive, .95)), .15) if positive.size else 1.
        panels.append(dict(cluster=cluster, data=sub, electrodes=electrodes,
                           matrices=matrices, vmax=vmax, n=len(electrodes)))
    return panels


def plot_draft(data, destination, out):
    """Approved new arrangement, original per-panel numerical/display recipes."""
    out, destination = Path(out), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    helpers = source_helpers()
    for font in Path(os.sys.prefix, "fonts").glob("arial*.ttf"):
        fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"],
                         "font.size": 7, "svg.fonttype": "none", "axes.linewidth": .75})
    cm = 1 / 2.54
    width = 18.8
    left, col_width, col_gap = 4.35, 4.05, .65
    trace_h, gap, row_title_h, lower_h, between_h = 1.25, .25, .50, .62, .64
    rows = [_matrices(data, spec) for spec in SPECS]
    raster_slots = [max(.35, min(3.2, 2 * max(p["n"] for p in panels) * .02)) for panels in rows]
    row_heights = [row_title_h + trace_h + gap + rh + lower_h for rh in raster_slots]
    height = .85 + sum(row_heights) + 2 * between_h + .75
    fig = plt.figure(figsize=(width * cm, height * cm))
    pos = lambda x, y, w, h: [x / width, y / height, w / width, h / height]
    for j, cluster in enumerate(CLUSTERS):
        fig.text((left + j * (col_width + col_gap) + col_width / 2) / width,
                 1 - .4 / height, LABELS[cluster], color=FUNCTION_COLORS[cluster],
                 ha="center", va="center", fontsize=9, fontweight="bold")
    counts, memberships, orders, scales = [], [], [], []
    top = height - .85
    for ri, (spec, panels, rh, row_h) in enumerate(zip(SPECS, rows, raster_slots, row_heights)):
        bottom = top - row_h
        raster_y = bottom + lower_h
        trace_y = raster_y + rh + gap
        row_data = data.loc[data.row.eq(spec["key"])]
        ylim = helpers["rfig3_compute_shared_ylim"](
            row_data, phases=(spec["phase"].lower(),), compare_field="level", levels=spec["levels"],
            xlim_by_phase={spec["phase"].lower(): spec["xlim"]})
        fig.text(.15 / width, (top - .15) / height, spec["letter"], fontsize=10, fontweight="bold", va="top")
        fig.text(.60 / width, (top - .15) / height, spec["title"], fontsize=8, fontweight="bold", va="top")
        fig.text(.60 / width, (top - .64) / height, spec["task_label"], fontsize=6.5, color=".35", va="top")
        for k, line in enumerate(spec["instructions"]):
            fig.text(.60 / width, (top - 1.26 - .38 * k) / height, line, fontsize=7, va="top")
        fig.text(.60 / width, (top - 2.32) / height, spec["phase"] + "-aligned", fontsize=7, va="top")
        cm_per = rh / (2 * max(p["n"] for p in panels)) if max(p["n"] for p in panels) else .02
        for ci, panel in enumerate(panels):
            cluster, sub, n = panel["cluster"], panel["data"], panel["n"]
            x = left + ci * (col_width + col_gap)
            ax = fig.add_axes(pos(x, trace_y, col_width, trace_h))
            for level in spec["levels"]:
                values = sub.loc[sub["level"].eq(level)].groupby(["electrode", "time"])["value"].mean().groupby("time").agg(["mean", "sem"])
                color = helpers["rfig3_condition_color"](level, FUNCTION_COLORS[cluster])
                ax.plot(values.index, values["mean"], color=color, lw=1., label=level.capitalize())
                ax.fill_between(values.index, values["mean"] - values["sem"], values["mean"] + values["sem"], color=color, alpha=.18, linewidth=0)
                condition_data = sub.loc[sub["level"].eq(level)]
                counts.append(dict(row=spec["key"], phase=spec["phase"], functional_cluster=cluster,
                                   condition=level, n_sites=condition_data.electrode.nunique(),
                                   n_subjects=condition_data.subject.nunique(),
                                   n_mask_positive_sites=condition_data.loc[condition_data["mask"]].electrode.nunique(),
                                   n_finite_sites=condition_data.loc[np.isfinite(condition_data.value)].electrode.nunique()))
            ax.axvline(0, color=".2", ls="--", lw=.7)
            ax.set(xlim=spec["xlim"], ylim=ylim)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7, width=.75, length=2, labelbottom=False)
            if ci == 0:
                ax.set_ylabel("HGA (z)", fontsize=7)
            ax.text(0, 1.10, f"n={n} sites, {sub.subject.nunique()} subjects", transform=ax.transAxes, fontsize=6.4, va="bottom")
            ax.legend(frameon=False, fontsize=6, loc="upper right", ncol=2, handlelength=1.1,
                      columnspacing=.65, handletextpad=.35, borderaxespad=0)
            raster_h = max(.04, n * 2 * cm_per)
            ar = fig.add_axes(pos(x, raster_y, col_width, raster_h))
            for half, level in enumerate(reversed(spec["levels"])):
                cmap = helpers["rfig3_condition_cmap"](level, FUNCTION_COLORS[cluster])
                cmap.set_bad("white")
                matrix = panel["matrices"][level]
                if n and matrix.shape[1]:
                    ar.imshow(matrix, extent=(*spec["xlim"], half * n, (half + 1) * n),
                              aspect="auto", origin="lower", cmap=cmap, vmin=0,
                              vmax=panel["vmax"], interpolation="nearest", rasterized=True)
                ar.text(.01, (half + .5) / 2, level.capitalize(), transform=ar.transAxes,
                        fontsize=6.2, va="center", bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=.4))
            ar.axhline(n, color=".25", ls="--", lw=.7)
            ar.axvline(0, color=".2", ls="--", lw=.7)
            ar.set(xlim=spec["xlim"], ylim=(0, max(2*n, 1)), yticks=[])
            ar.set_xticks(sorted(set([spec["xlim"][0], 0., spec["xlim"][1]])))
            ar.tick_params(labelsize=7, width=.75, length=2)
            ar.spines[["top", "right", "left"]].set_visible(False)
            if ci == 1:
                ar.set_xlabel(f"Time from {spec['phase'].lower()} onset (s)", fontsize=7)
            # Source single-phase recipe uses a separate P95 scale per cluster.
            ar.text(1., 1.015, f"0–{panel['vmax']:.2g} z", transform=ar.transAxes, fontsize=5.8, color=".4", ha="right", va="bottom")
            scales.append(dict(row=spec["key"], functional_cluster=cluster,
                               ylim=list(ylim), heatmap_vmin=0., heatmap_vmax=panel["vmax"], cm_per_raster_row=cm_per))
            for rank, electrode in enumerate(panel["electrodes"]):
                orders.append(dict(row=spec["key"], functional_cluster=cluster, rank=rank, electrode=electrode))
            memberships.append(sub[["row", "task", "subject", "channel", "functional_cluster", "phase", "level", "n_trials"]].drop_duplicates())
        top = bottom - between_h
    fig.text(.16 / width, .24 / height,
             "Draft · Frozen three-phase NMF labels · Heatmaps: baseline-significant samples only; white = not significant.",
             fontsize=6, color=".35", va="bottom")
    fig.savefig(destination, format="svg", dpi=300, bbox_inches=None, pad_inches=0)
    pd.DataFrame(counts).to_csv(out / "panel_counts.csv", index=False)
    pd.concat(memberships).to_csv(out / "membership.csv", index=False)
    pd.DataFrame(orders).to_csv(out / "electrode_order.csv", index=False)
    pd.DataFrame(scales).to_csv(out / "display_scales.csv", index=False)
    write_json(out / "draft_description.json", dict(
        svg=str(destination), specs=SPECS, numerical_reference="vizpub/fig3.ipynb",
        layout="Three task-comparison rows by three component columns; traces above paired masked heatmaps",
        tests="No new condition-difference test or significance annotation",
        mask_source="validated restored native time_perm_cluster statistics",
        heatmap_scale="original per-population positive-value P95, minimum .15",
        missing_masks=0, all_mask_false_points_blank=True,
        assignment_sha256=digest(ROOT / "results/nmf/channel_assignments.csv")))
    return fig, pd.DataFrame(counts)
