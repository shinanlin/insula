#!/usr/bin/env python3
"""Audit existing Figure 1 and four-phase NMF eligibility; never fit a model.

Read the original packaged HGA and frozen assignments. All outputs are audit
tables in a separate directory. Repeat-site membership, Figure 1 significance,
and NMF matrix eligibility are recorded separately rather than assumed nested.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from src.nmf.waveform_analysis import (
    TASKS, PHASE_WINDOWS_POSTONSET, concatenated_phase_matrix,
    phase_matrix, prepare_shape_matrix,
)

FIG_WINDOWS = {"stimulus": (0., 1.), "delay": (0., 1.), "response": (-.5, .5)}
ALIASES = {"audio": "stimulus", "resp": "response"}
COLS = ["channel", "time", "value", "mask", "subject", "description", "phase",
        "modality", "roi", "mix", "label", "hemi", "x", "y", "z"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def insula(frame):
    return frame.roi.isin(["AIC", "PIC"]) & ~frame.mix.fillna(False).astype(bool)


def examine_time(path):
    """Return memberships and small insula waveform subset from one file."""
    frame = pd.read_csv(path, usecols=COLS)
    frame["phase"] = frame.phase.astype(str).str.lower().replace(ALIASES)
    frame = frame.loc[frame.description.eq("Repeat")]
    sets = {"repeat_packaged_any_phase": set(frame.channel)}
    for phase, (lo, hi) in FIG_WINDOWS.items():
        sig = frame.loc[frame.phase.eq(phase) & frame.time.gt(lo) & frame.time.lt(hi)
                        & frame["mask"].eq(True)]
        sets[f"fig_sig_{phase}"] = set(sig.channel)
        sets[f"fig_positive_sig_{phase}"] = set(sig.loc[sig.value.gt(0), "channel"])
        sets[f"fig_sound_sig_{phase}"] = set(sig.loc[sig.modality.eq("sound"), "channel"])
        # Older Figure 1 pie path uses the anatomy attached to time rows and
        # additionally excludes numeric midpoint label=0.
        pie = sig.loc[insula(sig) & ~pd.to_numeric(sig.label, errors="coerce").eq(0)]
        sets[f"fig_pie_insula_{phase}"] = set(pie.channel)
    four = frame.loc[frame.phase.isin(PHASE_WINDOWS_POSTONSET)]
    sets["repeat_four_phase_present"] = set(four.channel)
    sound = four.loc[four.modality.eq("sound")]
    sets["repeat_sound_present"] = set(sound.channel)
    anatomical = sound.loc[insula(sound)]
    sets["sound_insula_mixfalse"] = set(anatomical.channel)
    no_subject = anatomical.loc[~anatomical.subject.eq("D0121")]
    sets["after_subject_exclusion"] = set(no_subject.channel)
    zero = pd.to_numeric(no_subject.label, errors="coerce").eq(0)
    sets["midpoint_zero_rows"] = set(no_subject.loc[zero, "channel"])
    eligible = no_subject.loc[~zero]
    sets["after_midpoint_zero_exclusion"] = set(eligible.channel)
    # Store only cropped input rows; no per-time significance mask is applied
    # to NMF waveforms, matching the historical input-loading implementation.
    keep = np.zeros(len(eligible), dtype=bool)
    for phase, (lo, hi) in PHASE_WINDOWS_POSTONSET.items():
        keep |= (eligible.phase.eq(phase) & eligible.time.gt(lo) & eligible.time.lt(hi)).to_numpy()
    reduced = eligible.loc[keep].copy()
    stat = path.stat()
    record = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
              "rows_repeat": len(frame), "channels_repeat": frame.channel.nunique()}
    return sets, reduced, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    if source == out or source in out.parents:
        raise ValueError("Audit outputs must be outside the original results tree")
    out.mkdir(parents=True, exist_ok=True)
    assignments_path = source / "nmf/channel_assignments.csv"
    rank_path = source / "nmf/rank_selection_meta.json"
    exclude_path = source / "nmf/exclude_channels.txt"
    small_sources = [assignments_path, rank_path, exclude_path,
                     source / "nmf/nmf_manifest.json",
                     source.parent / "vizpub/fig1.ipynb",
                     source.parent / "src/nmf/waveform_analysis.py",
                     source.parent / "src/hga/package_highgamma.py"]
    hashes = {str(p): digest(p) for p in small_sources}
    assignments = pd.read_csv(assignments_path)
    saved = set(assignments.channel)
    rank = json.loads(rank_path.read_text())
    rank_saved = set(rank["channels"])
    excluded = set(exclude_path.read_text().split())

    # Task order is intentional: Figure 1 keeps the first available coordinate
    # record for a repeated channel, before applying the insula eligibility rule.
    coord_paths, time_paths = [], []
    for task in TASKS:
        root = source / "hga" / task
        coord_paths += sorted(root.glob("sub-*/HGA/*_coord.csv"))
        time_paths += sorted(root.glob("sub-*/HGA/*desc-Repeat_time.csv"))
    coord_frames = []
    coord_manifest = []
    for path in coord_paths:
        frame = pd.read_csv(path)
        frame["source_path"] = str(path)
        coord_frames.append(frame)
        st = path.stat()
        coord_manifest.append({"path": str(path), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns})
    coords = pd.concat(coord_frames, ignore_index=True)
    valid = coords.dropna(subset=["x", "y", "z"])
    unique = valid.drop_duplicates("channel", keep="first")
    sampled = unique.loc[insula(unique)].copy().set_index("channel")
    sampled.to_csv(out / "sampled_464_coordinates.csv")
    S = set(sampled.index)
    any_insula = set(valid.loc[insula(valid), "channel"])
    fields = ["channel", "task", "phase", "modality", "description", "label", "roi", "mix",
              "hemi", "x", "y", "z", "contact_1_label", "contact_2_label", "source_path"]
    valid = valid.assign(eligible_insula=insula(valid))
    conflicts = valid.groupby("channel").eligible_insula.nunique()
    conflict_set = set(conflicts.loc[conflicts.gt(1)].index)
    valid.loc[valid.channel.isin(conflict_set), fields + ["eligible_insula"]].drop_duplicates(
        subset=[f for f in fields if f != "source_path"] + ["eligible_insula"]
    ).to_csv(out / "cross_task_anatomy_conflicts.csv", index=False)
    print(f"Coordinates: {len(coords)} rows, {len(unique)} valid unique sites; insula {len(S)}", flush=True)

    pools = defaultdict(set)
    reduced_frames, manifest = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for i, (memberships, reduced, record) in enumerate(executor.map(examine_time, time_paths), 1):
            for key, channels in memberships.items():
                pools[key].update(channels)
            if len(reduced):
                reduced_frames.append(reduced)
            manifest.append(record)
            if i % 100 == 0 or i == len(time_paths):
                print(f"Read {i}/{len(time_paths)} Repeat files", flush=True)
    for prefix in ["fig_sig", "fig_positive_sig", "fig_sound_sig", "fig_pie_insula"]:
        pools[prefix + "_union"] = set().union(*(pools[f"{prefix}_{p}"] for p in FIG_WINDOWS))
    pools["after_manual_exclusion"] = pools["after_midpoint_zero_exclusion"] - excluded
    rows = pd.concat(reduced_frames, ignore_index=True)
    rows = rows.loc[~rows.channel.isin(excluded)]
    rows.to_pickle(out / "nmf_cropped_input_rows.pkl.gz")
    pools["after_window_crop"] = set(rows.channel)
    coverage_records = []
    all_four_present = set(rows.channel)
    for phase in PHASE_WINDOWS_POSTONSET:
        phase_rows = rows.loc[rows.phase.eq(phase)]
        raw = phase_rows.groupby(["channel", "time"])["value"].mean().unstack("time")
        coverage = raw.notna().mean(axis=1)
        for channel, proportion in coverage.items():
            coverage_records.append({"channel": channel, "phase": phase,
                                     "n_time_columns": raw.shape[1], "coverage": proportion,
                                     "passes_95pct": proportion >= .95})
        mat, _ = phase_matrix(rows, phase, min_coverage=.95)
        pools[f"present_{phase}"] = set(phase_rows.channel)
        pools[f"coverage_{phase}"] = set(mat.index)
        all_four_present &= set(phase_rows.channel)
    pools["all_four_phases_present"] = all_four_present
    pd.DataFrame(coverage_records).to_csv(out / "phase_coverage.csv", index=False)
    raw_matrix, metadata, slices = concatenated_phase_matrix(rows, tuple(PHASE_WINDOWS_POSTONSET))
    pools["all_four_coverage95"] = set(raw_matrix.index)
    shape, keep = prepare_shape_matrix(raw_matrix.to_numpy())
    reconstructed = set(raw_matrix.index[keep])
    pools["reconstructed_nmf_eligible"] = reconstructed
    pools["saved_nmf"] = saved
    pools["rank_selection_saved"] = rank_saved
    print(f"Reconstructed historical INPUT ONLY: {shape.shape}; saved assignments {len(saved)}", flush=True)

    # Every relevant channel has separate flags so divergences are visible.
    universe = sorted(S | any_insula | saved | pools["sound_insula_mixfalse"])
    sites = pd.DataFrame({"channel": universe}).set_index("channel")
    sites["subject"] = [c.split("_")[0] for c in sites.index]
    sites["fig1_sampled"] = sites.index.isin(S)
    sites["any_task_anatomically_insula"] = sites.index.isin(any_insula)
    sites["cross_task_eligibility_conflict"] = sites.index.isin(conflict_set)
    sites["explicit_26_list"] = sites.index.isin(excluded)
    sites["excluded_subject_D0121"] = sites.subject.eq("D0121")
    for field in ["roi", "hemi", "label", "contact_1_label", "contact_2_label"]:
        sites[f"fig1_{field}"] = unique.set_index("channel")[field].reindex(sites.index)
    for key, channels in pools.items():
        sites[key] = sites.index.isin(channels)
    stages = ["repeat_packaged_any_phase", "repeat_four_phase_present", "repeat_sound_present",
              "sound_insula_mixfalse", "after_subject_exclusion", "after_midpoint_zero_exclusion",
              "after_manual_exclusion", "after_window_crop", "all_four_phases_present",
              "all_four_coverage95", "reconstructed_nmf_eligible"]
    reasons = ["not_in_Repeat_packaged_pool", "no_requested_phase", "no_sound_rows",
               "no_pure_insula_sound_rows", "excluded_subject_D0121", "midpoint_label_zero",
               "explicit_26_channel_QC_list", "no_rows_in_crop_windows", "missing_phase",
               "phase_coverage_below_95pct", "zero_positive_HGA_norm"]
    sites["first_nmf_exclusion"] = "eligible"
    previous = set(universe)
    stage_counts = []
    for key, reason in zip(stages, reasons):
        lost = previous - pools[key]
        sites.loc[sites.index.isin(lost), "first_nmf_exclusion"] = reason
        pd.DataFrame({"channel": sorted(lost & S)}).to_csv(out / f"excluded_{key}.csv", index=False)
        previous &= pools[key]
        stage_counts.append({"stage": key, "global_count": len(pools[key]),
                             "within_464": len(S & pools[key]), "fig1_responsive_within_464":
                             len(S & pools["fig_sig_union"] & pools[key])})
    sites["first_nmf_exclusion"] = sites.first_nmf_exclusion.where(
        ~sites.saved_nmf | sites.reconstructed_nmf_eligible, "SAVED_BUT_NOT_RECONSTRUCTED")
    sites.to_csv(out / "site_selection_audit.csv")
    pd.DataFrame(stage_counts).to_csv(out / "nmf_stages.csv", index=False)
    R = S & pools["fig_sig_union"]
    for name, channels in {
        "responsive_fig1": R, "not_responsive_fig1": S - R,
        "responsive_not_saved_nmf": R - saved, "saved_nmf_not_fig1_responsive": saved - R,
        "saved_nmf_outside_464": saved - S,
        "reconstructed_not_saved": reconstructed - saved,
        "saved_not_reconstructed": saved - reconstructed,
    }.items():
        sites.loc[sites.index.isin(channels)].to_csv(out / f"{name}.csv")
    per_phase = {p: len(S & pools[f"fig_sig_{p}"]) for p in FIG_WINDOWS}
    checks = {
        "sampled_count_464": len(S) == 464,
        "responsive_count_348": len(R) == 348,
        "saved_count_255": len(saved) == 255,
        "rank_and_assignments_same_membership": rank_saved == saved,
        "reconstructed_matches_saved": reconstructed == saved,
        "input_feature_count_matches_rank": shape.shape[1] == rank["n_features"],
        "small_sources_unchanged": hashes == {str(p): digest(p) for p in small_sources},
    }
    # Stat every input again to detect modifications during the audit.
    input_manifest = coord_manifest + manifest
    changed = [r["path"] for r in input_manifest
               if (Path(r["path"]).stat().st_size, Path(r["path"]).stat().st_mtime_ns)
               != (r["bytes"], r["mtime_ns"])]
    checks["packaged_inputs_unchanged_during_audit"] = not changed
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_results": str(source), "audit_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "audit_script_sha256": digest(Path(__file__)), "model_fitted": False,
        "task_order": list(TASKS), "coord_files": len(coord_paths), "repeat_time_files": len(time_paths),
        "sampled": len(S), "sampled_subjects": sampled.subject.nunique(),
        "sampled_hemispheres": sampled.hemi.value_counts().to_dict(),
        "responsive": len(R), "responsive_by_phase_overlapping": per_phase,
        "responsive_hemispheres": sampled.loc[sorted(R), "hemi"].value_counts().to_dict(),
        "responsive_sound_only": len(S & pools["fig_sound_sig_union"]),
        "responsive_with_positive_value": len(S & pools["fig_positive_sig_union"]),
        "fig1_pie_insula_route_count": len(pools["fig_pie_insula_union"]),
        "fig1_pie_route_in_464": len(S & pools["fig_pie_insula_union"]),
        "saved_nmf": len(saved), "saved_nmf_subjects": assignments.subject.nunique(),
        "nmf_in_464": len(S & saved), "nmf_in_responsive": len(R & saved),
        "responsive_not_nmf": len(R - saved), "nmf_outside_464": sorted(saved - S),
        "nmf_not_fig1_responsive": sorted(saved - R),
        "reconstructed_matrix_shape": list(shape.shape), "stages": stage_counts,
        "exclusions_among_464": sites.loc[sites.fig1_sampled].first_nmf_exclusion.value_counts().to_dict(),
        "exclusions_among_responsive": sites.loc[sites.index.isin(R)].first_nmf_exclusion.value_counts().to_dict(),
        "any_task_insula_count": len(any_insula), "checks": checks,
        "changed_input_files": changed, "small_source_hashes": hashes,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (out / "input_manifest.json").write_text(json.dumps(input_manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
