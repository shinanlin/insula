"""Build HGA Explorer JSON (electrodes + manifest) from packaged results(nw)/ HGA."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

VIEWER_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = VIEWER_ROOT.parent.parent
DEFAULT_DATA_DIR = VIEWER_ROOT / "public" / "data"
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "results(nw)"
_EXPORT_DIR = Path(__file__).resolve().parent
if str(_EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPORT_DIR))

PHASES = ("stimulus", "delay", "go", "response")
V1_TASKS = ("PhonemeSequencing", "LexicalDelay")
DEFAULT_CONDITION = "Repeat"
CONDITIONS_BY_TASK = {
    "PhonemeSequencing": ["Repeat"],
    "LexicalDelay": ["Repeat", "Decision"],
}
SIGNIFICANCE_WINDOWS = {
    "stimulus": (0.0, 0.5),
    "delay": (0.0, 0.5),
    "go": (0.0, 0.5),
    "response": (-0.5, 0.5),
}
DISPLAY_WAVEFORM_RANGE = (-0.5, 1.0)
DEFAULT_MAX_TRACE_POINTS = 160
DEFAULT_VALIDATION_SUBJECTS = ("D0094", "D0071", "D0084")
SUPPORTED_ATLASES = ("aparc2009s", "hammers")
DEFAULT_ATLAS = "hammers"
ATLAS_LABELS = {
    "aparc2009s": "APARC",
    "hammers": "Hammersmith",
}

META_COLS = [
    "electrode_id",
    "subject",
    "channel",
    "roi",
    "label",
    "hemi",
    "x",
    "y",
    "z",
    "x_native",
    "y_native",
    "z_native",
    "x1_native",
    "y1_native",
    "z1_native",
    "x2_native",
    "y2_native",
    "z2_native",
    "x1_template",
    "y1_template",
    "z1_template",
    "x2_template",
    "y2_template",
    "z2_template",
    "contact_1",
    "contact_2",
    "contact_1_label",
    "contact_2_label",
    "mix",
]

ENDPOINT_FLOAT_COLS = [
    "x",
    "y",
    "z",
    "x_native",
    "y_native",
    "z_native",
    "x1_native",
    "y1_native",
    "z1_native",
    "x2_native",
    "y2_native",
    "z2_native",
    "x1_template",
    "y1_template",
    "z1_template",
    "x2_template",
    "y2_template",
    "z2_template",
]

TASK_DIR_RE = re.compile(r"^(.+)\(([^)]+)\)$")


def task_dir_name(task: str, reference: str, atlas: str) -> str:
    return f"{task}({reference})({atlas})"


def discover_subjects(
    input_root: Path,
    tasks: list[str],
    reference: str = "bipolar",
    atlas: str = DEFAULT_ATLAS,
) -> list[str]:
    """Union of subjects with packaged HGA in any configured task (not intersection)."""
    subjects: set[str] = set()
    for task in tasks:
        root = input_root / task_dir_name(task, reference, atlas)
        if not root.is_dir():
            continue
        for sub_dir in sorted(root.glob("sub-*")):
            if not sub_dir.is_dir():
                continue
            if any(sub_dir.glob("HGA/*_time.csv")):
                subjects.add(sub_dir.name.removeprefix("sub-"))
    return sorted(subjects)


def discover_tasks(
    input_root: Path,
    reference: str = "bipolar",
    atlas: str = DEFAULT_ATLAS,
) -> list[str]:
    atlases = SUPPORTED_ATLASES if atlas == "all" else (atlas,)
    tasks: set[str] = set()
    if not input_root.is_dir():
        return []
    for atlas_name in atlases:
        suffix = f"({reference})({atlas_name})"
        for path in sorted(input_root.iterdir()):
            if not path.is_dir() or not path.name.endswith(suffix):
                continue
            if not any(path.glob("sub-*/HGA/*_time.csv")):
                continue
            tasks.add(path.name[: -len(suffix)])
    return sorted(tasks)


def load_hga(
    input_root: Path,
    tasks: list[str],
    reference: str,
    atlas: str,
    subjects: list[str] | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    subject_set = set(subjects) if subjects else None

    for task in tasks:
        root = input_root / task_dir_name(task, reference, atlas)
        paths = sorted(root.glob("sub-*/HGA/*_time.csv"))
        if subject_set is not None:
            paths = [
                path
                for path in paths
                if any(f"sub-{subject}" in path.parts for subject in subject_set)
            ]
        if not paths:
            if subject_set is not None:
                continue
            raise FileNotFoundError(f"No HGA *_time.csv files found for task {task} under {root}")
        frames.extend(pd.read_csv(path) for path in paths)

    if not frames:
        raise FileNotFoundError(f"No HGA files found under {input_root}")
    return pd.concat(frames, ignore_index=True)


def region_id_for_flags(flags: dict[str, bool]) -> str:
    active = [phase for phase in PHASES if flags.get(phase, False)]
    return "_".join(active) if active else "none"


def region_label(region_id: str) -> str:
    if region_id == "none":
        return "No selected phase"
    return " ∩ ".join(part.capitalize() for part in region_id.split("_"))


def optional_float(value) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if pd.isna(value):
        return None
    return float(value)


def optional_str(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def filter_significance_window(df: pd.DataFrame) -> pd.DataFrame:
    """Rows inside each phase's significance window (fallback when mask is missing)."""
    pieces: list[pd.DataFrame] = []
    for phase, (tmin, tmax) in SIGNIFICANCE_WINDOWS.items():
        phase_df = df[df["phase"].astype(str).str.lower().eq(phase)]
        if phase_df.empty:
            continue
        pieces.append(phase_df[phase_df["time"].between(tmin, tmax)])
    if not pieces:
        return df.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)


def metrics_source_for_task(df: pd.DataFrame, task: str | None = None) -> pd.DataFrame:
    """Prefer masked rows; fall back to significance-window samples when mask is absent."""
    scoped = df if task is None else df[df["task"].astype(str).eq(task)]
    masked = scoped[scoped["mask"]]
    if not masked.empty:
        return masked
    if scoped.empty:
        return scoped
    return filter_significance_window(scoped)


def compute_phase_flags(df: pd.DataFrame) -> pd.DataFrame:
    source = metrics_source_for_task(df)
    index = pd.Index(df["electrode_id"].unique(), name="electrode_id")
    if source.empty:
        return pd.DataFrame(False, index=index, columns=list(PHASES))

    flags = (
        source.groupby(["electrode_id", "phase"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(PHASES), fill_value=0)
        .gt(0)
    )
    return flags.reindex(index, fill_value=False)


def compute_phase_flags_by_task(df: pd.DataFrame, tasks: list[str]) -> dict[str, pd.DataFrame]:
    by_task: dict[str, pd.DataFrame] = {}
    electrode_ids = pd.Index(df["electrode_id"].unique(), name="electrode_id")
    for task in tasks:
        source = metrics_source_for_task(df, task=task)
        if source.empty:
            by_task[task] = pd.DataFrame(False, index=electrode_ids, columns=list(PHASES))
            continue
        flags = (
            source.groupby(["electrode_id", "phase"], observed=True)
            .size()
            .unstack(fill_value=0)
            .reindex(columns=list(PHASES), fill_value=0)
            .gt(0)
        )
        by_task[task] = flags.reindex(electrode_ids, fill_value=False)
    return by_task


def compute_hga_by_task_condition(
    df: pd.DataFrame,
    tasks: list[str],
) -> dict[str, dict[str, pd.DataFrame]]:
    by_task: dict[str, dict[str, pd.DataFrame]] = {}
    electrode_ids = pd.Index(df["electrode_id"].unique(), name="electrode_id")
    for task in tasks:
        task_df = metrics_source_for_task(df, task=task)
        conditions = CONDITIONS_BY_TASK.get(task, [DEFAULT_CONDITION])
        by_task[task] = {}
        for condition in conditions:
            condition_df = task_df[task_df["description"].astype(str).eq(condition)]
            if condition_df.empty:
                by_task[task][condition] = pd.DataFrame(index=electrode_ids, columns=[task], dtype=float)
                continue
            grouped = (
                condition_df.groupby(["electrode_id", "task"], observed=True)["value"]
                .mean()
                .unstack(fill_value=np.nan)
            )
            by_task[task][condition] = grouped.reindex(columns=[task])
    return by_task


def compute_hga_by_task(
    df: pd.DataFrame,
    tasks: list[str],
    condition: str = DEFAULT_CONDITION,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for task in tasks:
        task_df = metrics_source_for_task(df, task=task)
        task_df = task_df[task_df["description"].astype(str).eq(condition)]
        if task_df.empty:
            continue
        grouped = (
            task_df.groupby(["electrode_id", "task"], observed=True)["value"]
            .mean()
            .reset_index()
        )
        pieces.append(grouped)
    index = pd.Index(df["electrode_id"].unique(), name="electrode_id")
    if not pieces:
        return pd.DataFrame(index=index, columns=list(tasks), dtype=float)

    combined = pd.concat(pieces, ignore_index=True)
    return (
        combined.groupby(["electrode_id", "task"], observed=True)["value"]
        .mean()
        .unstack(fill_value=np.nan)
        .reindex(columns=list(tasks))
    )


def compute_hga_size_scale(hga_by_task: pd.DataFrame) -> dict[str, float | str]:
    values: list[float] = []
    for task in hga_by_task.columns:
        values.extend(hga_by_task[task].dropna().abs().tolist())
    if not values:
        return {"vmin": 0.0, "vmax": 1.0, "method": "p95_abs_masked"}
    abs_values = np.asarray(values, dtype=np.float64)
    vmax = float(np.percentile(abs_values, 95))
    if vmax <= 0:
        vmax = float(abs_values.max()) if len(abs_values) else 1.0
    if vmax <= 0:
        vmax = 1.0
    return {"vmin": 0.0, "vmax": vmax, "method": "p95_abs_masked"}


def normalize_hga_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["phase"] = df["phase"].astype(str).str.lower()
    df["electrode_id"] = df["subject"].astype(str) + "|" + df["channel"].astype(str)
    df["mask"] = df["mask"].astype(bool)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    return df


def clip_display_window(df: pd.DataFrame) -> pd.DataFrame:
    tmin, tmax = DISPLAY_WAVEFORM_RANGE
    return df[df["time"].between(tmin, tmax)].copy()


def downsample_trace(trace: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(trace) <= max_points:
        return trace
    idx = np.linspace(0, len(trace) - 1, max_points).round().astype(int)
    return trace.iloc[np.unique(idx)]


def build_traces(
    df: pd.DataFrame,
    electrode_ids: set[str],
    tasks: list[str],
    max_trace_points: int = DEFAULT_MAX_TRACE_POINTS,
) -> dict:
    """Build trace bundles keyed by electrode -> task -> phase -> condition."""
    traces: dict = {}
    trace_df = clip_display_window(df[df["electrode_id"].isin(electrode_ids)].copy())
    grouped = (
        trace_df.groupby(
            ["electrode_id", "task", "phase", "description", "time"],
            observed=True,
        )["value"]
        .mean()
        .reset_index()
    )

    for (electrode_id, task, phase), task_phase_df in grouped.groupby(
        ["electrode_id", "task", "phase"],
        observed=True,
    ):
        if task not in tasks:
            continue
        traces.setdefault(electrode_id, {})
        traces[electrode_id].setdefault(task, {})
        traces[electrode_id][task].setdefault(phase, {})
        for condition, condition_df in task_phase_df.groupby("description", observed=True):
            condition_df = condition_df.sort_values("time")
            condition_df = downsample_trace(condition_df, max_points=max_trace_points)
            traces[electrode_id][task][phase][str(condition)] = {
                "time": [float(x) for x in condition_df["time"].to_numpy()],
                "value": [
                    None if pd.isna(x) else float(x) for x in condition_df["value"].to_numpy()
                ],
            }
    return traces


def split_traces_by_subject(traces: dict, electrodes: list[dict]) -> dict[str, dict]:
    subject_by_electrode = {item["id"]: item["subject"] for item in electrodes}
    by_subject: dict[str, dict] = {}
    for electrode_id, electrode_traces in traces.items():
        subject = subject_by_electrode.get(electrode_id)
        if subject is None:
            continue
        by_subject.setdefault(subject, {})[electrode_id] = electrode_traces
    return by_subject


def build_electrode_metadata(df: pd.DataFrame) -> pd.DataFrame:
    meta = (
        df.sort_values(["subject", "channel", "task", "phase"])
        .drop_duplicates("electrode_id")
        .reindex(columns=META_COLS)
        .copy()
    )
    meta = meta[meta["x"].notna() & meta["y"].notna() & meta["z"].notna()].copy()
    return meta


def build_payload(
    df: pd.DataFrame,
    tasks: list[str],
    subjects: list[str] | None,
    condition: str = DEFAULT_CONDITION,
    max_trace_points: int = DEFAULT_MAX_TRACE_POINTS,
    include_traces: bool = True,
) -> dict:
    df = normalize_hga_frame(df)

    phase_flags_df = compute_phase_flags(df)
    phase_flags_by_task = compute_phase_flags_by_task(df, tasks)
    hga_by_task = compute_hga_by_task(df, tasks, condition=condition)
    hga_by_task_condition = compute_hga_by_task_condition(df, tasks)
    meta = build_electrode_metadata(df)

    electrodes: list[dict] = []
    region_members: dict[str, list[str]] = {}

    for row in meta.itertuples(index=False):
        flags = {
            phase: bool(phase_flags_df.loc[row.electrode_id, phase])
            if row.electrode_id in phase_flags_df.index
            else False
            for phase in PHASES
        }
        task_phase_flags = {
            task: {
                phase: bool(phase_flags_by_task[task].loc[row.electrode_id, phase])
                if row.electrode_id in phase_flags_by_task[task].index
                else False
                for phase in PHASES
            }
            for task in tasks
        }
        active_phases = [phase for phase in PHASES if flags[phase]]
        rid = region_id_for_flags(flags)
        region_members.setdefault(rid, []).append(row.electrode_id)

        electrode: dict = {
            "id": row.electrode_id,
            "subject": row.subject,
            "channel": row.channel,
            "roi": optional_str(row.roi) or "Unknown",
            "label": optional_str(row.label) or "Unknown",
            "hemi": optional_str(row.hemi),
            "active_phases": active_phases,
            "phase_flags": flags,
            "phase_flags_by_task": task_phase_flags,
            "hga_by_task": {
                task: None
                if row.electrode_id not in hga_by_task.index
                or pd.isna(hga_by_task.loc[row.electrode_id].get(task))
                else float(hga_by_task.loc[row.electrode_id].get(task))
                for task in tasks
            },
            "hga_by_task_condition": {
                task: {
                    cond: None
                    if row.electrode_id not in hga_by_task_condition[task][cond].index
                    or pd.isna(hga_by_task_condition[task][cond].loc[row.electrode_id].get(task))
                    else float(hga_by_task_condition[task][cond].loc[row.electrode_id].get(task))
                    for cond in CONDITIONS_BY_TASK.get(task, [DEFAULT_CONDITION])
                }
                for task in tasks
            },
            "region_ids": [],
        }

        for col in ENDPOINT_FLOAT_COLS:
            electrode[col] = optional_float(getattr(row, col))
        for col in ("contact_1", "contact_2", "contact_1_label", "contact_2_label"):
            electrode[col] = optional_str(getattr(row, col))
        mix_value = getattr(row, "mix", None)
        if mix_value is not None and not pd.isna(mix_value):
            electrode["mix"] = bool(mix_value)

        electrodes.append(electrode)

    regions: list[dict] = []
    for rid, ids in sorted(region_members.items(), key=lambda item: (-len(item[1]), item[0])):
        if rid == "none":
            continue
        active = rid.split("_")
        regions.append(
            {
                "id": rid,
                "label": region_label(rid),
                "phases_on": active,
                "phases_off": [phase for phase in PHASES if phase not in active],
                "electrode_ids": sorted(ids),
                "count": len(ids),
            }
        )

    region_lookup = {region["id"]: region for region in regions}
    for electrode in electrodes:
        rid = region_id_for_flags(electrode["phase_flags"])
        electrode["region_ids"] = [rid] if rid in region_lookup else []

    exported_subjects = sorted({item["subject"] for item in electrodes})
    metadata = {
        "source": "results(nw)",
        "tasks": list(tasks),
        "conditions": {task: list(CONDITIONS_BY_TASK.get(task, [DEFAULT_CONDITION])) for task in tasks},
        "default_condition": condition,
        "phases": list(PHASES),
        "significance_windows": {phase: list(SIGNIFICANCE_WINDOWS[phase]) for phase in PHASES},
        "display_waveform_range": list(DISPLAY_WAVEFORM_RANGE),
        "subjects": exported_subjects,
        "n_electrodes": len(electrodes),
        "hga_size_scale": compute_hga_size_scale(hga_by_task),
    }

    electrode_ids = {item["id"] for item in electrodes}
    traces = (
        build_traces(df, electrode_ids, tasks, max_trace_points=max_trace_points)
        if include_traces
        else {}
    )

    return {
        "metadata": metadata,
        "electrodes": electrodes,
        "regions": regions,
        "traces": traces,
    }


def write_split_layout(
    payload: dict,
    output_dir: Path,
    include_trace_paths: bool = True,
    write_traces: bool = True,
    write_animation: bool = True,
    write_kde: bool = True,
) -> None:
    metadata = payload["metadata"]
    electrodes = payload["electrodes"]
    regions = payload.get("regions", [])
    subjects = metadata["subjects"]

    output_dir.mkdir(parents=True, exist_ok=True)
    if include_trace_paths:
        (output_dir / "traces").mkdir(exist_ok=True)
        (output_dir / "animation").mkdir(exist_ok=True)
        (output_dir / "kde" / "roi").mkdir(parents=True, exist_ok=True)

    traces = payload.get("traces", {})
    traces_by_subject = split_traces_by_subject(traces, electrodes) if write_traces else {}
    electrodes_by_subject: dict[str, list[dict]] = {}
    for electrode in electrodes:
        electrodes_by_subject.setdefault(electrode["subject"], []).append(electrode)

    files: dict[str, object] = {"electrodes": "electrodes.json"}
    if write_traces:
        files["traces"] = {subject: f"traces/{subject}.json" for subject in subjects}
    if write_animation:
        files["animation"] = {
            subject: {phase: f"animation/{subject}/{phase}.json" for phase in PHASES}
            for subject in subjects
        }
    if write_kde:
        files["kde_roi_mean"] = {
            subject: f"kde/roi/{subject}/mean.json" for subject in subjects
        }

    manifest = {
        "version": 1,
        "layout": "split",
        "metadata": metadata,
        "paths": {
            "electrodes": "electrodes.json",
            "traces": "traces/{subject}.json",
            "animation": "animation/{subject}/{phase}.json",
            "kde": "kde/roi/{subject}/mean.json",
            "template_mesh": "../assets/cvs_avg35_pial.glb",
            "native_mesh": "../assets/native/{subject}_pial.glb",
        },
    }
    if include_trace_paths:
        manifest["subjects"] = subjects
        manifest["files"] = files
        if write_animation:
            from hga_explorer_animation import animation_bundle_keys

            manifest["animation_selections"] = list(animation_bundle_keys(metadata["tasks"]))

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "electrodes.json").write_text(
        json.dumps({"electrodes": electrodes, "regions": regions}, indent=2),
        encoding="utf-8",
    )

    if write_traces:
        for subject in subjects:
            subject_traces = traces_by_subject.get(subject, {})
            (output_dir / "traces" / f"{subject}.json").write_text(
                json.dumps({"subject": subject, "traces": subject_traces}, indent=2),
                encoding="utf-8",
            )

    if write_animation and write_traces:
        from hga_explorer_animation import build_subject_phase_animation_bundle

        for subject in subjects:
            subject_electrodes = electrodes_by_subject.get(subject, [])
            electrode_ids = [item["id"] for item in subject_electrodes]
            subject_traces = traces_by_subject.get(subject, {})
            animation_dir = output_dir / "animation" / subject
            animation_dir.mkdir(parents=True, exist_ok=True)
            for phase in PHASES:
                bundle = build_subject_phase_animation_bundle(
                    electrode_ids,
                    subject_traces,
                    phase,
                    tasks=metadata["tasks"],
                )
                (animation_dir / f"{phase}.json").write_text(
                    json.dumps(bundle, indent=2),
                    encoding="utf-8",
                )

    if write_kde:
        from hga_explorer_kde import build_roi_mean_sources

        for subject in subjects:
            subject_electrodes = electrodes_by_subject.get(subject, [])
            kde_dir = output_dir / "kde" / "roi" / subject
            kde_dir.mkdir(parents=True, exist_ok=True)
            (kde_dir / "mean.json").write_text(
                json.dumps(
                    {"subject": subject, **build_roi_mean_sources(subject_electrodes)},
                    indent=2,
                ),
                encoding="utf-8",
            )


def write_atlas_electrodes_and_kde(
    payload: dict,
    atlas_dir: Path,
    write_kde: bool = True,
) -> dict[str, object]:
    """Write per-atlas electrodes.json and optional KDE bundles."""
    from hga_explorer_kde import build_roi_mean_sources

    atlas_dir.mkdir(parents=True, exist_ok=True)
    electrodes = payload["electrodes"]
    regions = payload.get("regions", [])
    subjects = payload["metadata"]["subjects"]

    (atlas_dir / "electrodes.json").write_text(
        json.dumps({"electrodes": electrodes, "regions": regions}, indent=2),
        encoding="utf-8",
    )

    files: dict[str, object] = {"electrodes": "electrodes.json"}
    if write_kde:
        electrodes_by_subject: dict[str, list[dict]] = {}
        for electrode in electrodes:
            electrodes_by_subject.setdefault(electrode["subject"], []).append(electrode)
        kde_files: dict[str, str] = {}
        for subject in subjects:
            subject_electrodes = electrodes_by_subject.get(subject, [])
            kde_dir = atlas_dir / "kde" / "roi" / subject
            kde_dir.mkdir(parents=True, exist_ok=True)
            rel_path = f"kde/roi/{subject}/mean.json"
            (kde_dir / "mean.json").write_text(
                json.dumps(
                    {"subject": subject, **build_roi_mean_sources(subject_electrodes)},
                    indent=2,
                ),
                encoding="utf-8",
            )
            kde_files[subject] = rel_path
        files["kde_roi_mean"] = kde_files

    return files


def write_shared_traces_and_animation(
    payload: dict,
    shared_dir: Path,
    write_traces: bool = True,
    write_animation: bool = True,
) -> dict[str, object]:
    """Write atlas-independent traces and animation under shared/."""
    from hga_explorer_animation import animation_bundle_keys, build_subject_phase_animation_bundle

    metadata = payload["metadata"]
    electrodes = payload["electrodes"]
    subjects = metadata["subjects"]
    shared_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, object] = {}
    traces = payload.get("traces", {})
    traces_by_subject = split_traces_by_subject(traces, electrodes) if write_traces else {}
    electrodes_by_subject: dict[str, list[dict]] = {}
    for electrode in electrodes:
        electrodes_by_subject.setdefault(electrode["subject"], []).append(electrode)

    if write_traces:
        traces_dir = shared_dir / "traces"
        traces_dir.mkdir(exist_ok=True)
        trace_files: dict[str, str] = {}
        for subject in subjects:
            subject_traces = traces_by_subject.get(subject, {})
            rel_path = f"traces/{subject}.json"
            (traces_dir / f"{subject}.json").write_text(
                json.dumps({"subject": subject, "traces": subject_traces}, indent=2),
                encoding="utf-8",
            )
            trace_files[subject] = rel_path
        files["traces"] = trace_files

    if write_animation and write_traces:
        animation_files: dict[str, dict[str, str]] = {}
        for subject in subjects:
            subject_electrodes = electrodes_by_subject.get(subject, [])
            electrode_ids = [item["id"] for item in subject_electrodes]
            subject_traces = traces_by_subject.get(subject, {})
            animation_dir = shared_dir / "animation" / subject
            animation_dir.mkdir(parents=True, exist_ok=True)
            animation_files[subject] = {}
            for phase in PHASES:
                bundle = build_subject_phase_animation_bundle(
                    electrode_ids,
                    subject_traces,
                    phase,
                    tasks=metadata["tasks"],
                )
                rel_path = f"animation/{subject}/{phase}.json"
                (animation_dir / f"{phase}.json").write_text(
                    json.dumps(bundle, indent=2),
                    encoding="utf-8",
                )
                animation_files[subject][phase] = rel_path
        files["animation"] = animation_files

    return files, animation_bundle_keys(metadata["tasks"]) if write_animation else []


def write_multi_atlas_layout(
    atlas_payloads: dict[str, dict],
    shared_payload: dict,
    output_dir: Path,
    default_atlas: str = DEFAULT_ATLAS,
    write_traces: bool = True,
    write_animation: bool = True,
    write_kde: bool = True,
) -> None:
    """Write manifest v2 with shared traces/animation and per-atlas electrodes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = output_dir / "shared"

    shared_files, animation_selections = write_shared_traces_and_animation(
        shared_payload,
        shared_dir,
        write_traces=write_traces,
        write_animation=write_animation,
    )

    atlas_entries: dict[str, dict] = {}
    for atlas_id, payload in atlas_payloads.items():
        atlas_dir = output_dir / "atlas" / atlas_id
        local_files = write_atlas_electrodes_and_kde(
            payload,
            atlas_dir,
            write_kde=write_kde,
        )
        atlas_entries[atlas_id] = {
            "label": ATLAS_LABELS.get(atlas_id, atlas_id),
            "metadata": {
                "n_electrodes": payload["metadata"]["n_electrodes"],
                "hga_size_scale": payload["metadata"]["hga_size_scale"],
            },
            "files": {
                key: f"atlas/{atlas_id}/{value}"
                if isinstance(value, str)
                else {
                    subject: f"atlas/{atlas_id}/{path}"
                    for subject, path in value.items()
                }
                for key, value in local_files.items()
            },
        }

    shared_metadata = {k: v for k, v in shared_payload["metadata"].items()}
    subjects = shared_metadata["subjects"]

    manifest = {
        "version": 2,
        "layout": "split-multi-atlas",
        "default_atlas": default_atlas,
        "atlases": list(atlas_payloads.keys()),
        "metadata": shared_metadata,
        "shared": {
            "files": {
                key: (
                    {subject: f"shared/{path}" for subject, path in value.items()}
                    if isinstance(value, dict) and key != "animation"
                    else {
                        subject: {
                            phase: f"shared/{path}"
                            for phase, path in phases.items()
                        }
                        for subject, phases in value.items()
                    }
                    if key == "animation"
                    else value
                )
                for key, value in shared_files.items()
            },
        },
        "atlas": atlas_entries,
        "paths": {
            "electrodes": "atlas/{atlas}/electrodes.json",
            "traces": "shared/traces/{subject}.json",
            "animation": "shared/animation/{subject}/{phase}.json",
            "kde": "atlas/{atlas}/kde/roi/{subject}/mean.json",
            "template_mesh": "../assets/cvs_avg35_pial.glb",
            "native_mesh": "../assets/native/{subject}_pial.glb",
        },
        "subjects": subjects,
    }
    if write_animation and animation_selections:
        manifest["animation_selections"] = list(animation_selections)

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def validate_payload(payload: dict, tasks: list[str]) -> list[str]:
    issues: list[str] = []
    electrodes = payload["electrodes"]
    required_fields = {
        "id",
        "subject",
        "channel",
        "roi",
        "label",
        "hemi",
        "active_phases",
        "phase_flags",
        "phase_flags_by_task",
        "hga_by_task",
        "hga_by_task_condition",
        "region_ids",
        "x",
        "y",
        "z",
        "x_native",
        "y_native",
        "z_native",
        "x1_native",
        "y1_native",
        "z1_native",
        "x2_native",
        "y2_native",
        "z2_native",
        "x1_template",
        "y1_template",
        "z1_template",
        "x2_template",
        "y2_template",
        "z2_template",
        "contact_1",
        "contact_2",
        "contact_1_label",
        "contact_2_label",
    }

    for electrode in electrodes:
        missing = required_fields - set(electrode)
        if missing:
            issues.append(f"{electrode['id']} missing fields: {sorted(missing)}")
        if any(key.startswith("maper_") for key in electrode):
            issues.append(f"{electrode['id']} contains maper_* fields")
        if set(electrode.get("hga_by_task", {})) != set(tasks):
            issues.append(f"{electrode['id']} hga_by_task keys mismatch")
        if set(electrode.get("phase_flags", {})) != set(PHASES):
            issues.append(f"{electrode['id']} phase_flags keys mismatch")

    return issues


def validate_traces(
    traces: dict,
    tasks: list[str],
    display_range: tuple[float, float] = DISPLAY_WAVEFORM_RANGE,
) -> list[str]:
    issues: list[str] = []
    tmin, tmax = display_range

    for electrode_id, task_traces in traces.items():
        for task, phase_traces in task_traces.items():
            if task not in tasks:
                issues.append(f"{electrode_id}: unexpected task {task}")
            for phase, condition_traces in phase_traces.items():
                if phase not in PHASES:
                    issues.append(f"{electrode_id}/{task}: unexpected phase {phase}")
                for condition, payload in condition_traces.items():
                    times = payload.get("time") or []
                    values = payload.get("value") or []
                    if len(times) != len(values):
                        issues.append(f"{electrode_id}/{task}/{phase}/{condition}: time/value length mismatch")
                    if times and (min(times) < tmin - 1e-6 or max(times) > tmax + 1e-6):
                        issues.append(
                            f"{electrode_id}/{task}/{phase}/{condition}: "
                            f"time out of display range [{tmin}, {tmax}]"
                        )
    return issues


def validate_animation_files(
    output_dir: Path,
    subjects: list[str],
    *,
    shared: bool = False,
) -> list[str]:
    issues: list[str] = []
    base = output_dir / "shared" if shared else output_dir
    for subject in subjects:
        for phase in PHASES:
            path = base / "animation" / subject / f"{phase}.json"
            if not path.is_file():
                issues.append(f"missing animation file: {path.relative_to(output_dir)}")
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            bundles = payload.get("bundles") or {}
            if not bundles:
                issues.append(f"empty animation bundles: {path.relative_to(output_dir)}")
    return issues


def validate_kde_files(
    output_dir: Path,
    subjects: list[str],
    *,
    atlas: str | None = None,
) -> list[str]:
    issues: list[str] = []
    base = output_dir / "atlas" / atlas if atlas else output_dir
    for subject in subjects:
        path = base / "kde" / "roi" / subject / "mean.json"
        if not path.is_file():
            issues.append(f"missing kde file: {path.relative_to(output_dir)}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("sources"):
            issues.append(f"empty kde sources: {path.relative_to(output_dir)}")
    return issues


def resolve_tasks(
    input_root: Path,
    reference: str,
    tasks: list[str] | None,
    atlas: str = DEFAULT_ATLAS,
) -> list[str]:
    discovered = discover_tasks(input_root, reference=reference, atlas=atlas)
    if tasks:
        missing = [task for task in tasks if task not in discovered]
        if missing:
            raise FileNotFoundError(
                f"Requested tasks not found under {input_root}: {', '.join(missing)}"
            )
        return list(tasks)
    if not discovered:
        raise FileNotFoundError(f"No packaged HGA task folders found under {input_root}")
    return discovered


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HGA Explorer electrodes + manifest JSON.")
    parser.add_argument("--input_root", default=DEFAULT_INPUT_ROOT, type=Path)
    parser.add_argument("--reference", default="bipolar")
    parser.add_argument(
        "--atlas",
        default="all",
        choices=[*SUPPORTED_ATLASES, "all"],
        help="Parcellation atlas (aparc2009s, hammers, or all for dual-atlas export)",
    )
    parser.add_argument(
        "--trace_atlas",
        default=DEFAULT_ATLAS,
        choices=SUPPORTED_ATLASES,
        help="Atlas used for shared traces/animation when --atlas all",
    )
    parser.add_argument(
        "--default_atlas",
        default=DEFAULT_ATLAS,
        choices=SUPPORTED_ATLASES,
        help="Default atlas in manifest v2 UI",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=list(V1_TASKS),
        help="Task names without reference suffix, e.g. PhonemeSequencing LexicalDelay",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Explicit subject IDs (default: validation cohort D0094 D0071 D0084)",
    )
    parser.add_argument(
        "--all-subjects",
        action="store_true",
        help="Export every subject present in any configured task (union across tasks)",
    )
    parser.add_argument("--condition", default=DEFAULT_CONDITION)
    parser.add_argument("--max_trace_points", type=int, default=DEFAULT_MAX_TRACE_POINTS)
    parser.add_argument("--skip_traces", action="store_true")
    parser.add_argument("--skip_animation", action="store_true")
    parser.add_argument("--skip_kde", action="store_true")
    parser.add_argument("--output_dir", default=DEFAULT_DATA_DIR, type=Path)
    args = parser.parse_args()

    write_traces = not args.skip_traces
    write_animation = not args.skip_animation and write_traces
    write_kde = not args.skip_kde

    atlas_list = list(SUPPORTED_ATLASES) if args.atlas == "all" else [args.atlas]
    discovery_atlas = atlas_list[0]
    tasks = resolve_tasks(args.input_root, args.reference, args.tasks, atlas=discovery_atlas)

    if args.all_subjects and args.subjects:
        raise RuntimeError("Use either --all-subjects or --subjects, not both")
    if args.all_subjects:
        args.subjects = discover_subjects(
            args.input_root, tasks, reference=args.reference, atlas=discovery_atlas
        )
        if not args.subjects:
            raise FileNotFoundError(
                f"No subjects found for tasks {tasks} under {args.input_root}"
            )
    elif not args.subjects:
        args.subjects = list(DEFAULT_VALIDATION_SUBJECTS)

    atlas_payloads: dict[str, dict] = {}
    all_issues: list[str] = []
    for atlas_name in atlas_list:
        df = load_hga(
            args.input_root,
            tasks,
            args.reference,
            atlas_name,
            subjects=args.subjects,
        )
        payload = build_payload(
            df,
            tasks=tasks,
            subjects=args.subjects,
            condition=args.condition,
            max_trace_points=args.max_trace_points,
            include_traces=write_traces and len(atlas_list) == 1,
        )
        payload["metadata"]["atlas"] = atlas_name
        atlas_issues = validate_payload(payload, tasks=tasks)
        if write_traces and len(atlas_list) == 1:
            atlas_issues.extend(validate_traces(payload.get("traces", {}), tasks=tasks))
        if atlas_issues:
            all_issues.extend(f"[{atlas_name}] {issue}" for issue in atlas_issues)
        atlas_payloads[atlas_name] = payload

    if all_issues:
        raise RuntimeError("Export validation failed:\n" + "\n".join(all_issues[:30]))

    if len(atlas_list) > 1:
        trace_atlas = args.trace_atlas
        if trace_atlas not in atlas_payloads:
            trace_atlas = atlas_list[0]
        shared_df = load_hga(
            args.input_root,
            tasks,
            args.reference,
            trace_atlas,
            subjects=args.subjects,
        )
        shared_payload = build_payload(
            shared_df,
            tasks=tasks,
            subjects=args.subjects,
            condition=args.condition,
            max_trace_points=args.max_trace_points,
            include_traces=write_traces,
        )
        trace_issues = validate_traces(shared_payload.get("traces", {}), tasks=tasks) if write_traces else []
        if trace_issues:
            raise RuntimeError("Shared trace validation failed:\n" + "\n".join(trace_issues[:20]))

        write_multi_atlas_layout(
            atlas_payloads,
            shared_payload,
            args.output_dir,
            default_atlas=args.default_atlas,
            write_traces=write_traces,
            write_animation=write_animation,
            write_kde=write_kde,
        )

        subjects = shared_payload["metadata"]["subjects"]
        if write_animation:
            all_issues.extend(
                validate_animation_files(args.output_dir, subjects, shared=True)
            )
        if write_kde:
            for atlas_name in atlas_list:
                all_issues.extend(
                    validate_kde_files(args.output_dir, subjects, atlas=atlas_name)
                )
        metadata = shared_payload["metadata"]
        trace_electrodes = len(shared_payload.get("traces", {}))
    else:
        payload = atlas_payloads[atlas_list[0]]
        write_split_layout(
            payload,
            args.output_dir,
            write_traces=write_traces,
            write_animation=write_animation,
            write_kde=write_kde,
        )
        subjects = payload["metadata"]["subjects"]
        if write_animation:
            all_issues.extend(validate_animation_files(args.output_dir, subjects))
        if write_kde:
            all_issues.extend(validate_kde_files(args.output_dir, subjects))

        payload = atlas_payloads[atlas_list[0]]
        metadata = payload["metadata"]
        trace_electrodes = len(payload.get("traces", {}))

    if all_issues:
        raise RuntimeError("Post-write validation failed:\n" + "\n".join(all_issues[:20]))

    print(
        "Wrote HGA Explorer data:",
        f"atlases={','.join(atlas_list)}",
        f"default_atlas={args.default_atlas if len(atlas_list) > 1 else atlas_list[0]}",
        f"subjects={len(metadata['subjects'])}",
        f"trace_electrodes={trace_electrodes}",
        f"animation={'yes' if write_animation else 'no'}",
        f"kde={'yes' if write_kde else 'no'}",
        f"tasks={','.join(tasks)}",
        f"output={args.output_dir}",
    )
    for atlas_name, payload in atlas_payloads.items():
        print(f"  {atlas_name}: electrodes={payload['metadata']['n_electrodes']}")


if __name__ == "__main__":
    main()
