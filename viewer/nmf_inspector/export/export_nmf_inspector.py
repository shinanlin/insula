#!/usr/bin/env python3
"""Build NMF Inspector JSON from channel assignments + packaged HGA (results/hga/)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from mne_bids import BIDSPath

VIEWER_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = VIEWER_ROOT.parent.parent
DEFAULT_DATA_DIR = VIEWER_ROOT / "public" / "data"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import hga_results_dir, nmf_assignments_path  # noqa: E402

TASKS = (
    "LexicalDelay",
    "PhonemeSequence",
    "PictureNaming",
    "SentenceRep",
)
PHASES = ("Stimulus", "Delay", "Go", "Response")
DESCRIPTION = "Repeat"
MODALITY = "sound"
MAX_TRACE_POINTS = 160
LOADING_COLS = (
    "loading_sustain",
    "loading_motor",
    "loading_sensory",
)
CLUSTERS = ("sustain", "motor", "sensory")
NMF_K = 3


def downsample_trace(time: np.ndarray, value: np.ndarray, max_points: int) -> tuple[list, list]:
    if len(time) <= max_points:
        return time.tolist(), value.tolist()
    idx = np.linspace(0, len(time) - 1, max_points, dtype=int)
    return time[idx].tolist(), value[idx].tolist()


def load_assignments(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"channel", "subject", "functional_cluster", "x", "y", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"assignments missing columns: {sorted(missing)}")
    return df


def _hammers_label(value) -> str | None:
    text = str(value).strip()
    if not text or text == "0":
        return None
    return text


def discover_hga_paths(task: str) -> list[BIDSPath]:
    root = hga_results_dir(task)
    if not root.is_dir():
        return []
    return list(
        BIDSPath(
            root=str(root),
            datatype="HGA",
            suffix="time",
            description=DESCRIPTION,
            check=False,
        ).match()
    )


def export_traces(
    channels: set[str],
    max_points: int,
) -> tuple[dict[str, dict], dict[str, list[str]], dict[str, str]]:
    """Return per-subject trace trees, channel→tasks map, and Hammersmith labels."""
    traces_by_subject: dict[str, dict] = {}
    tasks_by_channel: dict[str, set[str]] = {ch: set() for ch in channels}
    labels_by_channel: dict[str, str] = {}

    for task in TASKS:
        for bids_path in discover_hga_paths(task):
            csv_path = Path(bids_path.fpath)
            if not csv_path.is_file():
                continue
            phase = bids_path.processing or ""
            if phase not in PHASES:
                continue
            frame = pd.read_csv(
                csv_path,
                usecols=["time", "channel", "value", "modality", "label"],
            )
            frame = frame[frame["channel"].isin(channels)]
            frame = frame[frame["modality"].astype(str).eq(MODALITY)]
            if frame.empty:
                continue
            subject = bids_path.subject
            if subject not in traces_by_subject:
                traces_by_subject[subject] = {}
            subj_tree = traces_by_subject[subject]
            for channel, grp in frame.groupby("channel", sort=True):
                if channel not in labels_by_channel:
                    for raw in grp["label"].dropna().unique():
                        parsed = _hammers_label(raw)
                        if parsed is not None:
                            labels_by_channel[channel] = parsed
                            break
                tasks_by_channel[channel].add(task)
                ch_tree = subj_tree.setdefault(channel, {})
                task_tree = ch_tree.setdefault(task, {})
                times = grp["time"].to_numpy(dtype=float)
                values = grp["value"].to_numpy(dtype=float)
                t_out, v_out = downsample_trace(times, values, max_points)
                task_tree[phase] = {"time": t_out, "value": v_out}

    tasks_map = {ch: sorted(tasks) for ch, tasks in tasks_by_channel.items() if tasks}
    return traces_by_subject, tasks_map, labels_by_channel


def build_electrodes(
    df: pd.DataFrame,
    tasks_map: dict[str, list[str]],
    labels_map: dict[str, str],
) -> list[dict]:
    electrodes = []
    for row in df.itertuples(index=False):
        channel = str(row.channel)
        roi = str(getattr(row, "roi", ""))
        loadings = {
            col.removeprefix("loading_"): float(getattr(row, col))
            for col in LOADING_COLS
            if hasattr(row, col)
        }
        electrodes.append(
            {
                "id": channel,
                "channel": channel,
                "subject": str(row.subject),
                "roi": roi,
                "label": labels_map.get(channel, roi),
                "hemi": str(getattr(row, "hemi", "")),
                "x": float(row.x),
                "y": float(row.y),
                "z": float(row.z),
                "component": int(getattr(row, "component", -1)),
                "functional_cluster": str(row.functional_cluster),
                "loadings": loadings,
                "dominance": float(getattr(row, "dominance", np.nan))
                if hasattr(row, "dominance")
                else None,
                "tasks": tasks_map.get(channel, []),
            }
        )
    return electrodes


def main(
    assignments_path: Path | None = None,
    output_dir: Path = DEFAULT_DATA_DIR,
    max_trace_points: int = MAX_TRACE_POINTS,
) -> None:
    assignments_path = assignments_path or nmf_assignments_path()
    output_dir = Path(output_dir)
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    df = load_assignments(assignments_path)
    channels = set(df["channel"].astype(str))
    traces_by_subject, tasks_map, labels_map = export_traces(channels, max_trace_points)
    electrodes = build_electrodes(df, tasks_map, labels_map)

    for stale in traces_dir.glob("*.json"):
        stale.unlink()
    for subject, tree in sorted(traces_by_subject.items()):
        (traces_dir / f"{subject}.json").write_text(json.dumps(tree))

    manifest = {
        "metadata": {
            "assignments_source": str(assignments_path.resolve()),
            "subjects": sorted(traces_by_subject.keys()),
            "tasks": list(TASKS),
            "phases": list(PHASES),
            "clusters": list(CLUSTERS),
            "k": NMF_K,
            "condition": DESCRIPTION,
            "modality": MODALITY,
            "n_electrodes": len(electrodes),
        }
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output_dir / "electrodes.json").write_text(json.dumps(electrodes, indent=2))

    print(f"Exported {len(electrodes)} electrodes → {output_dir}")
    print(f"  manifest.json, electrodes.json, traces/ ({len(traces_by_subject)} subjects)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assignments",
        type=Path,
        default=None,
        help="NMF channel assignments CSV (default: results/nmf/channel_assignments.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )
    parser.add_argument("--max_trace_points", type=int, default=MAX_TRACE_POINTS)
    args = parser.parse_args()
    main(
        assignments_path=args.assignments,
        output_dir=args.output_dir,
        max_trace_points=args.max_trace_points,
    )
