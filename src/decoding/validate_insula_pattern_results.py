#!/usr/bin/env python3
"""Census for INS whole-window Haufe pattern outputs (64 expected H5 files)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
from mne_bids import BIDSPath
import numpy as np

from src.decoding.run_decoding_patterns import pattern_datatype
from src.paths import decoding_results_dir, decoding_task_dir

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PSEUDO_SUBJECTS = ("INSl", "INSr")
PHASES = ("Stimulus", "Delay", "Go", "Response")
BAND = "highgamma"
REF = "bipolar"

REQUIRED_DATASETS = ("pattern", "pattern_mask", "pattern_p_values", "times", "channel")
REQUIRED_ATTRS = (
    "source_path",
    "method",
    "datatype",
    "phase",
    "description",
    "n_perm",
    "n_folds",
    "variance",
    "class_strategy",
)


def expected_pattern_path(
    bids_task: str,
    subject: str,
    datatype: str,
    phase: str,
    description: str,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    recording = "1" if bids_task == "PhonemeSequence" else None
    results_root = (
        decoding_task_dir(bids_task)
        if project_root.resolve() == PROJECT_ROOT.resolve()
        else project_root / "results" / "decoding" / bids_task
    )
    path = BIDSPath(
        root=str(results_root),
        datatype=pattern_datatype(datatype),
        subject=subject,
        task=bids_task,
        suffix=BAND,
        processing=phase,
        recording=recording,
        description=description,
        extension=".h5",
        check=False,
    )
    return Path(path.fpath)


def build_expected_jobs() -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    for subject in PSEUDO_SUBJECTS:
        for description in ("Repeat", "Decision"):
            for datatype in ("phoneme", "articulator", "lexicality"):
                for phase in PHASES:
                    jobs.append(
                        {
                            "bids_task": "LexicalDelay",
                            "subject": subject,
                            "datatype": datatype,
                            "phase": phase,
                            "description": description,
                        }
                    )
    for subject in PSEUDO_SUBJECTS:
        for datatype in ("phoneme", "articulator"):
            for phase in PHASES:
                jobs.append(
                    {
                        "bids_task": "PhonemeSequence",
                        "subject": subject,
                        "datatype": datatype,
                        "phase": phase,
                        "description": "Repeat",
                    }
                )
    return jobs


def inspect_pattern_file(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as stream:
        missing_datasets = [name for name in REQUIRED_DATASETS if name not in stream]
        missing_attrs = [name for name in REQUIRED_ATTRS if name not in stream.attrs]
        if missing_datasets or missing_attrs:
            raise ValueError(
                f"{path}: missing datasets={missing_datasets}, attrs={missing_attrs}"
            )
        pattern = stream["pattern"][()]
        times = stream["times"][()]
        channels = stream["channel"][()]
        n_channels = len(channels)
        n_times = len(times)
        if pattern.ndim == 2:
            if pattern.shape != (n_channels, n_times):
                raise ValueError(
                    f"{path}: pattern shape {pattern.shape} != ({n_channels}, {n_times})"
                )
        elif pattern.ndim == 3:
            if pattern.shape[1:] != (n_channels, n_times):
                raise ValueError(
                    f"{path}: pattern shape {pattern.shape} != (n_class, {n_channels}, {n_times})"
                )
        else:
            raise ValueError(f"{path}: unexpected pattern ndim={pattern.ndim}")
        return {
            "path": str(path),
            "pattern_shape": list(pattern.shape),
            "n_channels": n_channels,
            "n_times": n_times,
            "class_strategy": stream.attrs["class_strategy"],
            "n_perm": int(stream.attrs["n_perm"]),
        }


def run_census(*, strict: bool = True) -> dict[str, object]:
    jobs = build_expected_jobs()
    by_task: dict[str, list[dict[str, object]]] = {
        "LexicalDelay": [],
        "PhonemeSequence": [],
    }
    missing: list[str] = []
    errors: list[str] = []

    for job in jobs:
        path = expected_pattern_path(
            job["bids_task"],
            job["subject"],
            job["datatype"],
            job["phase"],
            job["description"],
        )
        if not path.exists():
            missing.append(str(path))
            continue
        try:
            record = {**job, **inspect_pattern_file(path)}
            by_task[job["bids_task"]].append(record)
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"{path}: {exc}")

    report = {
        "expected_total": len(jobs),
        "found_total": sum(len(v) for v in by_task.values()),
        "missing_total": len(missing),
        "error_total": len(errors),
        "by_task": {
            task: {
                "expected": 48 if task == "LexicalDelay" else 16,
                "found": len(records),
            }
            for task, records in by_task.items()
        },
        "missing_paths": missing,
        "errors": errors,
        "records": by_task,
        "ok": not missing and not errors,
    }
    if strict and not report["ok"]:
        raise SystemExit(
            f"Pattern census failed: missing={len(missing)} errors={len(errors)}"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=decoding_results_dir() / "pattern_census.json",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Write report even when files are missing or invalid.",
    )
    args = parser.parse_args()
    report = run_census(strict=not args.no_strict)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["by_task"], indent=2))
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
