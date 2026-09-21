#!/usr/bin/env python3
"""Validate functional decoding result census (96 window + 96 resolved)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
from mne_bids import BIDSPath
import numpy as np

from src.decoding.prepare_functional_decoding_dataset import (
    LEXICAL_CONDITIONS,
    LEXICAL_FEATURES,
    PHASES,
    PHONEME_SEQUENCE_FEATURES,
    PSEUDO_SUBJECTS,
)
from src.paths import decoding_results_dir, resolve_decoding_task_root


PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Bilateral pools: no allowlisted empty cells.
ALLOWED_MISSING_RESULTS: set[tuple[str, str, str, str, str]] = set()


def result_path(
    results_root: Path,
    bids_task: str,
    pseudo_subject: str,
    datatype: str,
    phase: str,
    description: str,
    *,
    recording: str | None = None,
) -> Path:
    task_root = resolve_decoding_task_root(results_root, bids_task)
    if task_root is None:
        # Prefer canonical method-first layout when nothing exists yet.
        task_root = Path(results_root) / bids_task
        if task_root.parent.name != "decoding" and (Path(results_root) / "decoding").is_dir():
            task_root = Path(results_root) / "decoding" / bids_task
    path = BIDSPath(
        root=str(task_root),
        subject=pseudo_subject,
        datatype=datatype,
        processing=phase,
        description=description,
        recording=recording,
        suffix="highgamma",
        extension=".h5",
        check=False,
    )
    return Path(path.fpath)


def inspect(
    path: Path,
    datasets: tuple[str, ...],
    attrs: dict[str, int | float],
) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as stream:
        missing = [name for name in datasets if name not in stream]
        if missing:
            raise ValueError(f"{path}: missing datasets {missing}")
        for name, expected in attrs.items():
            if name not in stream.attrs:
                raise ValueError(f"{path}: missing attr {name}")
            actual = stream.attrs[name]
            if isinstance(expected, float):
                if not np.isclose(float(actual), expected):
                    raise ValueError(f"{path}: {name}={actual}, expected {expected}")
            elif int(actual) != int(expected):
                raise ValueError(f"{path}: {name}={actual}, expected {expected}")
        return {
            "path": str(path),
            "datasets": {name: list(stream[name].shape) for name in datasets},
        }


def validate(results_root: Path) -> dict[str, object]:
    window = []
    resolved = []
    skipped = []
    for bids_task, features, descriptions in (
        ("LexicalDelay", LEXICAL_FEATURES, LEXICAL_CONDITIONS),
        ("PhonemeSequence", PHONEME_SEQUENCE_FEATURES, ("Repeat",)),
    ):
        recording = "1" if bids_task == "PhonemeSequence" else None
        for subject in PSEUDO_SUBJECTS:
            for description in descriptions:
                for feature in features:
                    for phase in PHASES:
                        key = (bids_task, subject, phase, feature, description)
                        win_path = result_path(
                            results_root,
                            bids_task,
                            subject,
                            f"(decode){feature}",
                            phase,
                            description,
                            recording=recording,
                        )
                        res_path = result_path(
                            results_root,
                            bids_task,
                            subject,
                            f"(decode)(resolved){feature}",
                            phase,
                            description,
                            recording=recording,
                        )
                        if key in ALLOWED_MISSING_RESULTS:
                            if win_path.exists() or res_path.exists():
                                raise ValueError(
                                    f"Allowlisted skip unexpectedly present: {key}"
                                )
                            skipped.append({"key": list(key)})
                            continue
                        win_attrs = {
                            "n_perm": 200,
                            "n_folds": 5,
                            "n_repeats": 30,
                        }
                        # Lexicality window lock uses PCA 0.95; others keep 0.80.
                        win_attrs["variance"] = 0.95 if feature == "lexicality" else 0.80
                        window.append(
                            inspect(
                                win_path,
                                ("accuracy", "perm_scores", "p_value", "accuracy_repeats"),
                                win_attrs,
                            )
                        )
                        resolved.append(
                            inspect(
                                res_path,
                                (
                                    "accuracy",
                                    "baseline",
                                    "time",
                                    "mask",
                                    "accuracy_repeats",
                                ),
                                {"n_perm": 200, "n_folds": 5, "n_repeats": 1},
                            )
                        )
    # LexicalDelay: 3×2×3×4 = 72; PhonemeSequence: 3×1×2×4 = 24 → 96 each.
    if (len(window), len(resolved), len(skipped)) != (96, 96, 0):
        raise ValueError(
            f"Unexpected census: window={len(window)}, resolved={len(resolved)}, "
            f"skipped={len(skipped)}"
        )
    return {
        "window": len(window),
        "resolved": len(resolved),
        "skipped_empty_cells": len(skipped),
        "files": {
            "window": window,
            "resolved": resolved,
            "skipped_empty_cells": skipped,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=decoding_results_dir(),
        help="Decoding scores root (default: results/decoding)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "results" / "decoding_functional" / "result_census.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate(args.results_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(f".{args.report.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True))
    temporary.replace(args.report)
    print("Validated 96 window + 96 resolved (bilateral Sensory/Sustain/Motor)")


if __name__ == "__main__":
    main()
