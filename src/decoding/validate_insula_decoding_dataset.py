#!/usr/bin/env python3
"""Validate merged INS decoding inputs and emit a machine-readable census."""

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
)
from src.decoding.prepare_insula_decoding_dataset import (
    LEXICAL_NO_DELAY_PHASES,
    PSEUDO_SUBJECTS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_DATASETS = ("X", "y", "trial", "condition", "label", "channel", "time")
REQUIRED_ATTRS = (
    "assignment_sha256",
    "assignment_rows",
    "functional_cluster",
    "hemi",
    "pseudo_subject",
    "channel_selection",
    "input_datatype",
    "significance_datatype",
    "dataset_kind",
    "candidate_channels_json",
    "retained_channels_json",
    "n_candidate_channels",
    "n_retained_channels",
)


def decode_strings(values) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def input_path(
    root: Path,
    bids_task: str,
    pseudo_subject: str,
    feature: str,
    phase: str,
    description: str,
    band: str = "highgamma",
) -> Path:
    path = BIDSPath(
        root=str(root),
        subject=pseudo_subject,
        task=bids_task,
        processing=phase,
        recording="1" if bids_task == "PhonemeSequence" else None,
        description=description,
        datatype=feature,
        suffix=band,
        extension=".h5",
        check=False,
    )
    return Path(path.fpath)


def inspect_file(path: Path, expected_kind: str = "regular") -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as stream:
        missing_datasets = [name for name in REQUIRED_DATASETS if name not in stream]
        missing_attrs = [name for name in REQUIRED_ATTRS if name not in stream.attrs]
        if missing_datasets or missing_attrs:
            raise ValueError(
                f"{path}: missing datasets={missing_datasets}, attrs={missing_attrs}"
            )
        X = stream["X"]
        y = stream["y"][()]
        trials = decode_strings(stream["trial"][()])
        labels = decode_strings(stream["label"][()])
        channels = decode_strings(stream["channel"][()])
        retained = json.loads(stream.attrs["retained_channels_json"])
        if stream.attrs["dataset_kind"] != expected_kind:
            raise ValueError(f"{path}: expected kind {expected_kind}")
        if stream.attrs["channel_selection"] != "nmf_insula_union":
            raise ValueError(f"{path}: unexpected channel_selection")
        if X.shape[0] != len(y) or len(y) != len(trials) or len(y) != len(labels):
            raise ValueError(f"{path}: trial dimension mismatch")
        if X.shape[1] != len(channels) or channels != retained:
            raise ValueError(f"{path}: channel dimension/provenance mismatch")
        if int(stream.attrs["n_retained_channels"]) != len(channels):
            raise ValueError(f"{path}: retained count mismatch")
        if len(channels) == 0 or len(y) == 0 or len(np.unique(y)) < 2:
            raise ValueError(f"{path}: empty channels/trials or fewer than two classes")
        if len(set(trials)) != len(trials):
            raise ValueError(f"{path}: duplicate trial IDs")
        return {
            "path": str(path),
            "kind": expected_kind,
            "shape": list(X.shape),
            "channels": channels,
            "trials": trials,
            "labels": labels,
            "classes": sorted(int(value) for value in np.unique(y)),
            "assignment_sha256": str(stream.attrs["assignment_sha256"]),
            "pseudo_subject": str(stream.attrs["pseudo_subject"]),
            "phase": str(stream.attrs.get("phase", "")),
            "feature": str(stream.attrs.get("feature", "")),
            "description": str(stream.attrs.get("description", "")),
        }


def validate_task(
    root: Path,
    bids_task: str,
    band: str = "highgamma",
) -> list[dict[str, object]]:
    if bids_task == "LexicalDelay":
        phases = PHASES
        features = LEXICAL_FEATURES
        descriptions = LEXICAL_CONDITIONS
        expected = 48
    elif bids_task == "LexicalNoDelay":
        phases = LEXICAL_NO_DELAY_PHASES
        features = LEXICAL_FEATURES
        descriptions = LEXICAL_CONDITIONS
        expected = 24
    elif bids_task == "PhonemeSequence":
        phases = PHASES
        features = PHONEME_SEQUENCE_FEATURES
        descriptions = ("Repeat",)
        expected = 16
    else:
        raise ValueError(f"Unsupported task: {bids_task}")

    records = []
    lookup: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for pseudo_subject in PSEUDO_SUBJECTS:
        for phase in phases:
            for feature in features:
                for description in descriptions:
                    path = input_path(
                        root,
                        bids_task,
                        pseudo_subject,
                        feature,
                        phase,
                        description,
                        band,
                    )
                    record = inspect_file(path)
                    records.append(record)
                    lookup[(pseudo_subject, phase, feature, description)] = record
                if bids_task in ("LexicalDelay", "LexicalNoDelay"):
                    decision = lookup[(pseudo_subject, phase, feature, "Decision")]
                    repeat = lookup[(pseudo_subject, phase, feature, "Repeat")]
                    if decision["channels"] != repeat["channels"]:
                        raise ValueError(
                            f"{pseudo_subject} {phase} {feature}: "
                            "Decision/Repeat channels differ"
                        )
    if len(records) != expected:
        raise ValueError(f"{bids_task}: expected {expected} files, found {len(records)}")
    return records


def validate(
    lexical_delay_root: Path,
    phoneme_root: Path,
    band: str = "highgamma",
) -> dict[str, object]:
    lexical = validate_task(lexical_delay_root, "LexicalDelay", band)
    phoneme = validate_task(phoneme_root, "PhonemeSequence", band)
    hashes = {record["assignment_sha256"] for record in lexical + phoneme}
    if len(hashes) != 1:
        raise ValueError(f"Inputs use multiple assignment versions: {sorted(hashes)}")
    return {
        "lexical_delay": len(lexical),
        "phoneme_sequence": len(phoneme),
        "assignment_sha256": hashes.pop(),
        "files": lexical + phoneme,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lexical-delay-root",
        type=Path,
        required=True,
        help="decoding(bipolar) root for LexicalDelay",
    )
    parser.add_argument(
        "--phoneme-root",
        type=Path,
        required=True,
        help="decoding(bipolar) root for PhonemeSequence",
    )
    parser.add_argument("--band", default="highgamma")
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "results" / "decoding_insula" / "input_census.json",
    )
    parser.add_argument(
        "--task-report",
        type=Path,
        help="Optional per-task census JSON (default: {task}_input_census.json)",
    )
    parser.add_argument(
        "--bids-task",
        choices=("LexicalDelay", "LexicalNoDelay", "PhonemeSequence", "both"),
        default="both",
        help="Validate one task or both",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bids_task == "both":
        report = validate(args.lexical_delay_root, args.phoneme_root, args.band)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_name(f".{args.report.name}.tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True))
        temporary.replace(args.report)
        print(
            f"Validated {report['lexical_delay']} LexicalDelay + "
            f"{report['phoneme_sequence']} PhonemeSequence inputs; "
            f"report={args.report}"
        )
        return

    root = (
        args.lexical_delay_root
        if args.bids_task in ("LexicalDelay", "LexicalNoDelay")
        else args.phoneme_root
    )
    records = validate_task(root, args.bids_task, args.band)
    report_path = args.task_report or (
        PROJECT_ROOT
        / "results"
        / "decoding_insula"
        / f"{args.bids_task}_input_census.json"
    )
    report = {
        "bids_task": args.bids_task,
        "n_files": len(records),
        "assignment_sha256": records[0]["assignment_sha256"] if records else None,
        "files": records,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True))
    temporary.replace(report_path)
    print(f"Validated {len(records)} {args.bids_task} inputs; report={report_path}")


if __name__ == "__main__":
    main()
