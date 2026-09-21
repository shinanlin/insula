#!/usr/bin/env python3
"""Validate functional-motif decoding inputs and emit a machine-readable census."""

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
# Empty cells with no significant activity (accepted; no input H5 written).
ALLOWED_MISSING_REGULAR = {
    ("PhonemeSequence", "Sensoryr", "Go", "phoneme", "Repeat"),
    ("PhonemeSequence", "Sensoryr", "Go", "articulator", "Repeat"),
    ("PhonemeSequence", "Sensoryr", "Response", "phoneme", "Repeat"),
    ("PhonemeSequence", "Sensoryr", "Response", "articulator", "Repeat"),
}


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


def inspect_file(path: Path, expected_kind: str) -> dict[str, object]:
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
        }


def validate_regular(
    root: Path,
    bids_task: str,
    band: str = "highgamma",
) -> list[dict[str, object]]:
    features = LEXICAL_FEATURES if bids_task == "LexicalDelay" else PHONEME_SEQUENCE_FEATURES
    descriptions = LEXICAL_CONDITIONS if bids_task == "LexicalDelay" else ("Repeat",)
    records = []
    skipped = []
    lookup = {}
    for pseudo_subject in PSEUDO_SUBJECTS:
        for phase in PHASES:
            for feature in features:
                for description in descriptions:
                    key = (bids_task, pseudo_subject, phase, feature, description)
                    path = input_path(
                        root,
                        bids_task,
                        pseudo_subject,
                        feature,
                        phase,
                        description,
                        band,
                    )
                    if key in ALLOWED_MISSING_REGULAR:
                        if path.exists():
                            raise ValueError(
                                f"Allowlisted skip unexpectedly present: {path}"
                            )
                        skipped.append(
                            {
                                "path": str(path),
                                "kind": "skipped",
                                "bids_task": bids_task,
                                "pseudo_subject": pseudo_subject,
                                "phase": phase,
                                "feature": feature,
                                "description": description,
                            }
                        )
                        continue
                    record = inspect_file(path, "regular")
                    records.append(record)
                    lookup[(pseudo_subject, phase, feature, description)] = record
                if bids_task == "LexicalDelay":
                    decision = lookup[(pseudo_subject, phase, feature, "Decision")]
                    repeat = lookup[(pseudo_subject, phase, feature, "Repeat")]
                    if decision["channels"] != repeat["channels"]:
                        raise ValueError(
                            f"{pseudo_subject} {phase} {feature}: Decision/Repeat channels differ"
                        )
    expected = 144 if bids_task == "LexicalDelay" else 44
    if len(records) != expected:
        raise ValueError(f"{bids_task}: expected {expected} files, found {len(records)}")
    if bids_task == "PhonemeSequence" and len(skipped) != 4:
        raise ValueError(
            f"PhonemeSequence: expected 4 allowlisted skips, found {len(skipped)}"
        )
    return records


def validate_intersection(root: Path, band: str = "highgamma") -> list[dict[str, object]]:
    records = []
    for pseudo_subject in PSEUDO_SUBJECTS:
        pair = {}
        for description in ("Repeat", "Decision"):
            path = input_path(
                root,
                "LexicalDelay",
                pseudo_subject,
                "lexicality",
                "Delay",
                description,
                band,
            )
            pair[description] = inspect_file(path, "intersection")
            records.append(pair[description])
        if pair["Repeat"]["channels"] != pair["Decision"]["channels"]:
            raise ValueError(f"{pseudo_subject}: cross-condition channels differ")
        if pair["Repeat"]["trials"] != pair["Decision"]["trials"]:
            raise ValueError(f"{pseudo_subject}: cross-condition trial IDs differ")
        if pair["Repeat"]["labels"] != pair["Decision"]["labels"]:
            raise ValueError(f"{pseudo_subject}: cross-condition labels differ")
    if len(records) != 12:
        raise ValueError(f"Expected 12 intersection files, found {len(records)}")
    return records


def validate(
    lexical_root: Path,
    phoneme_root: Path,
    intersection_root: Path,
    band: str = "highgamma",
) -> dict[str, object]:
    lexical = validate_regular(lexical_root, "LexicalDelay", band)
    phoneme = validate_regular(phoneme_root, "PhonemeSequence", band)
    intersection = validate_intersection(intersection_root, band)
    hashes = {
        record["assignment_sha256"]
        for record in lexical + phoneme + intersection
    }
    if len(hashes) != 1:
        raise ValueError(f"Inputs use multiple assignment versions: {sorted(hashes)}")
    return {
        "regular_lexical_delay": len(lexical),
        "regular_phoneme_sequence": len(phoneme),
        "intersection_lexical_delay": len(intersection),
        "assignment_sha256": hashes.pop(),
        "files": lexical + phoneme + intersection,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lexical-root", type=Path, required=True)
    parser.add_argument("--phoneme-root", type=Path, required=True)
    parser.add_argument("--intersection-root", type=Path, required=True)
    parser.add_argument("--band", default="highgamma")
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "decoding_functional"
        / "input_validation.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate(
        args.lexical_root,
        args.phoneme_root,
        args.intersection_root,
        args.band,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(f".{args.report.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True))
    temporary.replace(args.report)
    print(
        "Validated 144 LexicalDelay regular + 44 PhonemeSequence regular + "
        f"12 intersection inputs; report={args.report}"
    )


if __name__ == "__main__":
    main()
