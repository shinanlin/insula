#!/usr/bin/env python3
"""Prepare merged INS (INSl/INSr) pseudo-subject decoding datasets.

Reads task BIDS ``epoch(band)(zscore)`` trials and ``epoch(band)(sig)``
eligibility, unions all NMF-assigned insula channels per hemisphere, and writes
decoding H5 files under each task's ``derivatives/decoding(bipolar)/``.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Mapping

import mne
from mne_bids import BIDSPath
import pandas as pd

from src.decoding import prepare_functional_decoding_dataset as functional
from src.paths import nmf_assignments_path


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PSEUDO_SUBJECTS = ("INSl", "INSr")
LEXICAL_NO_DELAY_PHASES = ("Stimulus", "Response")
SUPPORTED_TASKS = ("LexicalDelay", "LexicalNoDelay", "PhonemeSequence")
CHANNEL_SELECTION = "nmf_insula_union"
INSULA_CLUSTER_LABEL = "nmf_insula_union"


def load_assignments(path: Path) -> pd.DataFrame:
    """Load NMF assignments mapped to hemisphere-level INS pseudo-subjects."""
    frame = pd.read_csv(path)
    required = {"subject", "channel", "hemi", "functional_cluster"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Assignment table is missing columns: {missing}")

    frame = frame.copy()
    frame["subject"] = frame["subject"].map(functional.normalize_subject)
    frame["channel"] = frame["channel"].astype(str)
    frame["hemi"] = frame["hemi"].astype(str).str.upper()
    if frame[["subject", "channel"]].duplicated().any():
        duplicate = frame.loc[
            frame[["subject", "channel"]].duplicated(keep=False),
            ["subject", "channel"],
        ]
        raise ValueError(
            "Duplicate subject+channel assignments: "
            f"{duplicate.to_dict(orient='records')[:10]}"
        )
    unknown_clusters = sorted(
        set(frame["functional_cluster"]) - set(functional.CLUSTER_TO_PSEUDO)
    )
    if unknown_clusters:
        raise ValueError(f"Unknown functional clusters: {unknown_clusters}")
    unknown_hemi = sorted(set(frame["hemi"]) - {"L", "R"})
    if unknown_hemi:
        raise ValueError(f"Unknown hemisphere labels: {unknown_hemi}")

    frame["pseudo_subject"] = "INS" + frame["hemi"].str.lower()
    observed = set(frame["pseudo_subject"])
    missing_groups = sorted(set(PSEUDO_SUBJECTS) - observed)
    if missing_groups:
        raise ValueError(f"Assignment table lacks pseudo-subjects: {missing_groups}")
    return frame


def significance_union(
    bids_task: str,
    channels_by_description: Mapping[str, set[str]],
) -> set[str]:
    """Return same-phase significant-channel union for lexical or sequence tasks."""
    if bids_task in ("LexicalDelay", "LexicalNoDelay"):
        return set(channels_by_description.get("Decision", set())) | set(
            channels_by_description.get("Repeat", set())
        )
    if bids_task == "PhonemeSequence":
        return set(channels_by_description.get("Repeat", set()))
    raise ValueError(f"Unsupported BIDS task: {bids_task}")


def is_lexical_task(bids_task: str) -> bool:
    return bids_task in ("LexicalDelay", "LexicalNoDelay")


def task_phases(bids_task: str) -> tuple[str, ...]:
    if bids_task == "LexicalNoDelay":
        return LEXICAL_NO_DELAY_PHASES
    return functional.PHASES


def task_descriptions(bids_task: str) -> tuple[str, ...]:
    if is_lexical_task(bids_task):
        return functional.LEXICAL_CONDITIONS
    return ("Repeat",)


def task_features(bids_task: str) -> tuple[str, ...]:
    if is_lexical_task(bids_task):
        return functional.LEXICAL_FEATURES
    return functional.PHONEME_SEQUENCE_FEATURES


def prepare_feature(
    bids_task: str,
    feature: str,
    frame: pd.DataFrame,
    X,
):
    """Label trials; LexicalNoDelay reuses LexicalDelay phoneme/articulator labelers."""
    effective_task = "LexicalDelay" if bids_task == "LexicalNoDelay" else bids_task
    return functional.prepare_feature(effective_task, feature, frame, X)


def _process_phase_group(
    *,
    bids_root: Path,
    bids_task: str,
    phase: str,
    pseudo_subject: str,
    assignments: pd.DataFrame,
    assignment_path: Path,
    assignment_sha256: str,
    zscore_index: Mapping[tuple[str, str, str], BIDSPath],
    sig_index: Mapping[tuple[str, str, str], BIDSPath],
    reference: str,
    band: str,
    regular_root: Path,
    dry_run: bool,
    overwrite: bool,
) -> list[dict[str, object]]:
    group_assignments = assignments[assignments["pseudo_subject"].eq(pseudo_subject)]
    hemi = pseudo_subject[-1].upper()
    descriptions = task_descriptions(bids_task)
    frames: list[pd.DataFrame] = []
    selection_stats = []

    for subject in sorted(set(group_assignments["subject"])):
        sig_by_description = {
            description: functional._read_sig_channels(
                sig_index, subject, description, phase
            )
            for description in descriptions
        }
        significant = significance_union(bids_task, sig_by_description)
        for description in descriptions:
            epoch_path = zscore_index.get((subject, description, phase))
            if epoch_path is None or not epoch_path.fpath.exists():
                continue
            epochs = mne.read_epochs(epoch_path, preload=True, verbose="error")
            include = functional.select_assigned_channels(
                assignments,
                pseudo_subject,
                subject,
                significant,
                epochs.ch_names,
            )
            include = [
                channel
                for channel in include
                if not (
                    bids_task == "PhonemeSequence"
                    and channel in functional.PHONEME_SEQUENCE_EXCLUDE_CHANNELS
                )
            ]
            selection_stats.append(
                {
                    "subject": subject,
                    "description": description,
                    "phase": phase,
                    "n_sig_decision": len(sig_by_description.get("Decision", set())),
                    "n_sig_repeat": len(sig_by_description.get("Repeat", set())),
                    "n_sig_union": len(significant),
                    "n_selected": len(include),
                }
            )
            if not include:
                continue
            epochs.pick(include)
            if is_lexical_task(bids_task):
                frame = functional._lexical_frame(epochs, subject, description, phase)
            else:
                frame = functional._phoneme_sequence_frame(
                    epochs, subject, description, phase, bids_root
                )
            frames.append(frame)

    if not frames:
        raise RuntimeError(f"No selected data for {pseudo_subject} {bids_task} {phase}")
    group = pd.concat(frames, ignore_index=True)
    if bids_task == "LexicalDelay":
        group = group[group["remark"].eq("CORRECT")]

    condition_frames = {
        description: group[group["description"].eq(description)]
        for description in descriptions
    }
    missing = [description for description, frame in condition_frames.items() if frame.empty]
    if missing:
        raise RuntimeError(
            f"Missing {missing} for {pseudo_subject} {bids_task} {phase}"
        )
    arrays = {
        description: (
            frame.groupby(["trial", "channel", "time"])["value"]
            .mean()
            .to_xarray()
        )
        for description, frame in condition_frames.items()
    }
    if is_lexical_task(bids_task):
        filtered, candidates, retained, channel_qc, removed = functional.symmetric_condition_qc(
            arrays
        )
    else:
        X, candidates, retained, channel_qc, removed = functional.single_condition_qc(
            arrays["Repeat"]
        )
        filtered = {"Repeat": X}
    if not retained:
        raise RuntimeError(
            f"No channels survive QC for {pseudo_subject} {bids_task} {phase}"
        )

    features = task_features(bids_task)
    records: list[dict[str, object]] = []
    significant_assigned = len(candidates)
    common_attrs = {
        "atlas": "hammers",
        "input_datatype": functional.ZSCORE_DATATYPE,
        "significance_datatype": functional.SIG_DATATYPE,
        "channel_selection": CHANNEL_SELECTION,
        "assignment_path": str(assignment_path.resolve()),
        "assignment_sha256": assignment_sha256,
        "assignment_rows": len(assignments),
        "functional_cluster": INSULA_CLUSTER_LABEL,
        "hemi": hemi,
        "pseudo_subject": pseudo_subject,
        "phase": phase,
        "reference": reference,
        "band": band,
        "selection_stats_json": selection_stats,
        "subjects_json": sorted(
            {
                str(value).split("_", 1)[0]
                for value in retained
                if "_" in str(value)
            }
        ),
    }

    for description in descriptions:
        for feature in features:
            prepared = prepare_feature(
                bids_task,
                feature,
                condition_frames[description],
                filtered[description],
            )
            target = functional._output_path(
                regular_root,
                bids_task,
                pseudo_subject,
                feature,
                description,
                phase,
                band,
            )
            attrs = {
                **common_attrs,
                "roi": pseudo_subject,
                "description": description,
                "task": bids_task,
                "feature": feature,
                "dataset_kind": "regular",
            }
            if not dry_run:
                functional._atomic_write_h5(
                    target,
                    prepared,
                    attrs=attrs,
                    candidate_channels=candidates,
                    retained_channels=retained,
                    channel_qc=channel_qc,
                    removed_channels=removed,
                    overwrite=overwrite,
                )
            records.append(
                functional._manifest_record(
                    kind="regular",
                    target=target,
                    bids_task=bids_task,
                    pseudo_subject=pseudo_subject,
                    cluster=INSULA_CLUSTER_LABEL,
                    hemi=hemi,
                    phase=phase,
                    description=description,
                    feature=feature,
                    assignment_count=len(group_assignments),
                    significant_assigned_count=significant_assigned,
                    candidates=candidates,
                    retained=retained,
                    prepared=prepared,
                    status="dry_run" if dry_run else "written",
                )
            )
    return records


def run(args: argparse.Namespace) -> pd.DataFrame:
    bids_root = Path(args.bids_root)
    assignment_path = Path(args.assignments)
    assignments = load_assignments(assignment_path)
    assignment_sha256 = functional.file_sha256(assignment_path)
    if len(assignments) != args.expected_assignment_rows:
        raise ValueError(
            f"Expected {args.expected_assignment_rows} assignments, found {len(assignments)}"
        )

    zscore_index = functional._epoch_index(
        bids_root, args.reference, args.band, args.bids_task, functional.ZSCORE_DATATYPE
    )
    sig_index = functional._epoch_index(
        bids_root, args.reference, args.band, args.bids_task, functional.SIG_DATATYPE
    )
    descriptions = task_descriptions(args.bids_task)
    phases = task_phases(args.bids_task)
    expected_zscore = {
        (subject, description, phase)
        for subject in assignments["subject"].unique()
        for description in descriptions
        for phase in phases
        if (subject, description, phase) in zscore_index
    }
    if not expected_zscore:
        raise RuntimeError(f"No zscore epochs match {args.bids_task}")

    regular_root = (
        Path(args.output_root)
        if args.output_root
        else bids_root / "derivatives" / f"decoding({args.reference})"
    )

    records: list[dict[str, object]] = []
    errors: list[str] = []
    for pseudo_subject in PSEUDO_SUBJECTS:
        for phase in phases:
            try:
                records.extend(
                    _process_phase_group(
                        bids_root=bids_root,
                        bids_task=args.bids_task,
                        phase=phase,
                        pseudo_subject=pseudo_subject,
                        assignments=assignments,
                        assignment_path=assignment_path,
                        assignment_sha256=assignment_sha256,
                        zscore_index=zscore_index,
                        sig_index=sig_index,
                        reference=args.reference,
                        band=args.band,
                        regular_root=regular_root,
                        dry_run=args.dry_run,
                        overwrite=args.overwrite,
                    )
                )
            except Exception as error:
                message = f"{pseudo_subject} {args.bids_task} {phase}: {error}"
                LOGGER.exception(message)
                errors.append(message)
                records.append(
                    {
                        "kind": "error",
                        "bids_task": args.bids_task,
                        "pseudo_subject": pseudo_subject,
                        "phase": phase,
                        "status": "error",
                        "error": str(error),
                    }
                )

    manifest = pd.DataFrame(records)
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else PROJECT_ROOT
        / "results"
        / "decoding_insula"
        / f"{args.bids_task}_prepare_manifest.csv"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}")
    manifest.to_csv(temporary, index=False)
    os.replace(temporary, manifest_path)
    LOGGER.info("Wrote manifest: %s", manifest_path)
    if errors:
        raise RuntimeError(
            f"INS preparation failed for {len(errors)} group/phase combinations; "
            f"see {manifest_path}"
        )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", required=True)
    parser.add_argument(
        "--bids-task", required=True, choices=SUPPORTED_TASKS
    )
    parser.add_argument(
        "--assignments",
        default=str(nmf_assignments_path()),
    )
    parser.add_argument("--reference", default="bipolar", choices=("bipolar", "car"))
    parser.add_argument("--band", default="highgamma")
    parser.add_argument("--output-root")
    parser.add_argument("--manifest")
    parser.add_argument("--expected-assignment-rows", type=int, default=281)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
