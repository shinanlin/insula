#!/usr/bin/env python3
"""Prepare NNLS-component pseudo-subject H5s for PhonemeSequence decoding.

Reads fixed-W NNLS traces ``H (n_trials, 3, n_times)`` and writes one
single-channel decode H5 per component (NNLSSustain / NNLSMotor / NNLSSensory)
using the same schema as ``prepare_functional_decoding_dataset._atomic_write_h5``.
Labels reuse ``sequence_phoneme`` / ``sequence_articulator`` and the same
RT<50 ms exclusion as hard-label PhonemeSequence prepare.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import mne
import numpy as np
import pandas as pd
import xarray as xr

from src.decoding.prepare_functional_decoding_dataset import (
    PHONEME_SEQUENCE_FEATURES,
    _atomic_write_h5,
    _output_path,
    compute_rt_by_syllable,
    normalize_subject,
    prepare_feature,
)
from src.nmf.nnls_projection import COMPONENT_NAMES, read_trace_h5
from src.paths import PROJECT_ROOT, nmf_nnls_dir


LOGGER = logging.getLogger(__name__)

BIDS_TASK = "PhonemeSequence"
DESCRIPTION = "Repeat"
PHASES = ("Stimulus", "Delay", "Go", "Response")
BAND = "highgamma"
REFERENCE = "bipolar"
FS = 128

COMPONENT_TO_PSEUDO = {
    # Prefix all NNLS pseudos so they never collide with hard-label
    # Sensory / Sustain / Motor.
    "sustain": "NNLSSustain",
    "motor": "NNLSMotor",
    "sensory": "NNLSSensory",
}
PSEUDO_SUBJECTS = tuple(COMPONENT_TO_PSEUDO[name] for name in COMPONENT_NAMES)


def nnls_trace_path(nnls_root: Path, phase: str) -> Path:
    return (
        Path(nnls_root)
        / "traces"
        / f"task-{BIDS_TASK}_proc-{phase}_desc-{DESCRIPTION}_nnls.h5"
    )


def _conditions_for_epochs(epochs: mne.Epochs) -> list[str]:
    id_to_name = {int(code): str(name) for name, code in epochs.event_id.items()}
    return [
        id_to_name[int(code)].split("/")[-1]
        for code in epochs.events[:, 2]
    ]


def _rt_bad_epochs(
    subject: str,
    conditions: list[str],
    bids_root: Path,
) -> set[int]:
    rt_by_syllable = compute_rt_by_syllable(subject, bids_root)
    if not rt_by_syllable:
        return set()
    epoch_order = pd.DataFrame(
        {
            "epoch": np.arange(len(conditions), dtype=int),
            "condition": conditions,
        }
    )
    epoch_order["_syl_rank"] = epoch_order.groupby("condition").cumcount()

    def lookup_rt(row: pd.Series) -> float:
        values = rt_by_syllable.get(str(row["condition"]), [])
        rank = int(row["_syl_rank"])
        return float(values[rank]) if rank < len(values) else np.nan

    epoch_order["rt_ms"] = epoch_order.apply(lookup_rt, axis=1)
    bad = set(epoch_order.loc[epoch_order["rt_ms"] < 50.0, "epoch"].astype(int))
    if bad:
        LOGGER.info("sub-%s: excluding %d RT<50 ms epochs", subject, len(bad))
    return bad


def _subject_conditions(
    subject: str,
    epoch_file: Path,
    bids_root: Path,
) -> tuple[list[str], set[int]]:
    epochs = mne.read_epochs(str(epoch_file), preload=False, verbose="error")
    conditions = _conditions_for_epochs(epochs)
    bad = _rt_bad_epochs(subject, conditions, bids_root)
    return conditions, bad


def collect_labeled_trials(
    *,
    H: np.ndarray,
    times: np.ndarray,
    trials: pd.DataFrame,
    component_names: list[str],
    bids_root: Path,
) -> pd.DataFrame:
    """Return one row per kept trial with condition labels and source H indices."""
    if list(component_names) != list(COMPONENT_NAMES):
        raise ValueError(
            f"Unexpected component_names={component_names}; expected {list(COMPONENT_NAMES)}"
        )
    if H.shape[0] != len(trials):
        raise ValueError(f"H/trials mismatch: {H.shape[0]} vs {len(trials)}")
    n_times = int(times.shape[0])
    if H.shape[1:] != (len(COMPONENT_NAMES), n_times):
        raise ValueError(
            f"Unexpected H shape {H.shape}; expected (*, {len(COMPONENT_NAMES)}, {n_times})"
        )

    records: list[dict[str, object]] = []
    for (subject, epoch_file), block in trials.groupby(
        ["subject", "epoch_file"], sort=False
    ):
        subject = normalize_subject(subject)
        epoch_path = Path(str(epoch_file))
        if not epoch_path.exists():
            raise FileNotFoundError(epoch_path)
        conditions, bad_epochs = _subject_conditions(subject, epoch_path, bids_root)
        n_epochs = len(conditions)
        for global_idx, row in block.iterrows():
            trial_index = int(row["trial_index"])
            if trial_index < 0 or trial_index >= n_epochs:
                LOGGER.warning(
                    "sub-%s: trial_index %d outside epochs (n=%d); drop",
                    subject,
                    trial_index,
                    n_epochs,
                )
                continue
            if trial_index in bad_epochs:
                continue
            h_row = np.asarray(H[int(global_idx)], dtype=float)
            if not np.isfinite(h_row).all():
                continue
            records.append(
                {
                    "subject": subject,
                    "trial_index": trial_index,
                    "condition": str(conditions[trial_index]),
                    "epoch_file": str(epoch_path),
                    "source_row": int(global_idx),
                }
            )

    if not records:
        raise RuntimeError("No labeled NNLS trials survived filtering")
    frame = pd.DataFrame.from_records(records)
    # Global trial IDs: condition_rank across pooled subjects (labeler uses rsplit).
    frame = frame.sort_values(["condition", "subject", "trial_index"]).reset_index(
        drop=True
    )
    frame["_idx"] = frame.groupby("condition").cumcount() + 1
    frame["trial"] = frame["condition"] + "_" + frame["_idx"].astype(str)
    return frame


def _component_array(
    frame: pd.DataFrame,
    H: np.ndarray,
    component: str,
    times: np.ndarray,
) -> xr.DataArray:
    comp_i = list(COMPONENT_NAMES).index(component)
    rows = frame["source_row"].to_numpy(dtype=int)
    data = np.asarray(H[rows, comp_i, :], dtype=np.float64)
    return xr.DataArray(
        data[:, None, :],
        dims=("trial", "channel", "time"),
        coords={
            "trial": frame["trial"].astype(str).to_list(),
            "channel": [component],
            "time": np.asarray(times, dtype=float),
        },
    )


def prepare_phase(
    *,
    phase: str,
    nnls_root: Path,
    bids_root: Path,
    output_root: Path,
    band: str,
    dry_run: bool,
    overwrite: bool,
) -> list[dict[str, object]]:
    path = nnls_trace_path(nnls_root, phase)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = read_trace_h5(path)
    H = np.asarray(payload["H"], dtype=float)
    times = np.asarray(payload["times"], dtype=float)
    trials = payload["trials"]
    component_names = list(payload["component_names"])
    LOGGER.info(
        "Loaded %s: H=%s n_trial_rows=%d",
        path.name,
        H.shape,
        len(trials),
    )
    labeled = collect_labeled_trials(
        H=H,
        times=times,
        trials=trials,
        component_names=component_names,
        bids_root=bids_root,
    )
    LOGGER.info(
        "%s: kept %d / %d trials after RT/finite filters",
        phase,
        len(labeled),
        len(trials),
    )

    # prepare_feature only needs frame for lexicality; pass minimal stub.
    stub_frame = labeled[["trial"]].copy()
    records: list[dict[str, object]] = []
    subjects_used = sorted(labeled["subject"].unique().tolist())

    for component in COMPONENT_NAMES:
        pseudo = COMPONENT_TO_PSEUDO[component]
        X = _component_array(labeled, H, component, times)
        common_attrs: dict[str, object] = {
            "channel_selection": "nnls_component",
            "nnls_source": str(path.resolve()),
            "nnls_component": component,
            "pseudo_subject": pseudo,
            "phase": phase,
            "reference": REFERENCE,
            "band": band,
            "description": DESCRIPTION,
            "bids_task": BIDS_TASK,
            "subjects_json": subjects_used,
            "n_source_trials": int(len(trials)),
            "n_kept_trials": int(len(labeled)),
            "fs": float(payload["attrs"].get("fs", FS)),
        }
        for feature in PHONEME_SEQUENCE_FEATURES:
            prepared = prepare_feature(BIDS_TASK, feature, stub_frame, X)
            target = _output_path(
                output_root,
                BIDS_TASK,
                pseudo,
                feature,
                DESCRIPTION,
                phase,
                band,
            )
            record = {
                "kind": "nnls_component",
                "bids_task": BIDS_TASK,
                "pseudo_subject": pseudo,
                "component": component,
                "phase": phase,
                "feature": feature,
                "description": DESCRIPTION,
                "n_trials": int(prepared.X.shape[0]),
                "n_classes": len(prepared.event_id),
                "path": str(target),
                "status": "dry_run" if dry_run else "written",
            }
            if dry_run:
                LOGGER.info(
                    "DRY-RUN %s %s %s → %s (n=%d)",
                    pseudo,
                    feature,
                    phase,
                    target,
                    prepared.X.shape[0],
                )
            else:
                _atomic_write_h5(
                    target,
                    prepared,
                    attrs=common_attrs,
                    candidate_channels=[component],
                    retained_channels=[component],
                    channel_qc={component: {"present": True, "fraction_nan_trials": 0.0}},
                    removed_channels=[],
                    overwrite=overwrite,
                )
                LOGGER.info("Wrote %s", target)
            records.append(record)
    return records


def run(args: argparse.Namespace) -> pd.DataFrame:
    bids_root = Path(args.bids_root)
    nnls_root = Path(args.nnls_root) if args.nnls_root else nmf_nnls_dir()
    output_root = (
        Path(args.output_root)
        if args.output_root
        else bids_root / "derivatives" / f"decoding({args.reference})"
    )
    phases = tuple(args.phases) if args.phases else PHASES
    records: list[dict[str, object]] = []
    errors: list[str] = []
    for phase in phases:
        try:
            records.extend(
                prepare_phase(
                    phase=phase,
                    nnls_root=nnls_root,
                    bids_root=bids_root,
                    output_root=output_root,
                    band=args.band,
                    dry_run=args.dry_run,
                    overwrite=args.overwrite,
                )
            )
        except Exception as error:
            message = f"{BIDS_TASK} {phase}: {error}"
            LOGGER.exception(message)
            errors.append(message)
            records.append(
                {
                    "kind": "error",
                    "bids_task": BIDS_TASK,
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
        / "decoding_nnls"
        / f"{BIDS_TASK}_prepare_manifest.csv"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}")
    manifest.to_csv(temporary, index=False)
    os.replace(temporary, manifest_path)
    LOGGER.info("Wrote manifest: %s", manifest_path)
    if errors:
        raise RuntimeError(
            f"NNLS prepare failed for {len(errors)} phase(s); see {manifest_path}"
        )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bids-root",
        default="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS",
    )
    parser.add_argument("--nnls-root", default=None, help="defaults to results/nmf/nnls_projection")
    parser.add_argument("--reference", default=REFERENCE, choices=("bipolar", "car"))
    parser.add_argument("--band", default=BAND)
    parser.add_argument("--output-root")
    parser.add_argument("--manifest")
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=PHASES,
        default=None,
    )
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
