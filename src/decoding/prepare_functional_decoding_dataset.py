#!/usr/bin/env python3
"""Prepare functional-motif pseudo-subject datasets for speech decoding.

This is deliberately separate from the task repositories' anatomical ROI
preparers.  NMF assignments define the functional group, same-phase
``epoch(band)(sig)`` files define eligibility, and trial-level signals always
come from ``epoch(band)(zscore)`` files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import glob
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Iterable, Mapping

import h5py
import mne
from mne_bids import BIDSPath
import numpy as np
import pandas as pd
import xarray as xr


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PHASES = ("Stimulus", "Delay", "Go", "Response")
LEXICAL_CONDITIONS = ("Decision", "Repeat")
ZSCORE_DATATYPE = "epoch(band)(zscore)"
SIG_DATATYPE = "epoch(band)(sig)"
QC_THRESHOLD = 0.5

CLUSTER_TO_PSEUDO = {
    "sensory": "Sensory",
    "sustain": "Sustain",
    "motor": "Motor",
}
# Bilateral pools: L+R electrodes share one pseudo-subject per cluster.
PSEUDO_SUBJECTS = (
    "Sensory",
    "Sustain",
    "Motor",
)

LEXICAL_FEATURES = ("phoneme", "articulator", "lexicality")
PHONEME_SEQUENCE_FEATURES = ("phoneme", "articulator")

ARTICULATORY_PHONE_TO_LABEL = {
    "M": "sonorant",
    "N": "sonorant",
    "NG": "sonorant",
    "L": "sonorant",
    "R": "sonorant",
    "W": "sonorant",
    "Y": "sonorant",
    "B": "labial_obstruent",
    "P": "labial_obstruent",
    "F": "labial_obstruent",
    "V": "labial_obstruent",
    "D": "coronal_obstruent",
    "T": "coronal_obstruent",
    "S": "coronal_obstruent",
    "Z": "coronal_obstruent",
    "JH": "coronal_obstruent",
    "CH": "coronal_obstruent",
    "SH": "coronal_obstruent",
    "ZH": "coronal_obstruent",
    "TH": "coronal_obstruent",
    "DH": "coronal_obstruent",
    "G": "posterior_obstruent",
    "K": "posterior_obstruent",
    "HH": "posterior_obstruent",
}

PHONEME_SEQUENCE_EXCLUDE_CHANNELS = {
    "D0040_L1IF3-4",
    "D0084_RFAI1-2",
    "D0086_LTPI2-3",
    "D0032_LAI4-5",
    "D0090_RIA4-5",
    "D0096_LFAI2-3",
    "D0096_LFAI4-5",
    "D0102_RFAI2-3",
    "D0106_LTAS2-3",
    "D0121_LFMI3-4",
    "D0122_LFAI3-4",
    "D0125_LIA4-5",
    "D0125_LIA7-8",
}

_G2P = None


@dataclass(frozen=True)
class PreparedFeature:
    X: xr.DataArray
    y: np.ndarray
    trials: list[str]
    conditions: list[str]
    labels: list[str]
    event_id: dict[str, int]


def normalize_subject(value: object) -> str:
    value = str(value)
    return value[4:] if value.startswith("sub-") else value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_assignments(path: Path) -> pd.DataFrame:
    """Load and validate the frozen NMF assignment table."""
    frame = pd.read_csv(path)
    required = {"subject", "channel", "hemi", "functional_cluster"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Assignment table is missing columns: {missing}")

    frame = frame.copy()
    frame["subject"] = frame["subject"].map(normalize_subject)
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
        set(frame["functional_cluster"]) - set(CLUSTER_TO_PSEUDO)
    )
    if unknown_clusters:
        raise ValueError(f"Unknown functional clusters: {unknown_clusters}")
    unknown_hemi = sorted(set(frame["hemi"]) - {"L", "R"})
    if unknown_hemi:
        raise ValueError(f"Unknown hemisphere labels: {unknown_hemi}")

    frame["pseudo_subject"] = [
        CLUSTER_TO_PSEUDO[cluster] for cluster in frame["functional_cluster"]
    ]
    observed = set(frame["pseudo_subject"])
    missing_groups = sorted(set(PSEUDO_SUBJECTS) - observed)
    if missing_groups:
        raise ValueError(f"Assignment table lacks pseudo-subjects: {missing_groups}")
    return frame


def significance_union(
    bids_task: str,
    channels_by_description: Mapping[str, set[str]],
) -> set[str]:
    """Return the pre-specified same-phase significant-channel set."""
    if bids_task == "LexicalDelay":
        return set(channels_by_description.get("Decision", set())) | set(
            channels_by_description.get("Repeat", set())
        )
    if bids_task == "PhonemeSequence":
        return set(channels_by_description.get("Repeat", set()))
    raise ValueError(f"Unsupported BIDS task: {bids_task}")


def select_assigned_channels(
    assignments: pd.DataFrame,
    pseudo_subject: str,
    subject: str,
    significant_channels: Iterable[str],
    available_channels: Iterable[str],
) -> list[str]:
    """Intersect frozen assignment, phase significance, and epoch availability."""
    subset = assignments[
        assignments["pseudo_subject"].eq(pseudo_subject)
        & assignments["subject"].eq(normalize_subject(subject))
    ]
    assigned = set(subset["channel"])
    return sorted(assigned & set(significant_channels) & set(available_channels))


def _epoch_index(
    bids_root: Path,
    reference: str,
    band: str,
    bids_task: str,
    datatype: str,
) -> dict[tuple[str, str, str], BIDSPath]:
    query = BIDSPath(
        root=str(bids_root / "derivatives" / f"epoch({reference})"),
        # Historical epoch(band)(sig) filenames omit the BIDS task entity;
        # zscore filenames include it.  Constraining task for sig would match
        # zero files in both target datasets.
        task=bids_task if datatype == ZSCORE_DATATYPE else None,
        datatype=datatype,
        suffix=band,
        extension=".h5",
        check=False,
    )
    index: dict[tuple[str, str, str], BIDSPath] = {}
    for path in query.match():
        key = (
            normalize_subject(path.subject),
            str(path.description),
            str(path.processing),
        )
        if key in index:
            raise ValueError(f"Multiple epoch files for {key}: {index[key]}, {path}")
        index[key] = path
    return index


def _read_sig_channels(
    sig_index: Mapping[tuple[str, str, str], BIDSPath],
    subject: str,
    description: str,
    phase: str,
) -> set[str]:
    path = sig_index.get((normalize_subject(subject), description, phase))
    if path is None or not path.fpath.exists():
        return set()
    epochs = mne.read_epochs(path, preload=False, verbose="error")
    return set(epochs.ch_names)


def _trial_ids(frame: pd.DataFrame) -> pd.DataFrame:
    trial_map = (
        frame[["epoch", "condition"]]
        .drop_duplicates()
        .assign(_idx=lambda data: data.groupby("condition").cumcount() + 1)
        .assign(
            trial=lambda data: data["condition"] + "_" + data["_idx"].astype(str)
        )[["epoch", "condition", "trial"]]
    )
    return frame.merge(trial_map, on=["epoch", "condition"], how="left")


def _lexical_frame(epochs: mne.Epochs, subject: str, description: str, phase: str):
    frame = epochs.to_data_frame(
        long_format=True, scalings={"seeg": 1}, verbose=False
    )
    split = frame["condition"].str.split("/")
    frame["remark"] = split.str[4:].str.join("/")
    frame["lexicality"] = split.str[2]
    frame["condition"] = split.str[3:4].str.join("/")
    frame["subject"] = normalize_subject(subject)
    frame["description"] = description
    frame["phase"] = phase
    return _trial_ids(frame)


@lru_cache(maxsize=None)
def compute_rt_by_syllable(subject: str, bids_root: Path) -> dict[str, list[float]]:
    """Replicate the PhonemeSequence RT<50 ms exclusion source."""
    pattern = str(
        bids_root
        / "derivatives"
        / "bipolar"
        / f"sub-{normalize_subject(subject)}"
        / "ieeg"
        / "*_desc-bipolar_events.tsv"
    )
    rt_by_syllable: dict[str, list[float]] = {}
    for fpath in sorted(glob.glob(pattern)):
        events = pd.read_csv(fpath, sep="\t")
        events = events[~events["trial_type"].str.startswith("bad ", na=False)]
        go = events[
            events["trial_type"].str.contains("Word/Go/LS/", na=False)
        ].copy()
        response = events[
            events["trial_type"].str.contains("Word/Response/LS/", na=False)
        ].copy()
        go["syllable"] = go["trial_type"].str.split("/").str[-1]
        response["syllable"] = response["trial_type"].str.split("/").str[-1]
        go = go.sort_values("onset").reset_index(drop=True)
        response = response.sort_values("onset").reset_index(drop=True)
        for syllable, go_group in go.groupby("syllable", sort=False):
            response_group = response[
                response["syllable"].eq(syllable)
            ].reset_index(drop=True)
            go_group = go_group.reset_index(drop=True)
            values = rt_by_syllable.setdefault(str(syllable), [])
            for index in range(min(len(go_group), len(response_group))):
                values.append(
                    float(response_group.loc[index, "onset"] - go_group.loc[index, "onset"])
                    * 1000.0
                )
    return rt_by_syllable


def _phoneme_sequence_frame(
    epochs: mne.Epochs,
    subject: str,
    description: str,
    phase: str,
    bids_root: Path,
):
    frame = epochs.to_data_frame(
        long_format=True, scalings={"seeg": 1}, verbose=False
    )
    frame["condition"] = frame["condition"].str.split("/").str[-1]
    frame["subject"] = normalize_subject(subject)
    frame["description"] = description
    frame["phase"] = phase
    frame = _trial_ids(frame)

    rt_by_syllable = compute_rt_by_syllable(subject, bids_root)
    if not rt_by_syllable:
        return frame
    epoch_order = (
        frame[["epoch", "condition", "trial"]]
        .drop_duplicates()
        .sort_values("epoch")
        .reset_index(drop=True)
    )
    epoch_order["_syl_rank"] = epoch_order.groupby("condition").cumcount()

    def lookup_rt(row):
        values = rt_by_syllable.get(str(row["condition"]), [])
        rank = int(row["_syl_rank"])
        return values[rank] if rank < len(values) else np.nan

    epoch_order["rt_ms"] = epoch_order.apply(lookup_rt, axis=1)
    bad_epochs = set(epoch_order.loc[epoch_order["rt_ms"] < 50.0, "epoch"])
    if bad_epochs:
        LOGGER.info("sub-%s: excluding %d RT<50 ms epochs", subject, len(bad_epochs))
        frame = frame[~frame["epoch"].isin(bad_epochs)]
    return frame


def _channel_qc(X: xr.DataArray, channels: list[str]) -> dict[str, dict[str, object]]:
    values = X.values
    fractions = np.isnan(values).any(axis=2).mean(axis=0)
    return {
        channel: {
            "present": bool(np.isfinite(values[:, index, :]).any()),
            "fraction_nan_trials": float(fractions[index]),
        }
        for index, channel in enumerate(channels)
    }


def _drop_bad_trials(X: xr.DataArray, threshold: float) -> xr.DataArray:
    if not X.sizes.get("trial", 0) or not X.sizes.get("channel", 0):
        return X.isel(trial=[])
    fraction = np.isnan(X.values).any(axis=2).mean(axis=1)
    return X.isel(trial=np.flatnonzero(fraction <= threshold))


def symmetric_condition_qc(
    arrays: Mapping[str, xr.DataArray],
    threshold: float = QC_THRESHOLD,
) -> tuple[
    dict[str, xr.DataArray],
    list[str],
    list[str],
    dict[str, dict[str, dict[str, object]]],
    dict[str, object],
]:
    """Jointly enforce the same channel list in Decision and Repeat."""
    missing = sorted(set(LEXICAL_CONDITIONS) - set(arrays))
    if missing:
        raise ValueError(f"Missing LexicalDelay conditions for QC: {missing}")
    candidates = sorted(
        {
            str(channel)
            for array in arrays.values()
            for channel in array["channel"].values
        }
    )
    aligned = {
        description: arrays[description].reindex(channel=candidates)
        for description in LEXICAL_CONDITIONS
    }
    initial_qc = {
        description: _channel_qc(aligned[description], candidates)
        for description in LEXICAL_CONDITIONS
    }
    active = [
        channel
        for channel in candidates
        if all(
            initial_qc[description][channel]["present"]
            and initial_qc[description][channel]["fraction_nan_trials"] <= threshold
            for description in LEXICAL_CONDITIONS
        )
    ]
    history: dict[str, dict[str, dict[str, object]]] = {
        description: {} for description in LEXICAL_CONDITIONS
    }
    filtered: dict[str, xr.DataArray] = {}
    for _ in range(len(candidates) + 1):
        if not active:
            filtered = {
                description: aligned[description].isel(channel=[])
                for description in LEXICAL_CONDITIONS
            }
            break
        post_qc = {}
        for description in LEXICAL_CONDITIONS:
            current = _drop_bad_trials(
                aligned[description].sel(channel=active), threshold
            )
            filtered[description] = current
            post_qc[description] = _channel_qc(current, active)
            history[description].update(post_qc[description])
        next_active = [
            channel
            for channel in active
            if all(
                post_qc[description][channel]["present"]
                and post_qc[description][channel]["fraction_nan_trials"] <= threshold
                for description in LEXICAL_CONDITIONS
            )
        ]
        if next_active == active:
            break
        active = next_active
    else:
        raise RuntimeError("Symmetric channel/trial QC did not converge")

    retained = list(active)
    for description in LEXICAL_CONDITIONS:
        for channel in candidates:
            record = history[description].get(channel)
            initial_qc[description][channel]["post_trial_fraction_nan_trials"] = (
                record["fraction_nan_trials"] if record is not None else None
            )
    removed = {
        channel: {
            description: initial_qc[description][channel]
            for description in LEXICAL_CONDITIONS
        }
        for channel in candidates
        if channel not in retained
    }
    filtered = {
        description: filtered[description].sel(channel=retained)
        for description in LEXICAL_CONDITIONS
    }
    return filtered, candidates, retained, initial_qc, removed


def single_condition_qc(
    X: xr.DataArray,
    threshold: float = QC_THRESHOLD,
) -> tuple[xr.DataArray, list[str], list[str], dict[str, object], dict[str, object]]:
    candidates = sorted(str(value) for value in X["channel"].values)
    aligned = X.reindex(channel=candidates)
    qc = _channel_qc(aligned, candidates)
    active = [
        channel
        for channel in candidates
        if qc[channel]["present"] and qc[channel]["fraction_nan_trials"] <= threshold
    ]
    history: dict[str, dict[str, object]] = {}
    filtered = aligned.isel(channel=[])
    for _ in range(len(candidates) + 1):
        if not active:
            break
        filtered = _drop_bad_trials(aligned.sel(channel=active), threshold)
        post_qc = _channel_qc(filtered, active)
        history.update(post_qc)
        next_active = [
            channel
            for channel in active
            if post_qc[channel]["present"]
            and post_qc[channel]["fraction_nan_trials"] <= threshold
        ]
        if next_active == active:
            break
        active = next_active
    else:
        raise RuntimeError("Single-condition channel/trial QC did not converge")
    retained = list(active)
    for channel in candidates:
        record = history.get(channel)
        qc[channel]["post_trial_fraction_nan_trials"] = (
            record["fraction_nan_trials"] if record is not None else None
        )
    removed = {channel: qc[channel] for channel in candidates if channel not in retained}
    return filtered.sel(channel=retained), candidates, retained, qc, removed


@lru_cache(maxsize=4096)
def _word_to_phonemes(word: str) -> tuple[str, ...]:
    global _G2P
    if _G2P is None:
        from g2p_en import G2p

        _G2P = G2p()
    phones = [value for value in _G2P(word) if value not in {" ", ""}]
    return tuple(
        value.rstrip("012")
        for value in phones
        if isinstance(value, str) and value.rstrip("012")
    )


def lexical_phoneme(stimulus: str) -> str | None:
    word = str(stimulus).split("/")[-1].lower()
    phones = _word_to_phonemes(word) if word else ()
    return phones[0] if phones else None


def lexical_articulator(stimulus: str) -> str | None:
    phoneme = lexical_phoneme(stimulus)
    return ARTICULATORY_PHONE_TO_LABEL.get(phoneme.upper()) if phoneme else None


def sequence_phoneme(stimulus: str) -> str | None:
    syllable = str(stimulus).split("/")[-1]
    if not syllable:
        return None
    return "ae" if syllable.startswith("ae") else syllable[0]


def sequence_articulator(stimulus: str) -> str:
    phoneme = sequence_phoneme(stimulus)
    if phoneme in {"a", "ae"}:
        return "low_vowel"
    if phoneme in {"i", "u"}:
        return "high_vowel"
    if phoneme in {"b", "p", "v"}:
        return "labial"
    if phoneme in {"g", "k"}:
        return "dorsal"
    return "other"


def prepare_feature(
    bids_task: str,
    feature: str,
    frame: pd.DataFrame,
    X: xr.DataArray,
) -> PreparedFeature:
    trials = [str(value) for value in X["trial"].values]
    conditions = [value.rsplit("_", 1)[0] for value in trials]
    if feature == "lexicality":
        lexicality_counts = frame.groupby("trial")["lexicality"].nunique()
        if (lexicality_counts > 1).any():
            bad = lexicality_counts[lexicality_counts > 1].index.tolist()
            raise ValueError(f"Conflicting lexicality labels for trials: {bad[:10]}")
        trial_to_label = (
            frame[["trial", "lexicality"]]
            .drop_duplicates()
            .set_index("trial")["lexicality"]
            .to_dict()
        )
        labels = [trial_to_label.get(trial) for trial in trials]
        valid = np.asarray([label in {"Word", "Nonword"} for label in labels])
        class_order = ["Word", "Nonword"]
    else:
        if bids_task == "LexicalDelay":
            labeler = lexical_phoneme if feature == "phoneme" else lexical_articulator
        elif bids_task == "PhonemeSequence":
            labeler = sequence_phoneme if feature == "phoneme" else sequence_articulator
        else:
            raise ValueError(f"Unsupported BIDS task: {bids_task}")
        labels = [labeler(condition) for condition in conditions]
        valid = np.asarray([label is not None and label != "" for label in labels])
        class_order = sorted({str(label) for label in labels if label is not None})

    keep = np.flatnonzero(valid)
    X = X.isel(trial=keep)
    trials = [trials[index] for index in keep]
    conditions = [conditions[index] for index in keep]
    labels = [str(labels[index]) for index in keep]
    observed = set(labels)
    class_order = [label for label in class_order if label in observed]
    if len(class_order) < 2:
        raise ValueError(f"{feature} has fewer than two classes: {class_order}")
    event_id = {label: index for index, label in enumerate(class_order)}
    y = np.asarray([event_id[label] for label in labels], dtype=int)
    if X.shape[0] != len(y):
        raise ValueError(f"X/y mismatch for {feature}: {X.shape[0]} vs {len(y)}")
    return PreparedFeature(X, y, trials, conditions, labels, event_id)


def pair_cross_conditions(
    repeat: PreparedFeature,
    decision: PreparedFeature,
) -> dict[str, PreparedFeature]:
    """Pair Repeat and Decision by trial ID for direct cross-decoding."""
    decision_index = {trial: index for index, trial in enumerate(decision.trials)}
    common = [trial for trial in repeat.trials if trial in decision_index]
    if not common:
        raise ValueError("Repeat and Decision have no common trial IDs")
    repeat_index = {trial: index for index, trial in enumerate(repeat.trials)}
    rep_indices = [repeat_index[trial] for trial in common]
    dec_indices = [decision_index[trial] for trial in common]
    rep_labels = [repeat.labels[index] for index in rep_indices]
    dec_labels = [decision.labels[index] for index in dec_indices]
    if rep_labels != dec_labels:
        mismatch = [
            trial
            for trial, left, right in zip(common, rep_labels, dec_labels)
            if left != right
        ]
        raise ValueError(f"Cross-condition labels differ for trials: {mismatch[:10]}")
    if repeat.event_id != decision.event_id:
        raise ValueError("Cross-condition event_id mappings differ")
    rep_conditions = [repeat.conditions[index] for index in rep_indices]
    dec_conditions = [decision.conditions[index] for index in dec_indices]
    return {
        "Repeat": PreparedFeature(
            repeat.X.isel(trial=rep_indices),
            repeat.y[rep_indices],
            common,
            rep_conditions,
            rep_labels,
            repeat.event_id,
        ),
        "Decision": PreparedFeature(
            decision.X.isel(trial=dec_indices),
            decision.y[dec_indices],
            common,
            dec_conditions,
            dec_labels,
            decision.event_id,
        ),
    }


def _output_path(
    root: Path,
    bids_task: str,
    pseudo_subject: str,
    feature: str,
    description: str,
    phase: str,
    band: str,
) -> Path:
    bids_path = BIDSPath(
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
    return Path(bids_path.fpath)


def _atomic_write_h5(
    target: Path,
    prepared: PreparedFeature,
    *,
    attrs: Mapping[str, object],
    candidate_channels: list[str],
    retained_channels: list[str],
    channel_qc: object,
    removed_channels: object,
    overwrite: bool,
) -> None:
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {target}; pass --overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    string_dtype = h5py.string_dtype(encoding="utf-8")
    try:
        with h5py.File(temporary, "w") as stream:
            stream.create_dataset("X", data=prepared.X.values)
            stream.create_dataset("y", data=prepared.y)
            stream.create_dataset(
                "trial", data=np.asarray(prepared.trials, dtype=string_dtype)
            )
            stream.create_dataset(
                "condition", data=np.asarray(prepared.conditions, dtype=string_dtype)
            )
            stream.create_dataset(
                "label", data=np.asarray(prepared.labels, dtype=string_dtype)
            )
            stream.create_dataset(
                "channel", data=np.asarray(retained_channels, dtype=string_dtype)
            )
            stream.create_dataset("time", data=np.asarray(prepared.X["time"].values))
            stream.attrs["event_id"] = json.dumps(prepared.event_id, sort_keys=True)
            stream.attrs["tmin"] = float(prepared.X["time"].min())
            stream.attrs["tmax"] = float(prepared.X["time"].max())
            stream.attrs["fs"] = 128
            stream.attrs["candidate_channels_json"] = json.dumps(candidate_channels)
            stream.attrs["retained_channels_json"] = json.dumps(retained_channels)
            stream.attrs["channel_qc_json"] = json.dumps(channel_qc, sort_keys=True)
            stream.attrs["removed_channels_json"] = json.dumps(
                removed_channels, sort_keys=True
            )
            stream.attrs["n_candidate_channels"] = len(candidate_channels)
            stream.attrs["n_retained_channels"] = len(retained_channels)
            for key, value in attrs.items():
                if isinstance(value, (dict, list, tuple, set)):
                    stream.attrs[key] = json.dumps(value, sort_keys=True)
                else:
                    stream.attrs[key] = value
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _manifest_record(
    *,
    kind: str,
    target: Path,
    bids_task: str,
    pseudo_subject: str,
    cluster: str,
    hemi: str,
    phase: str,
    description: str,
    feature: str,
    assignment_count: int,
    significant_assigned_count: int,
    candidates: list[str],
    retained: list[str],
    prepared: PreparedFeature,
    status: str,
) -> dict[str, object]:
    class_counts = {
        label: int(sum(value == label for value in prepared.labels))
        for label in sorted(set(prepared.labels))
    }
    return {
        "kind": kind,
        "path": str(target),
        "bids_task": bids_task,
        "pseudo_subject": pseudo_subject,
        "functional_cluster": cluster,
        "hemi": hemi,
        "phase": phase,
        "description": description,
        "feature": feature,
        "n_assignment_channels": assignment_count,
        "n_significant_assigned_channels": significant_assigned_count,
        "n_candidate_channels": len(candidates),
        "n_retained_channels": len(retained),
        "n_trials": len(prepared.y),
        "n_classes": len(class_counts),
        "class_counts_json": json.dumps(class_counts, sort_keys=True),
        "status": status,
    }


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
    intersection_root: Path | None,
    dry_run: bool,
    overwrite: bool,
) -> list[dict[str, object]]:
    group_assignments = assignments[assignments["pseudo_subject"].eq(pseudo_subject)]
    cluster = str(group_assignments["functional_cluster"].iloc[0])
    hemis = sorted({str(value).upper() for value in group_assignments["hemi"]})
    hemi = "LR" if set(hemis) >= {"L", "R"} else (hemis[0] if hemis else "NA")
    descriptions = LEXICAL_CONDITIONS if bids_task == "LexicalDelay" else ("Repeat",)
    frames: list[pd.DataFrame] = []
    selection_stats = []

    for subject in sorted(set(group_assignments["subject"])):
        sig_by_description = {
            description: _read_sig_channels(
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
            include = select_assigned_channels(
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
                    and channel in PHONEME_SEQUENCE_EXCLUDE_CHANNELS
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
            if bids_task == "LexicalDelay":
                frame = _lexical_frame(epochs, subject, description, phase)
            else:
                frame = _phoneme_sequence_frame(
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
    if bids_task == "LexicalDelay":
        filtered, candidates, retained, channel_qc, removed = symmetric_condition_qc(
            arrays
        )
    else:
        X, candidates, retained, channel_qc, removed = single_condition_qc(
            arrays["Repeat"]
        )
        filtered = {"Repeat": X}
    if not retained:
        raise RuntimeError(
            f"No channels survive QC for {pseudo_subject} {bids_task} {phase}"
        )

    features = (
        LEXICAL_FEATURES
        if bids_task == "LexicalDelay"
        else PHONEME_SEQUENCE_FEATURES
    )
    prepared_by_description: dict[str, dict[str, PreparedFeature]] = {
        description: {} for description in descriptions
    }
    records: list[dict[str, object]] = []
    # ``candidates`` is the exact assignment ∩ phase-significance ∩ available
    # set before NaN QC; it is therefore the relevant significant-assigned count.
    significant_assigned = len(candidates)
    common_attrs = {
        "atlas": "hammers",
        "input_datatype": ZSCORE_DATATYPE,
        "significance_datatype": SIG_DATATYPE,
        "channel_selection": (
            "functional_assignment_intersection_same_phase_decision_repeat_sig_union"
            if bids_task == "LexicalDelay"
            else "functional_assignment_intersection_same_phase_repeat_sig"
        ),
        "assignment_path": str(assignment_path.resolve()),
        "assignment_sha256": assignment_sha256,
        "assignment_rows": len(assignments),
        "functional_cluster": cluster,
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
            prepared_by_description[description][feature] = prepared
            target = _output_path(
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
                _atomic_write_h5(
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
                _manifest_record(
                    kind="regular",
                    target=target,
                    bids_task=bids_task,
                    pseudo_subject=pseudo_subject,
                    cluster=cluster,
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

    if (
        bids_task == "LexicalDelay"
        and phase == "Delay"
        and intersection_root is not None
    ):
        paired = pair_cross_conditions(
            prepared_by_description["Repeat"]["lexicality"],
            prepared_by_description["Decision"]["lexicality"],
        )
        for description in ("Repeat", "Decision"):
            prepared = paired[description]
            target = _output_path(
                intersection_root,
                bids_task,
                pseudo_subject,
                "lexicality",
                description,
                phase,
                band,
            )
            attrs = {
                **common_attrs,
                "roi": pseudo_subject,
                "description": description,
                "task": bids_task,
                "feature": "lexicality",
                "dataset_kind": "intersection",
                "trial_pairing": "exact_trial_id_intersection_repeat_order",
            }
            if not dry_run:
                _atomic_write_h5(
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
                _manifest_record(
                    kind="intersection",
                    target=target,
                    bids_task=bids_task,
                    pseudo_subject=pseudo_subject,
                    cluster=cluster,
                    hemi=hemi,
                    phase=phase,
                    description=description,
                    feature="lexicality",
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
    assignment_sha256 = file_sha256(assignment_path)
    if len(assignments) != args.expected_assignment_rows:
        raise ValueError(
            f"Expected {args.expected_assignment_rows} assignments, found {len(assignments)}"
        )

    zscore_index = _epoch_index(
        bids_root, args.reference, args.band, args.bids_task, ZSCORE_DATATYPE
    )
    sig_index = _epoch_index(
        bids_root, args.reference, args.band, args.bids_task, SIG_DATATYPE
    )
    descriptions = LEXICAL_CONDITIONS if args.bids_task == "LexicalDelay" else ("Repeat",)
    expected_zscore = {
        (subject, description, phase)
        for subject in assignments["subject"].unique()
        for description in descriptions
        for phase in PHASES
        if (subject, description, phase) in zscore_index
    }
    if not expected_zscore:
        raise RuntimeError(f"No zscore epochs match {args.bids_task}")

    regular_root = Path(args.output_root) if args.output_root else (
        bids_root / "derivatives" / f"decoding({args.reference})"
    )
    intersection_root = None
    if args.bids_task == "LexicalDelay":
        intersection_root = (
            Path(args.intersection_output_root)
            if args.intersection_output_root
            else bids_root
            / "derivatives"
            / f"decoding(intersection)({args.reference})"
        )

    records: list[dict[str, object]] = []
    errors: list[str] = []
    for pseudo_subject in PSEUDO_SUBJECTS:
        for phase in PHASES:
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
                        intersection_root=intersection_root,
                        dry_run=args.dry_run,
                        overwrite=args.overwrite,
                    )
                )
            except Exception as error:  # keep a complete phase/group audit
                message = f"{pseudo_subject} {args.bids_task} {phase}: {error}"
                error_text = str(error)
                if error_text.startswith("No selected data for "):
                    LOGGER.warning("Skipping empty group/phase: %s", message)
                    if args.overwrite and not args.dry_run:
                        features = (
                            LEXICAL_FEATURES
                            if args.bids_task == "LexicalDelay"
                            else PHONEME_SEQUENCE_FEATURES
                        )
                        for feature in features:
                            for description in descriptions:
                                stale = _output_path(
                                    regular_root,
                                    args.bids_task,
                                    pseudo_subject,
                                    feature,
                                    description,
                                    phase,
                                    args.band,
                                )
                                if stale.exists():
                                    stale.unlink()
                                    LOGGER.info("Removed stale empty-cell output: %s", stale)
                    records.append(
                        {
                            "kind": "skipped",
                            "bids_task": args.bids_task,
                            "pseudo_subject": pseudo_subject,
                            "phase": phase,
                            "status": "skipped",
                            "error": error_text,
                        }
                    )
                    continue
                LOGGER.exception(message)
                errors.append(message)
                records.append(
                    {
                        "kind": "error",
                        "bids_task": args.bids_task,
                        "pseudo_subject": pseudo_subject,
                        "phase": phase,
                        "status": "error",
                        "error": error_text,
                    }
                )

    manifest = pd.DataFrame(records)
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else PROJECT_ROOT
        / "results"
        / "decoding_functional"
        / f"{args.bids_task}_prepare_manifest.csv"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}")
    manifest.to_csv(temporary, index=False)
    os.replace(temporary, manifest_path)
    LOGGER.info("Wrote manifest: %s", manifest_path)
    if errors:
        raise RuntimeError(
            f"Functional preparation failed for {len(errors)} group/phase combinations; "
            f"see {manifest_path}"
        )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", required=True)
    parser.add_argument(
        "--bids-task", required=True, choices=("LexicalDelay", "PhonemeSequence")
    )
    parser.add_argument(
        "--assignments",
        default=str(PROJECT_ROOT / "results" / "nmf" / "channel_assignments.csv"),
    )
    parser.add_argument("--reference", default="bipolar", choices=("bipolar", "car"))
    parser.add_argument("--band", default="highgamma")
    parser.add_argument("--output-root")
    parser.add_argument("--intersection-output-root")
    parser.add_argument("--manifest")
    parser.add_argument("--expected-assignment-rows", type=int, default=255)
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
