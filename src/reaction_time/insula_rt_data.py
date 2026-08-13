"""Load strict-insula HGA trials and align them to Response-minus-Go RT."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from src.reaction_time.insula_ridge import parse_item_id


STRICT_INSULA_ROIS = ("AIC", "PIC")


class NoStrictInsulaError(RuntimeError):
    """Raised when a subject has no usable strict AIC/PIC electrodes."""


@dataclass(frozen=True)
class PhaseData:
    """One subject/phase dataset ready for time-resolved modelling."""

    X: np.ndarray  # trials x channels x time
    times: np.ndarray
    sfreq: float
    trial_meta: pd.DataFrame
    channel_meta: pd.DataFrame
    task: str
    subject: str
    phase: str
    description: str


def epoch_data_dir(bids_root: Path | str, subject: str, ref: str) -> Path:
    subject = subject[4:] if str(subject).startswith("sub-") else str(subject)
    return (
        Path(bids_root)
        / "derivatives"
        / f"epoch({ref})"
        / f"sub-{subject}"
        / "epoch(band)(zscore)"
    )


def find_phase_files(
    bids_root: Path | str,
    *,
    subject: str,
    phase: str,
    description: str,
    band: str,
    ref: str,
) -> list[Path]:
    """Find all recordings for one subject/phase/description."""

    root = epoch_data_dir(bids_root, subject, ref)
    pattern = f"*_proc-{phase}_*desc-{description}_{band}.h5"
    return sorted(root.glob(pattern))


def sibling_phase_path(path: Path | str, source_phase: str, target_phase: str) -> Path:
    path = Path(path)
    marker = f"_proc-{source_phase}_"
    if marker not in path.name:
        raise ValueError(f"Cannot find {marker!r} in {path.name!r}")
    return path.with_name(path.name.replace(marker, f"_proc-{target_phase}_", 1))


def _event_names(epochs: mne.Epochs) -> list[str]:
    inverse = {int(value): str(name) for name, value in epochs.event_id.items()}
    return [inverse.get(int(code), str(code)) for code in epochs.events[:, 2]]


def _recording_from_path(path: Path) -> str:
    match = re.search(r"_recording-([^_]+)", path.name)
    return match.group(1) if match else "default"


def _phase_event_table(
    path: Path,
    *,
    task: str,
    raw_sfreq: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    epochs = mne.read_epochs(path, preload=False, verbose="error")
    order = np.argsort(epochs.events[:, 0], kind="stable")
    event_names = np.asarray(_event_names(epochs), dtype=object)[order]
    samples = epochs.events[order, 0].astype(float)
    frame = pd.DataFrame(
        {
            "trial_index": np.arange(len(order), dtype=int),
            "event_sample": samples,
            "onset": samples / float(raw_sfreq),
            "event_name": event_names.astype(str),
        }
    )
    frame["item_id"] = [parse_item_id(task, value) for value in event_names]
    frame["is_correct"] = frame["event_name"].str.contains("CORRECT", na=False)
    return frame, order


def match_target_go_response(
    target: pd.DataFrame,
    go: pd.DataFrame,
    response: pd.DataFrame,
    *,
    phase: str,
) -> pd.DataFrame:
    """Match target events to Go/Response without relying on phase row number."""

    go_samples = go["event_sample"].to_numpy(dtype=float)
    response_samples = response["event_sample"].to_numpy(dtype=float)
    matched_rows: list[dict[str, object]] = []
    for _, row in target.iterrows():
        target_sample = float(row["target_event_sample"])
        go_position = int(np.searchsorted(go_samples, target_sample, side="left"))
        if go_position >= len(go_samples):
            continue
        if phase == "Go" and go_samples[go_position] != target_sample:
            continue
        go_row = go.iloc[go_position]
        if str(go_row["item_id"]) != str(row["item_id"]):
            continue

        go_sample = float(go_row["event_sample"])
        next_go_sample = (
            float(go_samples[go_position + 1])
            if go_position + 1 < len(go_samples)
            else np.inf
        )
        response_position = int(
            np.searchsorted(response_samples, go_sample, side="right")
        )
        if response_position >= len(response_samples):
            continue
        response_row = response.iloc[response_position]
        response_sample = float(response_row["event_sample"])
        if response_sample >= next_go_sample:
            continue
        if str(response_row["item_id"]) != str(row["item_id"]):
            continue

        matched = row.to_dict()
        matched.update(
            {
                "go_event_sample": go_sample,
                "go_onset": float(go_row["onset"]),
                "go_event_name": str(go_row["event_name"]),
                "response_event_sample": response_sample,
                "response_onset": float(response_row["onset"]),
                "response_event_name": str(response_row["event_name"]),
            }
        )
        matched_rows.append(matched)
    return pd.DataFrame(matched_rows)


def _aligned_trial_table(
    target_path: Path,
    *,
    task: str,
    phase: str,
    raw_sfreq: float,
    rt_min_s: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Align target epochs to Go/Response using absolute event samples.

    Epoch rejection can differ by phase, so row-number matching is unsafe.  A
    Delay trial is paired with the next Go event in the recording.  Its
    Response must fall after that Go and before the following Go.  Item ids are
    required to agree at every matched event.
    """

    target, target_order = _phase_event_table(
        target_path, task=task, raw_sfreq=raw_sfreq
    )
    target = target.rename(
        columns={
            "event_name": "target_event_name",
            "event_sample": "target_event_sample",
            "onset": "target_onset",
            "is_correct": "target_is_correct",
        }
    )
    target["source_row"] = target_order
    go_path = sibling_phase_path(target_path, phase, "Go")
    response_path = sibling_phase_path(target_path, phase, "Response")
    if not go_path.is_file():
        raise FileNotFoundError(go_path)
    if not response_path.is_file():
        raise FileNotFoundError(response_path)
    go, _ = _phase_event_table(go_path, task=task, raw_sfreq=raw_sfreq)
    response, _ = _phase_event_table(
        response_path, task=task, raw_sfreq=raw_sfreq
    )
    target = match_target_go_response(target, go, response, phase=phase)
    if target.empty:
        raise ValueError(f"No target/Go/Response trials aligned for {target_path}")
    target["rt_raw"] = target["response_onset"] - target["go_onset"]
    target["recording"] = _recording_from_path(target_path)
    target["trial_uid"] = (
        target["recording"].astype(str)
        + ":"
        + target["trial_index"].astype(str)
    )
    finite = np.isfinite(target["rt_raw"].to_numpy(dtype=float))
    valid = finite & (target["rt_raw"].to_numpy(dtype=float) >= float(rt_min_s))
    if target["target_is_correct"].any():
        valid &= target["target_is_correct"].to_numpy(dtype=bool)
    target = target.loc[valid].reset_index(drop=True)
    target["rt_log"] = np.log(target["rt_raw"].to_numpy(dtype=float))
    return target, target_order


def load_strict_insula_parcellation(
    bids_root: Path | str,
    *,
    subject: str,
    ref: str = "bipolar",
    atlas: str = "hammers",
) -> pd.DataFrame:
    """Load strict AIC/PIC bipolar contacts, preserving native/template coords."""

    subject = subject[4:] if str(subject).startswith("sub-") else str(subject)
    root = Path(bids_root) / "derivatives" / "parcellation" / f"sub-{subject}" / ref
    matches = sorted(root.glob(f"*_{atlas}.csv"))
    if not matches:
        raise FileNotFoundError(f"No {atlas} parcellation CSV under {root}")
    parcellation = pd.read_csv(matches[0])
    required = {"name", "roi", "hemi"}
    missing = required.difference(parcellation.columns)
    if missing:
        raise ValueError(f"Parcellation missing columns: {sorted(missing)}")
    out = parcellation[parcellation["roi"].isin(STRICT_INSULA_ROIS)].copy()
    out = out.drop_duplicates("name", keep="first").rename(columns={"name": "channel"})

    # Canonical plotting coordinates use the existing template-transformed xyz
    # when available.  Native and MNI coordinates remain available separately.
    for axis in "xyz":
        template = f"{axis}_t"
        out[f"{axis}_template"] = (
            out[template] if template in out.columns else out.get(axis, np.nan)
        )
        out[f"{axis}_native"] = out.get(axis, np.nan)
        out[f"{axis}_mni"] = out.get(f"{axis}_mni", np.nan)
    for column in ("label", "center", "mix"):
        if column not in out.columns:
            out[column] = "" if column != "mix" else False
    columns = [
        "channel",
        "roi",
        "hemi",
        "label",
        "center",
        "mix",
        "x_template",
        "y_template",
        "z_template",
        "x_native",
        "y_native",
        "z_native",
        "x_mni",
        "y_mni",
        "z_mni",
    ]
    return out[columns].reset_index(drop=True)


def load_phase_data(
    bids_root: Path | str,
    *,
    task: str,
    subject: str,
    phase: str,
    description: str = "Repeat",
    band: str = "highgamma",
    ref: str = "bipolar",
    atlas: str = "hammers",
    raw_sfreq: float = 2048.0,
    rt_min_s: float = 0.05,
) -> PhaseData:
    """Load all recordings for one subject/phase and retain strict AIC/PIC."""

    subject = subject[4:] if str(subject).startswith("sub-") else str(subject)
    paths = find_phase_files(
        bids_root,
        subject=subject,
        phase=phase,
        description=description,
        band=band,
        ref=ref,
    )
    if not paths:
        raise FileNotFoundError(
            f"No {phase}/{description}/{band} epochs for sub-{subject}"
        )
    channel_meta = load_strict_insula_parcellation(
        bids_root, subject=subject, ref=ref, atlas=atlas
    )
    if channel_meta.empty:
        raise NoStrictInsulaError(f"sub-{subject} has no strict AIC/PIC electrodes")

    loaded: list[tuple[mne.Epochs, pd.DataFrame]] = []
    present = set(channel_meta["channel"].astype(str))
    for path in paths:
        epochs = mne.read_epochs(path, preload=True, verbose="error")
        present &= set(epochs.ch_names)
        trial_meta, _ = _aligned_trial_table(
            path,
            task=task,
            phase=phase,
            raw_sfreq=raw_sfreq,
            rt_min_s=rt_min_s,
        )
        trial_meta["source_file"] = str(path)
        loaded.append((epochs, trial_meta))
    channels = [
        channel for channel in channel_meta["channel"].astype(str) if channel in present
    ]
    if not channels:
        raise NoStrictInsulaError(
            f"sub-{subject} has no strict insula channels in all {phase} files"
        )
    channel_meta = (
        channel_meta.set_index("channel").loc[channels].reset_index()
    )

    arrays: list[np.ndarray] = []
    metadata: list[pd.DataFrame] = []
    reference_times: np.ndarray | None = None
    reference_sfreq: float | None = None
    for epochs, trial_meta in loaded:
        if reference_times is None:
            reference_times = epochs.times.copy()
            reference_sfreq = float(epochs.info["sfreq"])
        elif not np.allclose(reference_times, epochs.times, atol=1e-10, rtol=0):
            raise ValueError(f"Inconsistent epoch times across {phase} recordings")
        source_rows = trial_meta["source_row"].to_numpy(dtype=int)
        channel_indices = [epochs.ch_names.index(channel) for channel in channels]
        data = epochs.get_data()[source_rows][:, channel_indices, :]
        arrays.append(np.asarray(data, dtype=float))
        metadata.append(trial_meta)

    X = np.concatenate(arrays, axis=0)
    trials = pd.concat(metadata, ignore_index=True)
    if X.shape[:2] != (len(trials), len(channel_meta)):
        raise RuntimeError("HGA/trial/channel alignment failed")
    if trials["trial_uid"].duplicated().any():
        raise RuntimeError("trial_uid is not unique after concatenating recordings")
    return PhaseData(
        X=X,
        times=np.asarray(reference_times, dtype=float),
        sfreq=float(reference_sfreq),
        trial_meta=trials,
        channel_meta=channel_meta,
        task=task,
        subject=subject,
        phase=phase,
        description=description,
    )
