"""BIDS discovery, input validation, and aligned analysis loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Mapping

import mne
import numpy as np
import pandas as pd

from .config import DEFAULT_DATASETS, phase_time_mask
from .pairs import enumerate_insula_to_all_pairs
from .seeds import strict_hammers_seed_frame


ENTITY_PATTERN = re.compile(
    r"(?:^|_)(sub|ses|task|acq|run|proc|recording|rec|desc)-([^_]+)"
)


def parse_filename_entities(path: str | Path) -> dict[str, str]:
    """Parse the BIDS-like entities used by the existing HDF5 derivatives."""

    filename = Path(path).name
    entities = {key: value for key, value in ENTITY_PATTERN.findall(filename)}
    if "sub" not in entities:
        subject_match = re.search(r"sub-([^_/]+)", str(path))
        if subject_match:
            entities["sub"] = subject_match.group(1)
    return entities


def _replace_hga_suffix(filename: str) -> str:
    if not filename.endswith("_highgamma.h5"):
        raise ValueError(f"Expected highgamma HDF5 filename, got {filename}")
    return filename[: -len("_highgamma.h5")] + "_raw.h5"


def matching_input_paths(zscore_path: str | Path) -> dict[str, Path]:
    """Resolve raw, effective, and Hammers files for one zscore epoch."""

    zscore = Path(zscore_path)
    if zscore.parent.name != "epoch(band)(zscore)":
        raise ValueError(f"Not an epoch(band)(zscore) path: {zscore}")
    subject_dir = zscore.parent.parent
    entities = parse_filename_entities(zscore)
    subject = entities.get("sub")
    if not subject:
        raise ValueError(f"Cannot parse subject from {zscore}")
    raw = subject_dir / "epoch(raw)" / _replace_hga_suffix(zscore.name)
    effective = (
        subject_dir / "epoch(band)(sig)(effective)" / zscore.name
    )
    derivatives = subject_dir.parent.parent
    hammers = (
        derivatives
        / "parcellation"
        / f"sub-{subject}"
        / "bipolar"
        / f"sub-{subject}_hammers.csv"
    )
    return {
        "zscore_path": zscore,
        "raw_path": raw,
        "effective_path": effective,
        "hammers_path": hammers,
    }


def discover_manifest(
    dataset_roots: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Discover every all-channel HGA entity and its required inputs."""

    roots = DEFAULT_DATASETS if dataset_roots is None else dataset_roots
    rows: list[dict[str, object]] = []
    for dataset_name, root_value in roots.items():
        root = Path(root_value)
        epoch_root = root / "derivatives" / "epoch(bipolar)"
        for zscore in sorted(
            epoch_root.glob(
                "sub-*/epoch(band)(zscore)/*_highgamma.h5"
            )
        ):
            paths = matching_input_paths(zscore)
            entities = parse_filename_entities(zscore)
            row: dict[str, object] = {
                "dataset": dataset_name,
                "bids_root": str(root),
                "subject": entities.get("sub", ""),
                "task": entities.get("task", dataset_name),
                "phase": entities.get("proc", ""),
                "description": entities.get("desc", ""),
                "recording": entities.get(
                    "recording", entities.get("rec", "")
                ),
                "run": entities.get("run", ""),
                "acquisition": entities.get("acq", ""),
                **{key: str(value) for key, value in paths.items()},
            }
            missing = [
                key
                for key in ("raw_path", "hammers_path")
                if not Path(str(row[key])).exists()
            ]
            row["effective_annotation_available"] = paths[
                "effective_path"
            ].exists()
            row["status"] = "ready" if not missing else "missing_input"
            row["reason"] = "" if not missing else ",".join(missing)
            rows.append(row)
    columns = [
        "dataset",
        "bids_root",
        "subject",
        "task",
        "phase",
        "description",
        "recording",
        "run",
        "acquisition",
        "zscore_path",
        "raw_path",
        "effective_path",
        "hammers_path",
        "effective_annotation_available",
        "status",
        "reason",
    ]
    return pd.DataFrame(rows, columns=columns)


def write_manifest(frame: pd.DataFrame, output: str | Path) -> Path:
    """Write a deterministic TSV manifest."""

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + ".tmp")
    frame.to_csv(temp, sep="\t", index=False)
    temp.replace(destination)
    return destination


def read_manifest_row(manifest: str | Path, row_index: int) -> pd.Series:
    frame = pd.read_csv(manifest, sep="\t", keep_default_na=False)
    if not 0 <= row_index < len(frame):
        raise IndexError(
            f"row_index={row_index} outside manifest with {len(frame)} rows"
        )
    return frame.iloc[row_index]


def input_fingerprint(path: str | Path) -> dict[str, object]:
    source = Path(path)
    stat = source.stat()
    return {
        "method": "path_size_mtime_ns",
        "path": str(source),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


@dataclass
class AnalysisData:
    """Signals and metadata aligned across all three estimators."""

    entities: dict[str, str]
    zscore_path: Path
    raw_path: Path
    hammers_path: Path
    effective_path: Path | None
    channel_names: list[str]
    hga_data: np.ndarray
    hga_times: np.ndarray
    hga_sfreq: float
    raw_data: np.ndarray
    raw_times: np.ndarray
    raw_sfreq: float
    events: np.ndarray
    trial_indices: np.ndarray
    pair_frame: pd.DataFrame
    seed_frame: pd.DataFrame
    parcellation: pd.DataFrame
    n_original_trials: int
    dropped_trials: list[int]
    dropped_channels: list[str]
    n_eligible_pairs_before_limit: int

    @property
    def n_trials(self) -> int:
        return int(self.hga_data.shape[0])

    @property
    def n_channels(self) -> int:
        return len(self.channel_names)


def _same_events(first: mne.Epochs, second: mne.Epochs) -> bool:
    return (
        first.events.shape == second.events.shape
        and np.array_equal(first.events, second.events)
        and first.event_id == second.event_id
    )


def _same_channel_names(first: mne.Epochs, second: mne.Epochs) -> bool:
    return list(first.ch_names) == list(second.ch_names)


def _effective_channels(path: Path | None) -> tuple[set[str], bool]:
    if path is None or not path.exists():
        return set(), False
    epochs = mne.read_epochs(path, preload=False, verbose="error")
    return set(epochs.ch_names), True


def _channel_is_usable(hga: np.ndarray, raw: np.ndarray) -> np.ndarray:
    hga_variance = np.nanvar(hga, axis=(0, 2))
    raw_variance = np.nanvar(raw, axis=(0, 2))
    return (
        np.isfinite(hga_variance)
        & np.isfinite(raw_variance)
        # Raw MNE voltage is expressed in volts, so physiological variance can
        # legitimately be far below the dimensionless float32 epsilon.
        & (hga_variance > 0.0)
        & (raw_variance > 0.0)
    )


def load_analysis_data(
    row: Mapping[str, object] | pd.Series,
    *,
    min_trials: int = 30,
    pair_limit: int | None = None,
) -> AnalysisData:
    """Load and align all inputs for one manifest entity."""

    zscore_path = Path(str(row["zscore_path"]))
    raw_path = Path(str(row["raw_path"]))
    hammers_path = Path(str(row["hammers_path"]))
    effective_value = str(row.get("effective_path", ""))
    effective_path = Path(effective_value) if effective_value else None
    for required in (zscore_path, raw_path, hammers_path):
        if not required.exists():
            raise FileNotFoundError(required)

    zscore_epochs = mne.read_epochs(
        zscore_path, preload=True, verbose="error"
    )
    raw_epochs = mne.read_epochs(raw_path, preload=True, verbose="error")
    if not _same_events(zscore_epochs, raw_epochs):
        raise ValueError(
            f"Event mismatch between {zscore_path} and {raw_path}"
        )

    phase = str(row.get("phase") or parse_filename_entities(zscore_path).get(
        "proc", ""
    ))
    hga_mask = phase_time_mask(zscore_epochs.times, phase)
    if not _same_channel_names(zscore_epochs, raw_epochs):
        zscore_only = sorted(
            set(zscore_epochs.ch_names).difference(raw_epochs.ch_names)
        )
        raw_only = sorted(
            set(raw_epochs.ch_names).difference(zscore_epochs.ch_names)
        )
        raise ValueError(
            "Channel-name/order mismatch between zscore and raw epochs; "
            f"zscore_only={zscore_only}, raw_only={raw_only}"
        )
    common_channels = list(zscore_epochs.ch_names)

    hga = zscore_epochs.get_data(copy=True)[..., hga_mask].astype(
        np.float32, copy=False
    )
    raw = raw_epochs.get_data(copy=True).astype(np.float32, copy=False)
    usable = _channel_is_usable(hga, raw)
    dropped_channels = [
        channel
        for channel, keep in zip(common_channels, usable)
        if not keep
    ]
    channel_names = [
        channel
        for channel, keep in zip(common_channels, usable)
        if keep
    ]
    if not channel_names:
        raise ValueError("No usable channels after finite/variance QC")
    hga = hga[:, usable]
    raw = raw[:, usable]

    valid_trials = np.isfinite(hga).all(axis=(1, 2))
    valid_trials &= np.isfinite(raw).all(axis=(1, 2))
    trial_indices = np.flatnonzero(valid_trials)
    if trial_indices.size < min_trials:
        raise ValueError(
            f"Only {trial_indices.size} valid trials; min_trials={min_trials}"
        )
    dropped_trials = np.flatnonzero(~valid_trials).astype(int).tolist()
    hga = hga[valid_trials]
    raw = raw[valid_trials]

    parcellation = pd.read_csv(hammers_path)
    seed_frame = strict_hammers_seed_frame(parcellation, channel_names)
    if seed_frame.empty:
        raise ValueError("No strict two-endpoint Hammers Insula seeds")
    effective_channels, effective_available = _effective_channels(
        effective_path
    )
    pair_frame = enumerate_insula_to_all_pairs(
        channel_names,
        seed_frame,
        parcellation,
        effective_channels=effective_channels,
        effective_annotation_available=effective_available,
    )
    channel_index = {
        channel: index for index, channel in enumerate(channel_names)
    }
    pair_frame["source_index"] = pair_frame["source"].map(channel_index)
    pair_frame["target_index"] = pair_frame["target"].map(channel_index)
    if pair_frame[["source_index", "target_index"]].isna().any().any():
        raise RuntimeError("Pair enumeration produced an unknown channel")
    n_eligible_pairs_before_limit = len(pair_frame)
    if pair_limit is not None:
        if pair_limit < 1:
            raise ValueError("pair_limit must be positive")
        pair_frame = pair_frame.iloc[:pair_limit].reset_index(drop=True)

    parsed = parse_filename_entities(zscore_path)
    entities = {
        "dataset": str(row.get("dataset", "")),
        "subject": str(row.get("subject", parsed.get("sub", ""))),
        "task": str(row.get("task", parsed.get("task", ""))),
        "phase": phase,
        "description": str(row.get("description", parsed.get("desc", ""))),
        "recording": str(
            row.get(
                "recording",
                parsed.get("recording", parsed.get("rec", "")),
            )
        ),
        "run": str(row.get("run", parsed.get("run", ""))),
        "acquisition": str(row.get("acquisition", parsed.get("acq", ""))),
    }
    return AnalysisData(
        entities=entities,
        zscore_path=zscore_path,
        raw_path=raw_path,
        hammers_path=hammers_path,
        effective_path=effective_path if effective_available else None,
        channel_names=channel_names,
        hga_data=hga,
        hga_times=zscore_epochs.times[hga_mask].copy(),
        hga_sfreq=float(zscore_epochs.info["sfreq"]),
        raw_data=raw,
        raw_times=raw_epochs.times.copy(),
        raw_sfreq=float(raw_epochs.info["sfreq"]),
        events=zscore_epochs.events[valid_trials].copy(),
        trial_indices=trial_indices,
        pair_frame=pair_frame,
        seed_frame=seed_frame,
        parcellation=parcellation,
        n_original_trials=len(valid_trials),
        dropped_trials=dropped_trials,
        dropped_channels=dropped_channels,
        n_eligible_pairs_before_limit=n_eligible_pairs_before_limit,
    )
