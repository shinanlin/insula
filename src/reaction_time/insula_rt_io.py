"""Atomic HDF5 I/O for the insula time-resolved RT ridge analysis."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from src.reaction_time.insula_ridge import ClusterResult
from src.reaction_time.insula_rt_data import PhaseData


DEFAULT_RT_OUTPUT_ROOT = Path(
    "/hpc/group/coganlab/nanlinshi/insula-functional/results/rt"
)


@dataclass(frozen=True)
class PhaseModelResult:
    """All model outputs for one subject/phase."""

    score_r: np.ndarray
    score_r2: np.ndarray
    score_mae: np.ndarray
    perm_score_r: np.ndarray
    oof_prediction: np.ndarray
    fold_id: np.ndarray
    window_start: np.ndarray
    window_end: np.ndarray
    window_center: np.ndarray
    cluster: ClusterResult


def phase_output_path(
    output_root: Path | str,
    *,
    task: str,
    subject: str,
    phase: str,
    description: str,
) -> Path:
    subject = subject[4:] if str(subject).startswith("sub-") else str(subject)
    return (
        Path(output_root)
        / f"task-{task}"
        / f"sub-{subject}"
        / (
            f"sub-{subject}_task-{task}_proc-{phase}_desc-{description}"
            "_rt-ridge.h5"
        )
    )


def _write_strings(group: h5py.Group, name: str, values) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    clean = np.asarray(["" if value is None else str(value) for value in values], dtype=object)
    group.create_dataset(name, data=clean, dtype=dtype, compression="gzip")


def _write_numeric(group: h5py.Group, name: str, values, *, dtype=None) -> None:
    array = np.asarray(values, dtype=dtype)
    kwargs = {"compression": "gzip", "shuffle": True} if array.ndim else {}
    group.create_dataset(name, data=array, **kwargs)


def write_phase_result(
    path: Path | str,
    *,
    data: PhaseData,
    result: PhaseModelResult,
    config: dict[str, object],
    overwrite: bool = False,
) -> Path:
    """Write one phase result via a same-directory temporary file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w") as h5:
            h5.attrs["schema_version"] = "1.0"
            h5.attrs["task"] = data.task
            h5.attrs["subject"] = data.subject
            h5.attrs["phase"] = data.phase
            h5.attrs["description"] = data.description
            h5.attrs["target"] = "log_rt"
            h5.attrs["rt_definition"] = "response_onset_minus_go_onset"
            h5.attrs["outer_cv"] = "shuffled_group_kfold_by_item"
            h5.attrs["permutation"] = "unrestricted_training_rt_shuffle"
            h5.attrs["sfreq"] = data.sfreq
            for key, value in config.items():
                if value is not None:
                    h5.attrs[str(key)] = value

            scores = h5.create_group("scores")
            _write_numeric(scores, "r", result.score_r, dtype=np.float32)
            _write_numeric(scores, "r2", result.score_r2, dtype=np.float32)
            _write_numeric(scores, "mae", result.score_mae, dtype=np.float32)
            _write_numeric(
                scores, "permutation_r", result.perm_score_r, dtype=np.float32
            )
            _write_numeric(
                scores, "oof_prediction", result.oof_prediction, dtype=np.float32
            )

            inference = h5.create_group("inference")
            _write_numeric(
                inference, "point_p", result.cluster.point_p, dtype=np.float32
            )
            _write_numeric(
                inference,
                "cluster_p_fwer",
                result.cluster.cluster_p_fwer,
                dtype=np.float32,
            )
            _write_numeric(
                inference,
                "sig_mask_fwer",
                result.cluster.sig_mask_fwer,
                dtype=np.uint8,
            )

            windows = h5.create_group("windows")
            _write_numeric(windows, "start", result.window_start, dtype=np.float64)
            _write_numeric(windows, "end", result.window_end, dtype=np.float64)
            _write_numeric(windows, "center", result.window_center, dtype=np.float64)

            trials = h5.create_group("trials")
            _write_numeric(trials, "rt_raw", data.trial_meta["rt_raw"], dtype=np.float64)
            _write_numeric(trials, "rt_log", data.trial_meta["rt_log"], dtype=np.float64)
            _write_numeric(trials, "fold_id", result.fold_id, dtype=np.int16)
            _write_numeric(
                trials, "trial_index", data.trial_meta["trial_index"], dtype=np.int32
            )
            _write_numeric(
                trials, "source_row", data.trial_meta["source_row"], dtype=np.int32
            )
            for column in (
                "target_event_sample",
                "go_event_sample",
                "response_event_sample",
                "target_onset",
                "go_onset",
                "response_onset",
            ):
                _write_numeric(trials, column, data.trial_meta[column], dtype=np.float64)
            for column in (
                "trial_uid",
                "item_id",
                "recording",
                "target_event_name",
                "go_event_name",
                "response_event_name",
                "source_file",
            ):
                _write_strings(trials, column, data.trial_meta[column])

            channels = h5.create_group("channels")
            for column in data.channel_meta.columns:
                values = data.channel_meta[column]
                if column == "mix":
                    _write_numeric(channels, column, values, dtype=np.uint8)
                elif np.issubdtype(values.dtype, np.number):
                    _write_numeric(channels, column, values, dtype=np.float64)
                else:
                    _write_strings(channels, column, values)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def decode_strings(values: np.ndarray) -> list[str]:
    """Decode an HDF5 string dataset into regular Python strings."""

    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]
