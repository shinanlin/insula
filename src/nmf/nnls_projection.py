"""Fixed-W NNLS projection of single-trial HGA onto canonical NMF components.

Freeze spatial loadings ``W`` from ``channel_assignments.csv`` and, for each
trial/time, solve ``argmin_{h>=0} ||x(t) - W_subj h||^2``. Output shape is
``(n_trials, k, n_times)`` with ``k=3`` components (sustain / motor / sensory).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import h5py
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath
from scipy.optimize import nnls

from src.connectivity.pairwise.config import DEFAULT_DATASETS
from src.nmf.waveform_analysis import CLUSTER_ORDER, FUNCTION_COLORS
from src.paths import (
    img_dir,
    nmf_assignments_path,
    nmf_nnls_dir,
    save_svg,
)

LOGGER = logging.getLogger(__name__)

PHASES = ("Stimulus", "Delay", "Go", "Response")
DESCRIPTION = "Repeat"
ZSCORE_DATATYPE = "epoch(band)(zscore)"
BAND = "highgamma"
REFERENCE = "bipolar"
COMPONENT_NAMES = tuple(CLUSTER_ORDER)
LOADING_COLS = tuple(f"loading_{name}" for name in COMPONENT_NAMES)


@dataclass(frozen=True)
class GroupBasis:
    """Canonical group spatial basis."""

    channels: tuple[str, ...]
    subjects: tuple[str, ...]
    W: np.ndarray  # (n_channels, k)
    assignments_path: Path
    assignments_sha256: str

    @property
    def k(self) -> int:
        return int(self.W.shape[1])

    def subject_rows(self, subject: str) -> np.ndarray:
        subject = _normalize_subject(subject)
        return np.flatnonzero(np.asarray(self.subjects) == subject)


def _normalize_subject(value: object) -> str:
    text = str(value)
    return text[4:] if text.startswith("sub-") else text


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_group_basis(assignments_path: Path | None = None) -> GroupBasis:
    path = Path(assignments_path or nmf_assignments_path())
    frame = pd.read_csv(path)
    required = {"channel", "subject", *LOADING_COLS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"assignments missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["subject"] = frame["subject"].map(_normalize_subject)
    frame["channel"] = frame["channel"].astype(str)
    W = frame.loc[:, list(LOADING_COLS)].to_numpy(dtype=float)
    if not np.isfinite(W).all() or (W < -1e-12).any():
        raise ValueError("loadings must be finite and non-negative")
    W = np.clip(W, 0.0, None)
    return GroupBasis(
        channels=tuple(frame["channel"].tolist()),
        subjects=tuple(frame["subject"].tolist()),
        W=W,
        assignments_path=path.resolve(),
        assignments_sha256=file_sha256(path),
    )


def save_group_basis(basis: GroupBasis, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        channels=np.asarray(basis.channels, dtype=object),
        subjects=np.asarray(basis.subjects, dtype=object),
        W=basis.W.astype(np.float64),
        component_names=np.asarray(COMPONENT_NAMES, dtype=object),
        assignments_path=str(basis.assignments_path),
        assignments_sha256=basis.assignments_sha256,
    )
    return path


def bids_root_for_task(task: str) -> Path:
    if task not in DEFAULT_DATASETS:
        raise KeyError(f"unknown task {task!r}; known={sorted(DEFAULT_DATASETS)}")
    return Path(DEFAULT_DATASETS[task])


def discover_epoch_paths(
    task: str,
    phase: str,
    *,
    description: str = DESCRIPTION,
    bids_root: Path | None = None,
) -> list[BIDSPath]:
    root = Path(bids_root or bids_root_for_task(task))
    query = BIDSPath(
        root=str(root / "derivatives" / f"epoch({REFERENCE})"),
        task=task,
        datatype=ZSCORE_DATATYPE,
        suffix=BAND,
        extension=".h5",
        description=description,
        processing=phase,
        check=False,
    )
    return sorted(query.match(), key=lambda path: str(path.subject))


def align_subject_channels(
    basis: GroupBasis,
    subject: str,
    epoch_channels: Sequence[str],
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return (basis_row_idx, shared_names, W_subj)."""
    subject = _normalize_subject(subject)
    rows = basis.subject_rows(subject)
    if rows.size == 0:
        return rows, [], np.zeros((0, basis.k), dtype=float)
    assigned = [basis.channels[i] for i in rows]
    epoch_index = {name: i for i, name in enumerate(epoch_channels)}
    keep_rows: list[int] = []
    shared: list[str] = []
    for row, name in zip(rows.tolist(), assigned):
        if name in epoch_index:
            keep_rows.append(row)
            shared.append(name)
    if not keep_rows:
        return np.asarray([], dtype=int), [], np.zeros((0, basis.k), dtype=float)
    row_idx = np.asarray(keep_rows, dtype=int)
    # Order W rows to match shared channel order (epoch pick order follows shared).
    W_subj = basis.W[row_idx]
    return row_idx, shared, W_subj


def project_nnls(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Project trials onto fixed W with NNLS.

    Parameters
    ----------
    X : (n_trials, n_channels, n_times)
    W : (n_channels, k)

    Returns
    -------
    H : (n_trials, k, n_times) float64
    """
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)
    if X.ndim != 3:
        raise ValueError(f"X must be (n_trials, n_channels, n_times), got {X.shape}")
    if W.ndim != 2 or W.shape[0] != X.shape[1]:
        raise ValueError(f"W must be (n_channels={X.shape[1]}, k), got {W.shape}")
    n_trials, n_ch, n_times = X.shape
    k = W.shape[1]
    X = np.clip(X, 0.0, None)
    H = np.full((n_trials, k, n_times), np.nan, dtype=float)

    finite = np.isfinite(X)
    valid = finite.all(axis=2)  # (n_trials, n_ch) present at every t
    any_finite = finite.any(axis=2)
    time_varying = any_finite & ~valid

    for i in range(n_trials):
        keep = valid[i] & ~time_varying[i]
        if keep.sum() < k:
            continue
        W_use = W[keep]
        X_use = X[i, keep, :]
        for t in range(n_times):
            coef, _ = nnls(W_use, X_use[:, t])
            H[i, :, t] = coef
    return H


def project_subject_epochs(
    basis: GroupBasis,
    subject: str,
    epochs: mne.BaseEpochs,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, object]]:
    """Return H, times, shared channels, and info for one subject's epochs."""
    row_idx, shared, W_subj = align_subject_channels(
        basis, subject, epochs.ch_names
    )
    info: dict[str, object] = {
        "subject": _normalize_subject(subject),
        "n_assigned": int(basis.subject_rows(subject).size),
        "n_shared": len(shared),
        "shared_channels": shared,
    }
    if not shared:
        times = np.asarray(epochs.times, dtype=float)
        return (
            np.zeros((0, basis.k, times.size), dtype=float),
            times,
            shared,
            info,
        )
    picks = [epochs.ch_names.index(name) for name in shared]
    data = epochs.get_data(copy=True)[:, picks, :]
    H = project_nnls(data, W_subj)
    info["n_trials"] = int(H.shape[0])
    info["n_projected"] = int(np.isfinite(H[:, 0, 0]).sum())
    return H, np.asarray(epochs.times, dtype=float), shared, info


def trace_h5_name(task: str, phase: str, description: str = DESCRIPTION) -> str:
    return f"task-{task}_proc-{phase}_desc-{description}_nnls.h5"


def write_trace_h5(
    path: Path,
    *,
    H: np.ndarray,
    times: np.ndarray,
    trials: pd.DataFrame,
    attrs: Mapping[str, object],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    vlen = h5py.string_dtype(encoding="utf-8")
    with h5py.File(temporary, "w") as handle:
        handle.create_dataset(
            "H",
            data=np.asarray(H, dtype=np.float32),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        handle.create_dataset("times", data=np.asarray(times, dtype=float))
        handle.create_dataset(
            "component_names",
            data=np.asarray(COMPONENT_NAMES, dtype=object),
            dtype=vlen,
        )
        group = handle.create_group("trials")
        for column in trials.columns:
            values = trials[column].to_numpy()
            if values.dtype == object or values.dtype.kind in "USO":
                group.create_dataset(
                    column,
                    data=np.asarray([str(v) for v in values], dtype=object),
                    dtype=vlen,
                )
            elif values.dtype == bool:
                group.create_dataset(column, data=values)
            else:
                group.create_dataset(column, data=values)
        for key, value in attrs.items():
            if isinstance(value, (dict, list, tuple, set)):
                handle.attrs[key] = json.dumps(value, sort_keys=True)
            else:
                handle.attrs[key] = value
    temporary.replace(path)
    return path


def read_trace_h5(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        trials = pd.DataFrame(
            {
                key: (
                    [
                        x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)
                        for x in dataset[:]
                    ]
                    if dataset.dtype.kind in "OSU"
                    else dataset[:]
                )
                for key, dataset in handle["trials"].items()
            }
        )
        attrs = {
            key: (value.decode() if isinstance(value, bytes) else value)
            for key, value in handle.attrs.items()
        }
        names = [
            x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)
            for x in handle["component_names"][:]
        ]
        return {
            "H": handle["H"][:],
            "times": handle["times"][:],
            "component_names": names,
            "trials": trials,
            "attrs": attrs,
        }


def _mean_sem(series: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return mean, SEM, and n finite trials for (n_trials, n_times)."""
    finite = np.isfinite(series).all(axis=1)
    if not finite.any():
        n_times = series.shape[1]
        nan = np.full(n_times, np.nan)
        return nan, nan, 0
    values = series[finite]
    mean = values.mean(axis=0)
    if values.shape[0] > 1:
        sem = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    else:
        sem = np.zeros_like(mean)
    return mean, sem, int(values.shape[0])


def baseline_correct_nnls(
    H: np.ndarray,
    times: np.ndarray,
    *,
    tmin: float | None = None,
    tmax: float = 0.0,
) -> np.ndarray:
    """Subtract per-trial, per-component mean over a pre-event window.

    Default window is ``t < 0``. If no samples fall in that window (e.g. delay
    epochs starting at 0), fall back to the first 50 ms of the epoch.
    Trials/components with no finite baseline samples are left unchanged.
    """
    H = np.asarray(H, dtype=float).copy()
    times = np.asarray(times, dtype=float)
    if H.ndim != 3:
        raise ValueError(f"H must be (n_trials, k, n_times), got {H.shape}")
    if tmin is None:
        mask = times < tmax
    else:
        mask = (times >= tmin) & (times < tmax)
    if not mask.any():
        mask = times <= (float(times.min()) + 0.05)
    if not mask.any():
        return H
    window = H[:, :, mask]
    counts = np.isfinite(window).sum(axis=2)
    sums = np.nansum(window, axis=2)
    baseline = np.full(counts.shape, np.nan, dtype=float)
    good = counts > 0
    baseline[good] = sums[good] / counts[good]
    H[good] = H[good] - baseline[good, None]
    return H


def plot_nnls_overview(
    phase_traces: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    task: str,
    description: str = DESCRIPTION,
    phases: Sequence[str] = PHASES,
    path: Path,
) -> Path:
    """One panel per phase (H_overview style); components overlaid as mean±SEM.

    Traces are pre-event baseline-corrected (``t < 0``) before averaging so
    components share a common zero baseline for visual comparison.
    """
    phase_list = [phase for phase in phases if phase in phase_traces]
    if not phase_list:
        raise ValueError("phase_traces is empty")

    fig, axes = plt.subplots(
        1,
        len(phase_list),
        figsize=(3.0 * len(phase_list), 3.0),
        sharey=True,
        squeeze=False,
        dpi=150,
    )
    n_for_legend: dict[str, int] = {}
    for col, phase in enumerate(phase_list):
        axis = axes[0, col]
        H, times = phase_traces[phase]
        times = np.asarray(times, dtype=float)
        H = baseline_correct_nnls(np.asarray(H, dtype=float), times)
        for index, name in enumerate(COMPONENT_NAMES):
            mean, sem, n_trials = _mean_sem(H[:, index, :])
            n_for_legend[name] = n_trials
            if n_trials == 0:
                continue
            color = FUNCTION_COLORS.get(name, "0.35")
            axis.plot(times, mean, color=color, linewidth=1.5, label=name)
            axis.fill_between(
                times, mean - sem, mean + sem, color=color, alpha=0.22, linewidth=0
            )
        axis.axvline(0.0, color="0.25", linestyle="--", linewidth=0.6)
        axis.axhline(0.0, color="0.6", linewidth=0.4)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_title(phase, fontsize=8)
        axis.set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("NNLS (bl-corr.)")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    if handles:
        axes[0, -1].legend(
            handles,
            [f"{lab} (n={n_for_legend.get(lab, 0)})" for lab in labels],
            frameon=False,
            fontsize=7,
            loc="best",
        )
    fig.suptitle(
        f"NNLS projection by phase ({task}, {description}; pre-event baseline)",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()
    return save_svg(fig, path, close=True)


def plot_nnls_overview_from_traces(
    task: str,
    *,
    phases: Sequence[str] = PHASES,
    description: str = DESCRIPTION,
    output_dir: Path | None = None,
    img_out: Path | None = None,
) -> Path:
    """Rebuild the combined overview SVG from existing per-phase H5 caches."""
    output_dir = Path(output_dir or nmf_nnls_dir())
    img_out = Path(img_out or img_dir("nmf"))
    phase_traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for phase in phases:
        h5_path = output_dir / "traces" / trace_h5_name(task, phase, description)
        if not h5_path.is_file():
            raise FileNotFoundError(h5_path)
        payload = read_trace_h5(h5_path)
        phase_traces[phase] = (payload["H"], payload["times"])
    svg_path = img_out / f"nnls_H_overview_{task}.svg"
    return plot_nnls_overview(
        phase_traces,
        task=task,
        description=description,
        phases=phases,
        path=svg_path,
    )


def run_task_phase(
    basis: GroupBasis,
    task: str,
    phase: str,
    *,
    description: str = DESCRIPTION,
    output_dir: Path | None = None,
) -> dict[str, object]:
    """Project all subjects for one task×phase and write H5."""
    output_dir = Path(output_dir or nmf_nnls_dir())
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    paths = discover_epoch_paths(task, phase, description=description)
    H_blocks: list[np.ndarray] = []
    trial_rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    used: list[dict[str, object]] = []
    times: np.ndarray | None = None
    fs: float | None = None

    for bids_path in paths:
        subject = _normalize_subject(bids_path.subject)
        epoch_path = Path(bids_path.fpath)
        if not epoch_path.is_file():
            skipped.append(
                {"subject": subject, "path": str(epoch_path), "reason": "missing_file"}
            )
            continue
        try:
            epochs = mne.read_epochs(epoch_path, preload=True, verbose="error")
        except Exception as exc:  # noqa: BLE001 — keep job going across subjects
            skipped.append(
                {
                    "subject": subject,
                    "path": str(epoch_path),
                    "reason": f"read_failed: {exc}",
                }
            )
            continue
        H, epoch_times, shared, info = project_subject_epochs(basis, subject, epochs)
        if times is None:
            times = epoch_times
            fs = float(epochs.info["sfreq"])
        elif not np.allclose(times, epoch_times):
            skipped.append(
                {
                    "subject": subject,
                    "path": str(epoch_path),
                    "reason": "time_axis_mismatch",
                }
            )
            del epochs
            continue
        if info["n_shared"] == 0 or H.shape[0] == 0:
            skipped.append(
                {
                    "subject": subject,
                    "path": str(epoch_path),
                    "reason": "no_shared_channels",
                    "n_assigned": info["n_assigned"],
                }
            )
            del epochs
            continue
        H_blocks.append(H)
        for trial_index in range(H.shape[0]):
            trial_rows.append(
                {
                    "subject": subject,
                    "trial_index": int(trial_index),
                    "epoch_file": str(epoch_path),
                    "n_shared_channels": int(info["n_shared"]),
                }
            )
        used.append(
            {
                "subject": subject,
                "path": str(epoch_path),
                "n_trials": int(H.shape[0]),
                "n_shared": int(info["n_shared"]),
            }
        )
        del epochs

    if not H_blocks or times is None:
        raise RuntimeError(f"no projected trials for {task} {phase} {description}")

    H_all = np.concatenate(H_blocks, axis=0)
    trials = pd.DataFrame(trial_rows)
    h5_path = traces_dir / trace_h5_name(task, phase, description)
    attrs = {
        "method": "nnls",
        "task": task,
        "phase": phase,
        "description": description,
        "assignments_path": str(basis.assignments_path),
        "assignments_sha256": basis.assignments_sha256,
        "fs": float(fs or np.nan),
        "k": int(basis.k),
        "n_trials": int(H_all.shape[0]),
        "n_subjects_used": len(used),
        "n_subjects_skipped": len(skipped),
    }
    write_trace_h5(h5_path, H=H_all, times=times, trials=trials, attrs=attrs)

    return {
        "h5": str(h5_path),
        "H": H_all,
        "times": times,
        "n_trials": int(H_all.shape[0]),
        "used": used,
        "skipped": skipped,
        "attrs": attrs,
    }


def run_nnls_projection(
    *,
    task: str = "LexicalDelay",
    phases: Iterable[str] = PHASES,
    description: str = DESCRIPTION,
    assignments_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, object]:
    """Run NNLS projection for one task across phases."""
    output_dir = Path(output_dir or nmf_nnls_dir())
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "traces").mkdir(parents=True, exist_ok=True)
    images = img_dir("nmf")
    phase_list = list(phases)

    basis = load_group_basis(assignments_path)
    save_group_basis(basis, output_dir / "W_group.npz")

    phase_results = {}
    phase_traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for phase in phase_list:
        LOGGER.info("NNLS projection %s %s %s", task, phase, description)
        detail = run_task_phase(
            basis,
            task,
            phase,
            description=description,
            output_dir=output_dir,
        )
        phase_results[phase] = detail
        phase_traces[phase] = (detail["H"], detail["times"])

    svg_path = images / f"nnls_H_overview_{task}.svg"
    plot_nnls_overview(
        phase_traces,
        task=task,
        description=description,
        phases=phase_list,
        path=svg_path,
    )

    manifest = {
        "task": task,
        "description": description,
        "phases": phase_list,
        "method": "nnls",
        "component_names": list(COMPONENT_NAMES),
        "assignments_path": str(basis.assignments_path),
        "assignments_sha256": basis.assignments_sha256,
        "output_dir": str(output_dir.resolve()),
        "W_group": str((output_dir / "W_group.npz").resolve()),
        "svg": str(svg_path.resolve()),
        "phases_detail": {
            phase: {
                "h5": detail["h5"],
                "n_trials": detail["n_trials"],
                "n_subjects_used": detail["attrs"]["n_subjects_used"],
                "n_subjects_skipped": detail["attrs"]["n_subjects_skipped"],
                "skipped": detail["skipped"],
            }
            for phase, detail in phase_results.items()
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
