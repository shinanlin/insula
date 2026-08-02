#!/usr/bin/env python3
"""Whole-window Haufe patterns for ROI pseudo-subject decoding datasets."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time as _time

import h5py
import numpy as np
from mne_bids import BIDSPath
from sklearn.model_selection import StratifiedKFold

from src.decoding.patterns import (
    cv_mean_pattern,
    make_decoding_pipeline,
    pattern_ct_cluster_correction,
    wholewindow_pattern_null,
)
from src.paths import decoding_task_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
BINARY_DATATYPES = {"lexicality"}
MULTICLASS_DATATYPES = {"phoneme", "articulator"}


def pattern_datatype(datatype: str) -> str:
    return f"(decode)(pattern){datatype}"


def load_roi_epoch(
    bids_root: str,
    ref: str,
    subject: str,
    description: str,
    phase: str,
    band: str,
    datatype: str,
    bids_task: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, BIDSPath, dict[str, int]]:
    """Load one whole-window ROI decoding H5 (no phase-window crop)."""
    recording = "1" if bids_task == "PhonemeSequence" else None
    root = BIDSPath(
        root=os.path.join(bids_root, "derivatives", f"decoding({ref})"),
        datatype=datatype,
        description=description,
        suffix=band,
        processing=phase,
        recording=recording,
        extension=".h5",
        check=False,
    )
    roi_path = root.copy().update(subject=subject)
    roi_files = roi_path.match()
    if not roi_files:
        raise FileNotFoundError(f"No files found for ROI {subject} {phase} {datatype}")

    source_path = roi_files[0]
    with h5py.File(source_path, "r") as data:
        X = data["X"][()]
        y = data["y"][()]
        times = data["time"][()]
        channels = data["channel"][()]
        raw_event_id = data.attrs.get("event_id")
        if raw_event_id is None:
            raise KeyError(f"No event_id attribute in {source_path}")
        event_id = json.loads(raw_event_id)

    channels = np.asarray(
        [c.decode() if isinstance(c, bytes) else str(c) for c in channels]
    )
    return X, y, times, channels, source_path, event_id


def eligible_ovr_classes(
    y: np.ndarray,
    event_id: dict[str, int],
    min_trials_per_class: int,
) -> tuple[list[tuple[str, int, int]], dict[str, int]]:
    eligible: list[tuple[str, int, int]] = []
    excluded: dict[str, int] = {}
    for name, class_id in sorted(event_id.items(), key=lambda item: item[1]):
        count = int((y == class_id).sum())
        if count >= min_trials_per_class:
            eligible.append((name, int(class_id), count))
        else:
            excluded[name] = count
    return eligible, excluded


def fit_binary_contrast(
    X: np.ndarray,
    y_binary: np.ndarray,
    *,
    variance: float,
    n_perm: int,
    n_folds: int,
    n_jobs: int,
    cluster_forming_p: float,
    cluster_alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if min(np.bincount(y_binary.astype(int), minlength=2)) < n_folds:
        raise ValueError(
            f"Need >= {n_folds} trials per binary class, got "
            f"{np.bincount(y_binary.astype(int), minlength=2).tolist()}"
        )
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    pipeline = make_decoding_pipeline(variance=variance, random_state=RANDOM_SEED)
    pattern = cv_mean_pattern(X, y_binary, cv, pipeline, random_state=RANDOM_SEED)
    perm_pattern = wholewindow_pattern_null(
        X,
        y_binary,
        cv,
        pipeline,
        n_permutations=n_perm,
        n_jobs=n_jobs,
        random_state=RANDOM_SEED,
    )
    mask, p_values = pattern_ct_cluster_correction(
        pattern,
        perm_pattern,
        cluster_forming_p=cluster_forming_p,
        cluster_alpha=cluster_alpha,
        tails=2,
    )
    return pattern, mask, p_values


def save_result(
    save_path,
    *,
    pattern: np.ndarray,
    pattern_mask: np.ndarray,
    pattern_p_values: np.ndarray,
    times: np.ndarray,
    channels: np.ndarray,
    attrs: dict,
    class_names: list[str] | None = None,
    class_ids: list[int] | None = None,
    class_counts: list[int] | None = None,
) -> None:
    with h5py.File(save_path, "w") as handle:
        handle.create_dataset("pattern", data=pattern)
        handle.create_dataset("pattern_mask", data=pattern_mask)
        handle.create_dataset("pattern_p_values", data=pattern_p_values)
        handle.create_dataset("times", data=times)
        handle.create_dataset("channel", data=np.asarray(channels, dtype="S"))
        if class_names is not None:
            handle.create_dataset(
                "class_names", data=np.asarray(class_names, dtype="S")
            )
            handle.create_dataset(
                "class_ids", data=np.asarray(class_ids, dtype=np.int64)
            )
            handle.create_dataset(
                "class_counts", data=np.asarray(class_counts, dtype=np.int64)
            )
        for key, value in attrs.items():
            handle.attrs[key] = (
                json.dumps(value) if isinstance(value, (dict, list)) else value
            )


def main(
    bids_root: str,
    bids_task: str,
    subject: str,
    ref: str,
    description: str,
    phase: str,
    band: str,
    datatype: str,
    variance: float,
    n_perm: int,
    n_folds: int,
    n_jobs: int,
    cluster_forming_p: float,
    cluster_alpha: float,
    min_trials_per_class: int,
) -> None:
    X, y, times, channels, source_path, event_id = load_roi_epoch(
        bids_root, ref, subject, description, phase, band, datatype, bids_task
    )
    logger.info(
        "Loaded sub-%s %s %s %s: X=%s",
        subject,
        phase,
        description,
        datatype,
        X.shape,
    )
    file_t0 = _time.time()

    if datatype in BINARY_DATATYPES:
        pattern, pattern_mask, pattern_p_values = fit_binary_contrast(
            X,
            y.astype(np.int8),
            variance=variance,
            n_perm=n_perm,
            n_folds=n_folds,
            n_jobs=n_jobs,
            cluster_forming_p=cluster_forming_p,
            cluster_alpha=cluster_alpha,
        )
        class_names = class_ids = class_counts = None
        excluded = {}
        class_strategy = "binary"
    elif datatype in MULTICLASS_DATATYPES:
        eligible, excluded = eligible_ovr_classes(y, event_id, min_trials_per_class)
        if not eligible:
            raise ValueError(
                f"No {datatype} classes meet min_trials_per_class={min_trials_per_class}: "
                f"{excluded}"
            )
        patterns, masks, p_values = [], [], []
        class_names, class_ids, class_counts = [], [], []
        for name, class_id, count in eligible:
            logger.info("OvR %s: n+=%d n-=%d", name, count, len(y) - count)
            y_binary = (y == class_id).astype(np.int8)
            pattern, mask, p_value = fit_binary_contrast(
                X,
                y_binary,
                variance=variance,
                n_perm=n_perm,
                n_folds=n_folds,
                n_jobs=n_jobs,
                cluster_forming_p=cluster_forming_p,
                cluster_alpha=cluster_alpha,
            )
            patterns.append(pattern)
            masks.append(mask)
            p_values.append(p_value)
            class_names.append(name)
            class_ids.append(class_id)
            class_counts.append(count)
        pattern = np.stack(patterns)
        pattern_mask = np.stack(masks)
        pattern_p_values = np.stack(p_values)
        class_strategy = "one_vs_rest"
    else:
        raise ValueError(f"Unsupported datatype: {datatype}")

    save_path = BIDSPath(
        root=str(decoding_task_dir(str(source_path.task))),
        datatype=pattern_datatype(datatype),
        subject=subject,
        task=source_path.task,
        suffix=band,
        processing=phase,
        recording=source_path.recording,
        description=description,
        extension=".h5",
        check=False,
    )
    save_path.mkdir(exist_ok=True)
    attrs = {
        "source_path": str(source_path),
        "fs": 128,
        "tmin": float(times[0]),
        "tmax": float(times[-1]),
        "datatype": datatype,
        "method": "haufe2014",
        "pointwise": False,
        "pattern_stat": "signed",
        "pattern_tails": 2,
        "pattern_stat_method": "wholewindow_global_perm",
        "n_perm": n_perm,
        "n_folds": n_folds,
        "variance": variance,
        "cluster_forming_p": cluster_forming_p,
        "cluster_alpha": cluster_alpha,
        "min_trials_per_class": min_trials_per_class,
        "class_strategy": class_strategy,
        "excluded_class_counts": excluded,
        "phase": phase,
        "description": description,
        "output_root": str(Path(save_path.root).resolve()),
    }
    save_result(
        save_path,
        pattern=pattern,
        pattern_mask=pattern_mask,
        pattern_p_values=pattern_p_values,
        times=times,
        channels=channels,
        attrs=attrs,
        class_names=class_names,
        class_ids=class_ids,
        class_counts=class_counts,
    )
    logger.info("Saved %s in %.2fs", save_path, _time.time() - file_t0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bids_root",
        type=str,
        default="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/",
    )
    parser.add_argument(
        "--bids_task",
        type=str,
        default="LexicalDelay",
        choices=["LexicalDelay", "PhonemeSequence"],
    )
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--ref", type=str, default="bipolar", choices=["car", "bipolar"])
    parser.add_argument(
        "--description",
        type=str,
        default="Repeat",
        choices=["Repeat", "Passive", "Decision"],
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="Stimulus",
        choices=["Stimulus", "Delay", "Go", "Response"],
    )
    parser.add_argument("--band", type=str, default="highgamma")
    parser.add_argument(
        "--datatype",
        type=str,
        default="lexicality",
        choices=["phoneme", "articulator", "lexicality"],
    )
    parser.add_argument("--variance", type=float, default=0.85)
    parser.add_argument("--n_perm", type=int, default=300)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=32)
    parser.add_argument("--cluster_forming_p", type=float, default=0.10)
    parser.add_argument("--cluster_alpha", type=float, default=0.05)
    parser.add_argument("--min_trials_per_class", type=int, default=10)
    args = parser.parse_args()
    main(**vars(args))
