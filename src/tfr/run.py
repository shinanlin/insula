#!/usr/bin/env python3
"""Hammers AIC/PIC bipolar TFR using the AF multitaper + cluster recipe (SHI-87).

Numerical products:
  results/tfr/{task}/sub-{id}/tfr/*_tfr.h5 and *_stats.h5
Figures:
  img/tfr/{task}/sub-{id}/*_chan-*_tfr.svg
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from mne_bids import BIDSPath

from src.connectivity.pairwise.config import DEFAULT_DATASETS
from src.paths import (
    nmf_assignments_path,
    tfr_fig_dir,
    tfr_results_dir,
)
from src.reaction_time.insula_rt_data import (
    NoStrictInsulaError,
    load_strict_insula_parcellation,
)
from src.tfr.plot import plot_channel_phases

logger = logging.getLogger(__name__)

PHASES = ("Stimulus", "Delay", "Go", "Response")
CONDITION = "Repeat"
REFERENCE = "bipolar"
FREQS = np.arange(4, 251, 3)
DECIM = 40
N_PERM = 1000
P_THRESH = 0.05
P_PLOT = 0.05
CLUSTER_N_JOBS = 1


def _normalize_subject(subject: str) -> str:
    text = str(subject)
    return text[4:] if text.startswith("sub-") else text


def _bids_root(task: str) -> Path:
    if task not in DEFAULT_DATASETS:
        raise KeyError(f"unknown task {task!r}; known={sorted(DEFAULT_DATASETS)}")
    return Path(DEFAULT_DATASETS[task])


def _epoch_root(bids_root: Path | str) -> str:
    return str(Path(bids_root) / "derivatives" / f"epoch({REFERENCE})")


def match_epoch_raw(bids_root: Path | str, subject: str, phase: str, condition: str = CONDITION):
    query = BIDSPath(
        root=_epoch_root(bids_root),
        datatype="epoch(raw)",
        subject=_normalize_subject(subject),
        suffix="raw",
        processing=phase,
        description=condition,
        extension=".h5",
        check=False,
    )
    return list(query.match())


def tfr_paths(subject, task, phase, condition=CONDITION, recording=None, output_root=None):
    root = Path(output_root) / task if output_root is not None else tfr_results_dir(task)
    path = BIDSPath(
        root=str(root),
        datatype="tfr",
        subject=_normalize_subject(subject),
        task=task,
        processing=phase,
        description=condition,
        suffix="tfr",
        extension=".h5",
        recording=recording,
        check=False,
    )
    stats = path.copy().update(suffix="stats")
    return {"tfr": Path(path.fpath), "stats": Path(stats.fpath)}


def channel_fig_path(subject, task, channel, condition=CONDITION):
    subject = _normalize_subject(subject)
    return (
        tfr_fig_dir(task)
        / f"sub-{subject}"
        / f"sub-{subject}_task-{task}_desc-{condition}_chan-{channel}_tfr.svg"
    )


def insula_channels(bids_root: Path | str, subject: str) -> list[str]:
    frame = load_strict_insula_parcellation(
        bids_root, subject=subject, ref=REFERENCE, atlas="hammers"
    )
    return frame["channel"].astype(str).tolist()


def select_insula_channels(epoch_names, anatomical):
    names = set(epoch_names)
    return [ch for ch in anatomical if ch in names]


def subset_epochs(epochs, channels):
    selected = select_insula_channels(epochs.ch_names, channels)
    if not selected:
        return None
    if not getattr(epochs, "preload", False):
        epochs.load_data()
    epochs.pick(selected)
    return epochs


def load_picked(path, channels):
    import mne
    from ieeg.navigate import outliers_to_nan

    epochs = mne.read_epochs(path, preload=False, verbose="error")
    epochs = subset_epochs(epochs, channels)
    if epochs is None:
        return None
    outliers_to_nan(epochs, outliers=10)
    return epochs


def positive_jobs(n_jobs):
    n = int(n_jobs)
    return 1 if n < 1 else n


def compute_tfr(epochs, n_jobs):
    from mne.time_frequency import tfr_multitaper

    n_cycles = FREQS / 2.0
    return tfr_multitaper(
        epochs,
        freqs=FREQS,
        n_cycles=n_cycles,
        time_bandwidth=4.0,
        return_itc=False,
        decim=DECIM,
        n_jobs=positive_jobs(n_jobs),
        average=False,
        verbose="error",
    )


def _comp_by_sort(diff, axis=0):
    m = diff.shape[axis] - 1
    sorted_indices = np.argsort(diff, axis=axis, kind="quicksort")
    proportions = np.arange(diff.shape[axis]) / m
    return proportions[np.argsort(sorted_indices, axis=axis, kind="quicksort")]


def cluster_tfr(task_data, base_data, n_perm):
    from ieeg.calc import stats as ieeg_stats
    from ieeg.calc.stats import time_perm_cluster

    ieeg_stats._comp_by_sort = _comp_by_sort
    return time_perm_cluster(
        task_data,
        base_data,
        p_thresh=P_THRESH,
        tails=2,
        ignore_adjacency=1,
        n_perm=n_perm,
        n_jobs=CLUSTER_N_JOBS,
    )


def assignment_map():
    path = nmf_assignments_path()
    if not path.is_file():
        return {}
    frame = pd.read_csv(path)
    if "channel" not in frame.columns or "functional_cluster" not in frame.columns:
        return {}
    return dict(zip(frame.channel.astype(str), frame.functional_cluster.astype(str)))


def available_phases(bids_root: Path | str, subject: str) -> list[str]:
    if not match_epoch_raw(bids_root, subject, "baseline"):
        return []
    return [phase for phase in PHASES if match_epoch_raw(bids_root, subject, phase)]


def job_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task, bids_root in DEFAULT_DATASETS.items():
        root = Path(bids_root)
        query = BIDSPath(
            root=_epoch_root(root),
            datatype="epoch(raw)",
            suffix="raw",
            description=CONDITION,
            extension=".h5",
            check=False,
        )
        subjects = sorted({_normalize_subject(path.subject) for path in query.match()})
        for subject in subjects:
            try:
                channels = insula_channels(root, subject)
            except (FileNotFoundError, NoStrictInsulaError, ValueError):
                continue
            if not channels:
                continue
            phases = available_phases(root, subject)
            if not phases:
                continue
            rows.append(
                {
                    "task": task,
                    "subject": subject,
                    "bids_root": str(root),
                    "n_insula": len(channels),
                    "phases": ",".join(phases),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["task", "subject"]).reset_index(drop=True)


def run_phase(record, phase, n_jobs, n_perm, output_root=None):
    from ieeg.calc.scaling import rescale

    bids_root = record["bids_root"]
    subject = record["subject"]
    task = record["task"]
    task_paths = match_epoch_raw(bids_root, subject, phase)
    base_paths = match_epoch_raw(bids_root, subject, "baseline")
    if not task_paths or not base_paths:
        logger.info("%s %s %s missing_epoch", subject, task, phase)
        return None
    try:
        anatomical = insula_channels(bids_root, subject)
    except (FileNotFoundError, NoStrictInsulaError, ValueError) as exc:
        logger.info("%s %s no_insula (%s)", subject, task, exc)
        return None
    base_by_rec = {getattr(path, "recording", None): path for path in base_paths}
    wrote = []
    for task_path in task_paths:
        recording = getattr(task_path, "recording", None)
        base_path = base_by_rec.get(recording) or (base_paths[0] if len(base_paths) == 1 else None)
        if base_path is None:
            logger.info("%s %s %s recording=%s missing_baseline", subject, task, phase, recording)
            continue
        paths = tfr_paths(subject, task, phase, CONDITION, recording, output_root=output_root)
        if all(paths[key].is_file() for key in ("tfr", "stats")):
            logger.info("%s %s %s skip %s", subject, phase, recording, paths["tfr"])
            wrote.append(paths)
            continue
        task_epochs = load_picked(task_path.fpath, anatomical)
        if task_epochs is None:
            logger.info("%s %s %s no_insula_in_epochs", subject, task, phase)
            continue
        channels = list(task_epochs.ch_names)
        base_epochs = load_picked(base_path.fpath, channels)
        if base_epochs is None:
            logger.info("%s %s %s no_baseline_insula", subject, task, phase)
            continue
        keep = select_insula_channels(base_epochs.ch_names, channels)
        task_epochs.pick(keep)
        base_epochs.pick(keep)
        logger.info(
            "%s %s %s n_ch=%d n_jobs=%d n_perm=%d",
            subject, task, phase, len(keep), n_jobs, n_perm,
        )
        tfr_task = compute_tfr(task_epochs, n_jobs)
        tfr_base = compute_tfr(base_epochs, n_jobs)
        tfr_task.crop(tfr_task.tmin + 0.5, tfr_task.tmax - 0.5)
        tfr_base.crop(tfr_base.tmin + 0.5, tfr_base.tmax - 0.5)
        mask, pvals = cluster_tfr(tfr_task._data, tfr_base._data, n_perm)
        tfr_task = rescale(tfr_task, tfr_base, copy=True, mode="ratio").average(
            lambda x: np.nanmean(x, axis=0), copy=True
        )
        tfr_task._data = np.log10(tfr_task._data) * 20
        paths["tfr"].parent.mkdir(parents=True, exist_ok=True)
        tfr_task.save(paths["tfr"], overwrite=True)
        with h5py.File(paths["stats"], "w") as handle:
            handle.create_dataset("mask", data=np.asarray(mask))
            handle.create_dataset("pvals", data=np.asarray(pvals))
            handle.create_dataset("ch_names", data=np.asarray(tfr_task.ch_names, dtype="S"))
        logger.info("%s %s wrote %s", subject, phase, paths["tfr"])
        wrote.append(paths)
    return wrote


def _decode_names(values):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in values]


def load_phase_product(subject, task, phase, condition=CONDITION, output_root=None):
    import mne

    paths = tfr_paths(subject, task, phase, condition, output_root=output_root)
    if not paths["tfr"].is_file() or not paths["stats"].is_file():
        return None
    tfr = mne.time_frequency.read_tfrs(paths["tfr"], verbose="error")
    if isinstance(tfr, list):
        tfr = tfr[0]
    with h5py.File(paths["stats"], "r") as handle:
        mask = np.asarray(handle["mask"], float)
        pvals = np.asarray(handle["pvals"], float)
        stats_names = _decode_names(handle["ch_names"][:])
    names = list(tfr.ch_names)
    stats_index = {name: i for i, name in enumerate(stats_names)}
    missing = [name for name in names if name not in stats_index]
    if missing:
        raise ValueError(f"{subject} {task} {phase} stats missing {missing}")
    order = [stats_index[name] for name in names]
    if order != list(range(len(names))):
        mask = mask[order]
        pvals = pvals[order]
    return {
        "ch_names": names,
        "times": np.asarray(tfr.times, float),
        "freqs": np.asarray(tfr.freqs, float),
        "data": np.asarray(tfr.data, float),
        "mask": mask,
        "pvals": pvals,
    }


def shared_channels(products):
    phases = list(products)
    names = [set(products[phase]["ch_names"]) for phase in phases]
    keep = names[0].intersection(*names[1:]) if names else set()
    return [ch for ch in products[phases[0]]["ch_names"] if ch in keep]


def channel_panels(products, channel):
    panels = {}
    for phase, product in products.items():
        idx = product["ch_names"].index(channel)
        panels[phase] = {
            "times": product["times"],
            "freqs": product["freqs"],
            "data": product["data"][idx],
            "mask": product["mask"][idx],
            "pvals": product["pvals"][idx],
        }
    return panels


def run_figures():
    assign = assignment_map()
    n_ok = 0
    n_skip = 0
    for record in job_table().itertuples():
        products = {}
        for phase in str(record.phases).split(","):
            product = load_phase_product(record.subject, record.task, phase)
            if product is not None:
                products[phase] = product
        if not products:
            n_skip += 1
            logger.info("%s %s missing_tfr", record.subject, record.task)
            continue
        channels = shared_channels(products)
        phases = tuple(phase for phase in PHASES if phase in products)
        for channel in channels:
            dest = channel_fig_path(record.subject, record.task, channel)
            plot_channel_phases(
                channel_panels(products, channel),
                dest,
                subject=record.subject,
                channel=channel,
                task=record.task,
                condition=CONDITION,
                cluster=assign.get(channel, ""),
                p_threshold=P_PLOT,
                phases=phases,
            )
            n_ok += 1
        logger.info("%s %s n_ch=%d n_phase=%d", record.subject, record.task, len(channels), len(phases))
    logger.info("wrote %d skipped_jobs %d", n_ok, n_skip)
    return n_ok


def _select_jobs(task=None, subject=None) -> pd.DataFrame:
    jobs = job_table()
    if task:
        jobs = jobs[jobs.task.eq(task)]
    if subject:
        jobs = jobs[jobs.subject.eq(_normalize_subject(subject))]
    return jobs.reset_index(drop=True)


def run_job(index, n_perm=N_PERM, n_jobs=1, task=None, subject=None, output_root=None):
    jobs = _select_jobs(task=task, subject=subject)
    if index < 0 or index >= len(jobs):
        raise SystemExit(f"index {index} out of range 0..{len(jobs) - 1}")
    record = jobs.iloc[index].to_dict()
    logger.info("job %s %s %s phases=%s", index, record["task"], record["subject"], record["phases"])
    for phase in str(record["phases"]).split(","):
        run_phase(record, phase, n_jobs=n_jobs, n_perm=n_perm, output_root=output_root)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["jobs", "run", "figures"])
    parser.add_argument("--index", type=int, default=int(os.getenv("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--n-jobs", type=int, default=int(os.getenv("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--task", type=str, default=None, choices=sorted(DEFAULT_DATASETS))
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    args = parser.parse_args(argv)
    args.n_jobs = positive_jobs(args.n_jobs)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    if args.command == "jobs":
        table = _select_jobs(task=args.task, subject=args.subject)
        if table.empty:
            print("n_jobs 0")
            return
        print(table[["task", "subject", "n_insula", "phases"]].to_string(index=True))
        print("n_jobs", len(table), flush=True)
        return
    if args.command == "figures":
        run_figures()
        return
    output_root = Path(args.output_root) if args.output_root else None
    run_job(
        args.index,
        n_perm=args.n_perm,
        n_jobs=args.n_jobs,
        task=args.task,
        subject=args.subject,
        output_root=output_root,
    )


if __name__ == "__main__":
    main()
