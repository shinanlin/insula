#!/usr/bin/env python3
"""2D cross-condition decoding with shrinkage LDA and pooled OOF AUC.

Train on one LexicalDelay condition (Repeat or Decision) and test on the
other, over a train-time × test-time grid. Estimator, features, metric, CV,
and word-level null match ``run_decoding_resolved_lda.py``. Outputs go to
``(cross)(resolved)(lda){datatype}`` so the old PCA+SVC cross maps stay
untouched.

Each fold trains on source-domain train-words and scores target-domain
held-out words. Sliding windows are fit once per train time and scored at
every test time.
"""

import rootutils

path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import argparse
import gc
import json
import logging
import sys
import time as _time

import h5py
import numpy as np
from mne_bids import BIDSPath
from scipy.ndimage import label as nd_label
from scipy.stats import rankdata

from src.decoding.decoder import decode_cross_permutation_auc_pooled
from src.decoding.pooled_io import load_pooled_roi, pair_pooled_conditions
from src.decoding.run_decoding_resolved_lda import (
    CHANCE_AUC,
    CV_SCHEMES,
    RANDOM_SEED,
    build_classifier,
    build_cv,
    build_transformer,
)
from src.paths import decoding_task_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def window_slices(tmin, tmax, window, step, n_times, fs):
    """Half-open sample slices whose right edge is the labelled time point."""
    window_samples = int(window * fs)
    times = []
    slices = []
    for time_end in np.arange(tmin + window, tmax + step, step):
        end_sample = int((time_end - tmin) * fs) + 1
        start_sample = end_sample - window_samples
        if start_sample < 0 or end_sample > n_times:
            logger.warning("Window out of bounds at %.3fs, skipping", time_end)
            continue
        times.append(float(time_end))
        slices.append((start_sample, end_sample))
    if not times:
        raise ValueError("No sliding window fitted inside the requested bounds")
    return np.asarray(times), tuple(slices)


def cluster_correction_2d(scores, baseline, p_thresh=0.05, chance=CHANCE_AUC):
    """Cluster-mass FWER correction on a train×test AUC map.

    Cells with uncorrected p < ``p_thresh`` form 8-connected clusters. Cluster
    mass is the sum of (AUC - chance) inside the cluster. A cluster survives
    if its mass exceeds the max-cluster-mass null built from the permutation
    maps, each thresholded at the same per-cell p.
    """
    scores = np.asarray(scores, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D, got {scores.shape}")
    if baseline.ndim != 3 or baseline.shape[1:] != scores.shape:
        raise ValueError(
            f"baseline must be (n_perm, T, T) matching scores {scores.shape}, "
            f"got {baseline.shape}"
        )
    n_perm = baseline.shape[0]
    filled = np.where(np.isfinite(baseline), baseline, chance)
    p_act = (np.sum(filled >= scores, axis=0) + 1.0) / (n_perm + 1.0)
    structure = np.ones((3, 3), dtype=int)

    def masses(binary, values):
        labeled, n_feat = nd_label(binary, structure=structure)
        out = []
        for index in range(1, n_feat + 1):
            members = labeled == index
            out.append((index, float(np.sum(values[members]))))
        return labeled, out

    obs_binary = p_act < p_thresh
    obs_stat = scores - chance
    obs_labeled, obs_clusters = masses(obs_binary, obs_stat)

    ranks = rankdata(filled, axis=0, method="max")
    p_maps = (n_perm - ranks + 1.0) / n_perm
    max_null = np.zeros(n_perm, dtype=float)
    perm_stat = filled - chance
    for perm_i in range(n_perm):
        _, perm_clusters = masses(p_maps[perm_i] < p_thresh, perm_stat[perm_i])
        if perm_clusters:
            max_null[perm_i] = max(mass for _, mass in perm_clusters)

    mask = np.zeros(scores.shape, dtype=bool)
    for index, mass in obs_clusters:
        p_cluster = (np.sum(max_null >= mass) + 1.0) / (n_perm + 1.0)
        if p_cluster < p_thresh:
            mask[obs_labeled == index] = True
    return mask, p_act


def main(
    bids_root,
    ref,
    subject,
    phase,
    train_on,
    test_on,
    band,
    datatype,
    n_bins,
    window,
    step,
    n_perm,
    n_folds,
    n_jobs,
    cv_scheme="group",
    drop_labels=(),
    tmin=-0.5,
    tmax=1.5,
):
    if cv_scheme not in CV_SCHEMES:
        raise ValueError(
            f"Unknown cv_scheme {cv_scheme!r}, expected {sorted(CV_SCHEMES)}"
        )
    if train_on == test_on:
        raise ValueError("train_on and test_on must differ")
    drop_labels = tuple(drop_labels or ())

    src_pack = load_pooled_roi(
        bids_root, ref, subject, datatype, train_on, phase, band, tmin, tmax
    )
    tgt_pack = load_pooled_roi(
        bids_root, ref, subject, datatype, test_on, phase, band, tmin, tmax
    )
    X_src, y_src, cond_src, lab_src, ch_src, trials_src, src_path = src_pack
    X_tgt, y_tgt, cond_tgt, lab_tgt, ch_tgt, trials_tgt, tgt_path = tgt_pack
    logger.info("Loaded source %s", src_path)
    logger.info("Loaded target %s", tgt_path)

    X_src, y, conditions, labels, channels, keys, X_tgt = pair_pooled_conditions(
        X_src, y_src, cond_src, lab_src, ch_src, trials_src,
        X_tgt, y_tgt, cond_tgt, lab_tgt, ch_tgt, trials_tgt,
    )
    logger.info("Paired %d trials, %d channels", len(y), len(channels))

    if drop_labels:
        keep = ~np.isin(labels, list(drop_labels))
        if not keep.any():
            raise ValueError(f"Dropping {sorted(drop_labels)} removed every trial")
        n_dropped = int((~keep).sum())
        X_src, X_tgt = X_src[keep], X_tgt[keep]
        y, conditions, labels, keys = y[keep], conditions[keep], labels[keep], keys[keep]
        logger.info("Dropped %d trials labelled %s", n_dropped, sorted(drop_labels))
    else:
        n_dropped = 0

    class_counts = {label: int((labels == label).sum()) for label in sorted(set(labels))}
    logger.info(
        "X_src %s, X_tgt %s, y %s, %d channels, %d unique stimuli, classes %s",
        X_src.shape,
        X_tgt.shape,
        y.shape,
        len(channels),
        len(set(conditions)),
        class_counts,
    )
    smallest = min(class_counts.values())
    if smallest < n_folds:
        raise ValueError(
            f"Smallest class has {smallest} trials, fewer than n_folds={n_folds}: "
            f"{class_counts}"
        )

    fs = 128
    transformer = build_transformer(n_bins)
    classifier = build_classifier()
    time_points, slices = window_slices(tmin, tmax, window, step, X_src.shape[-1], fs)
    n_time = len(slices)
    logger.info("Grid %d × %d windows", n_time, n_time)

    auc = np.empty((n_time, n_time), dtype=float)
    baseline = np.empty((n_perm, n_time, n_time), dtype=float)
    p_perm = np.empty((n_time, n_time), dtype=float)
    run_t0 = _time.time()

    for train_i, (src_start, src_end) in enumerate(slices):
        cv = build_cv(cv_scheme, n_folds)
        obs_aucs, perm_aucs, p_values = decode_cross_permutation_auc_pooled(
            X_src[..., src_start:src_end],
            X_tgt,
            y,
            cv,
            classifier,
            transformer=transformer,
            groups=conditions,
            permute_groups=True,
            tgt_slices=slices,
            n_jobs=n_jobs,
            n_permutations=n_perm,
            random_state=RANDOM_SEED,
        )
        auc[train_i] = obs_aucs
        baseline[:, train_i, :] = np.where(np.isfinite(perm_aucs), perm_aucs, CHANCE_AUC)
        p_perm[train_i] = p_values
        logger.info(
            "t_train=%.3fs  diag AUC=%.3f  p=%.4f  (%.1fs elapsed)",
            time_points[train_i],
            obs_aucs[train_i],
            p_values[train_i],
            _time.time() - run_t0,
        )

    mask, p_values = cluster_correction_2d(auc, baseline)

    save_path = BIDSPath(
        root=str(decoding_task_dir(str(src_path.task))),
        datatype="(cross)(resolved)(lda)" + str(datatype),
        subject=subject,
        suffix=band,
        processing=src_path.processing,
        description=f"{train_on}2{test_on}",
        recording=src_path.recording,
        extension=".h5",
        check=False,
    )
    save_path.mkdir(exist_ok=True)
    logger.info("Saving results to: %s", save_path)

    with h5py.File(save_path, "w") as stream:
        stream.create_dataset(name="auc", data=auc)
        stream.create_dataset(name="baseline", data=baseline)
        stream.create_dataset(name="train_time", data=time_points)
        stream.create_dataset(name="test_time", data=time_points)
        stream.create_dataset(name="mask", data=mask)
        stream.create_dataset(name="p_values", data=p_values)
        stream.create_dataset(name="p_perm", data=p_perm)
        stream.create_dataset(
            name="channel", data=np.array(channels, dtype=h5py.string_dtype())
        )
        stream.create_dataset(
            name="trial", data=np.array(keys, dtype=h5py.string_dtype())
        )

        stream.attrs["fs"] = fs
        stream.attrs["tmin"] = tmin
        stream.attrs["tmax"] = tmax
        stream.attrs["window"] = window
        stream.attrs["step"] = step
        stream.attrs["n_folds"] = n_folds
        stream.attrs["n_perm"] = n_perm
        stream.attrs["n_bins"] = n_bins
        stream.attrs["n_channels"] = len(channels)
        stream.attrs["n_items"] = len(set(conditions))
        stream.attrs["n_classes"] = len(class_counts)
        stream.attrs["class_counts"] = json.dumps(class_counts, sort_keys=True)
        stream.attrs["dropped_labels"] = ",".join(sorted(drop_labels))
        stream.attrs["cv_random_state"] = RANDOM_SEED
        stream.attrs["estimator"] = "LDA(solver=lsqr, shrinkage=auto)"
        stream.attrs["scoring"] = (
            "roc_auc_pooled_oof"
            if len(class_counts) == 2
            else "macro_ovr_auc_pooled_oof"
        )
        stream.attrs["cv_scheme"] = CV_SCHEMES[cv_scheme]
        stream.attrs["perm_scheme"] = "group_label_shuffle(condition)"
        stream.attrs["train_on"] = train_on
        stream.attrs["test_on"] = test_on
        stream.attrs["source"] = str(src_path)
        stream.attrs["target"] = str(tgt_path)

    logger.info("Completed in %.2fs", _time.time() - run_t0)
    del X_src, X_tgt, baseline
    gc.collect()
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bids_root",
        type=str,
        default="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/",
    )
    parser.add_argument("--subject", type=str, default="STGl")
    parser.add_argument("--ref", type=str, default="bipolar", choices=["car", "bipolar"])
    parser.add_argument(
        "--phase",
        type=str,
        default="Delay",
        choices=["Stimulus", "Delay", "Go", "Response"],
    )
    parser.add_argument(
        "--train_on", type=str, default="Repeat", choices=["Repeat", "Decision"]
    )
    parser.add_argument(
        "--test_on", type=str, default="Decision", choices=["Repeat", "Decision"]
    )
    parser.add_argument("--band", type=str, default="highgamma")
    parser.add_argument(
        "--datatype",
        type=str,
        default="lexicality",
        choices=["phoneme", "articulator", "structure", "word", "token", "lexicality"],
    )
    parser.add_argument("--n_bins", type=int, default=5)
    parser.add_argument("--window", type=float, default=0.3)
    parser.add_argument("--step", type=float, default=0.03)
    parser.add_argument("--n_perm", type=int, default=200)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument(
        "--cv_scheme", type=str, default="group", choices=sorted(CV_SCHEMES)
    )
    parser.add_argument("--drop_labels", nargs="*", default=[])
    parser.add_argument("--n_jobs", type=int, default=2)
    parser.add_argument("--tmin", type=float, default=-0.5)
    parser.add_argument("--tmax", type=float, default=1.5)
    args = parser.parse_args()
    main(**vars(args))
