#!/usr/bin/env python3
"""Test whether time-resolved lexicality decoding survives item-grouped CV.

The pooled decoding datasets align trials across subjects by stimulus identity:
one row is ``{word}_{occurrence}``, so every subject's first ``banic`` becomes a
single pseudo-trial. Each word is presented twice, and lexicality is a
deterministic function of the word, so a random stratified split lets the same
word land in both train and test. A classifier that merely memorises item
acoustics then scores above chance.

This script re-runs the production time-resolved pipeline unchanged except for
the CV splitter, under three arms:

``production``
    ``MinimumNaNSplit`` exactly as ``run_decoding_resolved.py`` uses it.
``nogroup``
    Plain ``StratifiedKFold``. Isolates the effect of ``MinimumNaNSplit``'s
    NaN-aware fold rejection relative to the grouped arm.
``itemgroup``
    ``StratifiedGroupKFold`` grouped on the stimulus word, so no word appears in
    both train and test.

``nogroup`` vs ``itemgroup`` is the controlled contrast: same splitter family,
same seed, grouping as the only difference.
"""

import rootutils

path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import argparse
import json
import logging
import sys
import time as _time

import h5py
import numpy as np
from ieeg.calc.oversample import MinimumNaNSplit
from mne.decoding import Vectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.decoding.decoder import decode_permutation_scores
from src.decoding.pooled_io import load_pooled_roi
from src.decoding.run_decoding_resolved import cluster_correction
from src.paths import RESULTS_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
ARMS = ("production", "nogroup", "itemgroup")


class FixedGroupSplitter:
    """Adapt a group-aware splitter to the ``cv.split(X, y)`` call used downstream.

    ``decoder.decode_permutation_scores`` never forwards ``groups``, so the
    grouping vector is bound here instead of threading a new argument through
    the production code path.
    """

    def __init__(self, splitter, groups):
        self.splitter = splitter
        self.groups = np.asarray(groups)

    def split(self, X, y=None, groups=None):
        return self.splitter.split(X, y, self.groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.splitter.get_n_splits(X, y, self.groups)


def build_cv(arm, n_folds, seed, conditions):
    if arm == "production":
        return MinimumNaNSplit(n_splits=n_folds, n_repeats=1, random_state=seed)
    if arm == "nogroup":
        return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    if arm == "itemgroup":
        inner = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return FixedGroupSplitter(inner, conditions)
    raise ValueError(f"Unknown arm: {arm}")


def twin_in_train_fraction(cv, X, y, conditions):
    """Fraction of test trials whose stimulus word also appears in the train split."""
    fractions = []
    for train_idx, test_idx in cv.split(X, y):
        train_items = set(conditions[train_idx])
        hits = sum(1 for i in test_idx if conditions[i] in train_items)
        fractions.append(hits / len(test_idx))
    return float(np.mean(fractions))


def run_arm(arm, X, y, conditions, *, n_folds, n_perm, n_jobs, variance, window, step, tmin, tmax, fs):
    pipeline = make_pipeline(
        Vectorizer(),
        StandardScaler(),
        PCA(n_components=variance, random_state=RANDOM_SEED),
        LinearSVC(random_state=RANDOM_SEED),
    )

    time_points = np.arange(tmin + window, tmax + step, step)
    window_samples = int(window * fs)

    accuracies = np.zeros((len(time_points), n_folds))
    baseline = np.zeros((len(time_points), n_folds, n_perm))

    cv_probe = build_cv(arm, n_folds, RANDOM_SEED, conditions)
    twin_frac = twin_in_train_fraction(cv_probe, X, y, conditions)
    logger.info("[%s] test trials whose word is also in train: %.3f", arm, twin_frac)

    for t_idx, time_end in enumerate(time_points):
        end_sample = int((time_end - tmin) * fs) + 1
        start_sample = end_sample - window_samples
        if start_sample < 0 or end_sample > X.shape[-1]:
            logger.warning("Window out of bounds at %.3fs, skipping", time_end)
            continue

        X_segment = X[..., start_sample:end_sample].copy()
        cv = build_cv(arm, n_folds, RANDOM_SEED, conditions)
        score, perm_scores, _ = decode_permutation_scores(
            X_segment,
            y,
            cv,
            pipeline,
            n_jobs=n_jobs,
            n_permutations=n_perm,
            scoring="balanced_accuracy",
            random_state=RANDOM_SEED,
        )
        accuracies[t_idx] = score
        baseline[t_idx] = perm_scores
        logger.info(
            "[%s] t=%.3fs acc=%.4f perm=%.4f", arm, time_end, np.mean(score), float(perm_scores.mean())
        )

    mask, p_values = cluster_correction(accuracies.mean(axis=-1), baseline.mean(axis=1).T)
    return {
        "time": time_points,
        "accuracy": accuracies,
        "baseline": baseline,
        "mask": mask,
        "p_values": p_values,
        "twin_in_train_fraction": twin_frac,
    }


def summarise(arm, result, time_points):
    mask = np.asarray(result["mask"]).astype(bool)
    mean_acc = result["accuracy"].mean(axis=-1)
    early = mask & (time_points <= 0.5)
    peak_idx = int(np.argmax(mean_acc))
    return {
        "arm": arm,
        "n_significant": int(mask.sum()),
        "n_significant_early": int(early.sum()),
        "peak_accuracy": float(mean_acc[peak_idx]),
        "peak_time": float(time_points[peak_idx]),
        "mean_accuracy": float(mean_acc.mean()),
        "significant_window": (
            [float(time_points[mask].min()), float(time_points[mask].max())] if mask.any() else None
        ),
        "twin_in_train_fraction": result["twin_in_train_fraction"],
    }


def plot_comparison(results, time_points, out_path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.paths import save_svg

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"production": "#888888", "nogroup": "#1f77b4", "itemgroup": "#d62728"}

    for arm, result in results.items():
        mean_acc = result["accuracy"].mean(axis=-1)
        ax.plot(time_points, mean_acc, label=arm, color=colors.get(arm), lw=1.6)
        mask = np.asarray(result["mask"]).astype(bool)
        if mask.any():
            offset = 0.34 + 0.012 * list(results).index(arm)
            ax.plot(
                time_points[mask],
                np.full(mask.sum(), offset),
                "|",
                color=colors.get(arm),
                markersize=8,
            )

    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.axvline(0.0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return save_svg(fig, out_path, close=True)


def main(
    bids_root,
    ref,
    roi,
    datatype,
    description,
    phase,
    band,
    variance,
    window,
    step,
    n_perm,
    n_folds,
    n_jobs,
    tmin,
    tmax,
):
    fs = 128
    X, y, conditions, labels, channels, _trials, src_path = load_pooled_roi(
        bids_root, ref, roi, datatype, description, phase, band, tmin, tmax
    )
    subjects = sorted({c.split("_")[0] for c in channels})
    logger.info("Source: %s", src_path)
    logger.info(
        "X=%s trials=%d items=%d subjects=%d nan_trials=%.3f",
        X.shape,
        len(y),
        len(set(conditions)),
        len(subjects),
        float(np.isnan(X).any(axis=(1, 2)).mean()),
    )

    out_dir = RESULTS_ROOT / "decoding" / "diagnostics" / "item_leakage"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{roi}_{datatype}_{phase}_{description}"

    results = {}
    summaries = []
    for arm in ARMS:
        t0 = _time.time()
        logger.info("=== arm=%s ===", arm)
        results[arm] = run_arm(
            arm,
            X,
            y,
            conditions,
            n_folds=n_folds,
            n_perm=n_perm,
            n_jobs=n_jobs,
            variance=variance,
            window=window,
            step=step,
            tmin=tmin,
            tmax=tmax,
            fs=fs,
        )
        summaries.append(summarise(arm, results[arm], results[arm]["time"]))
        logger.info("arm=%s done in %.1fs", arm, _time.time() - t0)

    time_points = results[ARMS[0]]["time"]

    with h5py.File(out_dir / f"{stem}.h5", "w") as stream:
        stream.create_dataset("time", data=time_points)
        for arm, result in results.items():
            group = stream.create_group(arm)
            group.create_dataset("accuracy", data=result["accuracy"])
            group.create_dataset("baseline", data=result["baseline"])
            group.create_dataset("mask", data=np.asarray(result["mask"]).astype(np.int8))
            group.create_dataset("p_values", data=result["p_values"])
            group.attrs["twin_in_train_fraction"] = result["twin_in_train_fraction"]
        stream.attrs["roi"] = roi
        stream.attrs["datatype"] = datatype
        stream.attrs["phase"] = phase
        stream.attrs["description"] = description
        stream.attrs["n_folds"] = n_folds
        stream.attrs["n_perm"] = n_perm
        stream.attrs["variance"] = variance
        stream.attrs["window"] = window
        stream.attrs["step"] = step
        stream.attrs["source_file"] = str(src_path)
        stream.attrs["n_items"] = len(set(conditions))
        stream.attrs["n_subjects"] = len(subjects)

    with open(out_dir / f"{stem}_summary.json", "w") as handle:
        json.dump(
            {
                "roi": roi,
                "datatype": datatype,
                "phase": phase,
                "description": description,
                "n_trials": int(len(y)),
                "n_items": len(set(conditions)),
                "n_subjects": len(subjects),
                "arms": summaries,
            },
            handle,
            indent=2,
        )

    svg = plot_comparison(
        results, time_points, out_dir / stem, f"{roi} {datatype} {phase}/{description}"
    )

    logger.info("")
    logger.info("%-12s %8s %8s %8s %10s %10s", "arm", "n_sig", "n_sig<=0.5s", "peak", "peak_t", "twin_frac")
    for row in summaries:
        logger.info(
            "%-12s %8d %8d %8.4f %10.3f %10.3f",
            row["arm"],
            row["n_significant"],
            row["n_significant_early"],
            row["peak_accuracy"],
            row["peak_time"],
            row["twin_in_train_fraction"],
        )
    logger.info("Wrote %s and %s", out_dir / f"{stem}.h5", svg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", type=str, default="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/")
    parser.add_argument("--ref", type=str, default="bipolar", choices=["car", "bipolar"])
    parser.add_argument("--roi", type=str, default="STGl")
    parser.add_argument("--datatype", type=str, default="lexicality")
    parser.add_argument("--description", type=str, default="Repeat", choices=["Repeat", "Decision", "Passive"])
    parser.add_argument("--phase", type=str, default="Stimulus", choices=["Stimulus", "Delay", "Go", "Response"])
    parser.add_argument("--band", type=str, default="highgamma")
    parser.add_argument("--variance", type=float, default=0.9)
    parser.add_argument("--window", type=float, default=0.3)
    parser.add_argument("--step", type=float, default=0.03)
    parser.add_argument("--n_perm", type=int, default=200)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=16)
    parser.add_argument("--tmin", type=float, default=-0.5)
    parser.add_argument("--tmax", type=float, default=1.5)
    args = parser.parse_args()
    main(**vars(args))
