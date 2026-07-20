#!/usr/bin/env python3
"""Matched repeated-CV permutation test for early AICl Delay decoding."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

LAUNCH_CWD = Path.cwd()
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/ns458-numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ns458-matplotlib")

import h5py
import numpy as np
from ieeg.calc.oversample import MinimumNaNSplit, mixup
from joblib import Parallel, delayed
from mne.decoding import Vectorizer
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


PROJECT = Path("/hpc/group/coganlab/nanlinshi/insula-aicl-delay")
sys.path.insert(0, str(PROJECT))
from src.decoding.run_decoding import load_roi_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-timepoints", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def prepare_fold(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the existing fold preprocessing with deterministic NaN filling."""
    X_train = X[train_idx].copy()
    X_test = X[test_idx].copy()
    y_train = y[train_idx]

    for cls in np.unique(y_train):
        cls_mask = y_train == cls
        X_cls = X_train[cls_mask]
        nan_mask = np.isnan(X_cls)
        if nan_mask.any():
            X_cls[nan_mask] = 0.0
        mixup(X_cls, obs_axis=0, rng=fold_seed)
        X_train[cls_mask] = X_cls

    rng = np.random.RandomState(fold_seed)
    train_nan = np.isnan(X_train)
    if train_nan.any():
        X_train[train_nan] = rng.normal(0.0, 1.0, int(train_nan.sum()))
    test_nan = np.isnan(X_test)
    if test_nan.any():
        X_test[test_nan] = rng.normal(0.0, 1.0, int(test_nan.sum()))
    return X_train, X_test


def score_labels(
    estimator,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> float:
    model = clone(estimator)
    model.fit(X_train, y_train)
    return balanced_accuracy_score(y_test, model.predict(X_test))


def score_split(
    estimator,
    X: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold_seed: int,
) -> float:
    """Run all label-dependent fold preprocessing inside the permutation."""
    X_train, X_test = prepare_fold(X, labels, train_idx, test_idx, fold_seed)
    return score_labels(
        estimator,
        X_train,
        X_test,
        labels[train_idx],
        labels[test_idx],
    )


def main() -> None:
    args = parse_args()
    if not args.output.is_absolute():
        args.output = LAUNCH_CWD / args.output
    args.output.parent.mkdir(parents=True, exist_ok=True)

    Xs, ys, _ = load_roi_data(
        "/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/",
        "bipolar",
        "AICl",
        "Repeat",
        "Delay",
        "highgamma",
        "articulator",
        -0.5,
        1.5,
    )
    X, y = Xs[0], ys[0]

    fs = 128
    window = 0.3
    step = 0.03
    tmin, tmax = -0.5, 1.5
    all_times = np.arange(tmin + window, tmax + step, step)
    selected = (all_times >= 0.0) & (all_times <= 0.5)
    times = all_times[selected]
    if args.max_timepoints is not None:
        times = times[: args.max_timepoints]
    window_samples = int(window * fs)

    expected_splits = 5 * args.n_repeats

    def make_splits(labels: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        splitter = MinimumNaNSplit(
            n_splits=5,
            n_repeats=args.n_repeats,
            random_state=args.seed,
        )
        result = list(splitter.split(X, labels))
        if len(result) != expected_splits:
            raise RuntimeError(
                f"Expected {expected_splits} splits, got {len(result)}"
            )
        return result

    observed_splits = make_splits(y)

    permutation_rng = np.random.RandomState(args.seed)
    permuted_labels = np.stack(
        [permutation_rng.permutation(y) for _ in range(args.n_permutations)]
    )
    permutation_splits = [make_splits(labels) for labels in permuted_labels]

    estimator = make_pipeline(
        Vectorizer(),
        StandardScaler(),
        PCA(n_components=0.95, random_state=args.seed),
        LinearSVC(C=1.0, random_state=args.seed, max_iter=10_000),
    )

    accuracy = np.empty((len(times), expected_splits), dtype=float)
    baseline = np.empty(
        (len(times), expected_splits, args.n_permutations), dtype=float
    )
    started = time.time()

    for time_idx, time_end in enumerate(times):
        end_sample = int((time_end - tmin) * fs) + 1
        start_sample = end_sample - window_samples
        X_window = X[..., start_sample:end_sample]

        for split_idx, (train_idx, test_idx) in enumerate(observed_splits):
            fold_seed = args.seed + 10_000 * time_idx + split_idx
            accuracy[time_idx, split_idx] = score_split(
                estimator,
                X_window,
                y,
                train_idx,
                test_idx,
                fold_seed,
            )

        permuted_scores = Parallel(n_jobs=args.n_jobs, prefer="threads")(
            delayed(score_split)(
                estimator,
                X_window,
                permuted_labels[perm_idx],
                train_idx,
                test_idx,
                args.seed + 10_000 * time_idx + split_idx,
            )
            for perm_idx, splits in enumerate(permutation_splits)
            for split_idx, (train_idx, test_idx) in enumerate(splits)
        )
        baseline[time_idx] = np.asarray(permuted_scores).reshape(
            args.n_permutations, expected_splits
        ).T

        elapsed = time.time() - started
        print(
            f"time {time_idx + 1}/{len(times)} ({time_end:.2f}s) "
            f"elapsed={elapsed / 60:.1f} min",
            flush=True,
        )

    observed_stat = float(accuracy.mean())
    null_stats = baseline.mean(axis=(0, 1))
    p_value = float(
        (1 + np.count_nonzero(null_stats >= observed_stat))
        / (args.n_permutations + 1)
    )

    with h5py.File(args.output, "w") as handle:
        handle.create_dataset("accuracy", data=accuracy.reshape(len(times), args.n_repeats, 5))
        handle.create_dataset(
            "baseline",
            data=baseline.reshape(len(times), args.n_repeats, 5, args.n_permutations),
        )
        handle.create_dataset("time", data=times)
        handle.create_dataset("permuted_labels", data=permuted_labels)
        handle.create_dataset("null_window_statistics", data=null_stats)
        handle.attrs["observed_window_statistic"] = observed_stat
        handle.attrs["p_value"] = p_value
        handle.attrs["classifier"] = "LinearSVC"
        handle.attrs["variance"] = 0.95
        handle.attrs["n_folds"] = 5
        handle.attrs["n_repeats"] = args.n_repeats
        handle.attrs["n_permutations"] = args.n_permutations
        handle.attrs["random_seed"] = args.seed
        handle.attrs["window_seconds"] = window
        handle.attrs["step_seconds"] = step
        handle.attrs["window_label_start"] = 0.0
        handle.attrs["window_label_end"] = 0.5
        handle.attrs["scoring"] = "balanced_accuracy"
        handle.attrs["elapsed_seconds"] = time.time() - started

    summary = {
        "output": str(args.output),
        "observed_window_statistic": observed_stat,
        "null_mean": float(null_stats.mean()),
        "null_95th_percentile": float(np.quantile(null_stats, 0.95)),
        "p_value": p_value,
        "n_timepoints": len(times),
        "timepoints": times.round(2).tolist(),
        "n_folds": 5,
        "n_repeats": args.n_repeats,
        "n_permutations": args.n_permutations,
        "elapsed_seconds": time.time() - started,
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
