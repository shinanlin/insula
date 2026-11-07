"""Shared, re-stratified repeated-CV engine for the approved Figure 4 analyses.

Each global null draw repeats the complete learning algorithm, including
label-dependent splitting, original full-reference imputation and train-only
standardisation/PCA. No transform from a true-label fold is reused for a
different permutation split. Historical frozen-split diagnostics remain in
``repeated_cv.py`` and are not inputs to these formal results.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from ieeg.calc.oversample import MinimumNaNSplit
from mne.decoding import Vectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.decoding.decoder import sample_fold

METHOD = "repeated_cv_mean_restratified_whole_stimulus"
SCHEMA_VERSION = 3
SE_DESCRIPTION = (
    "SE across CV-repeat means; split variability conditional on this fixed "
    "dataset, not participant/population uncertainty"
)


def unit_solver_specification(spec, plan, model, mode):
    """Apply a recorded solver ceiling amendment only to its reviewed unit.

    The base specification and identities of every other running/completed
    unit remain unchanged. This changes no data, folds, estimator objective,
    preprocessing, tolerance or inference statistic.
    """
    amendments = plan.get("solver_amendments", [])
    matches = [a for a in amendments if
        (a["mode"], a["roi"], a["phase"]) == (mode, model["roi"], model["phase"])]
    if len(matches) > 1:
        raise ValueError("Multiple solver amendments target the same unit")
    if not matches:
        return spec, None
    amendment = matches[0]
    allowed = {"mode", "roi", "phase", "max_iter", "approved_date", "reason",
        "base_specification_sha256", "approval_file", "approval_sha256"}
    if set(amendment) - allowed:
        raise ValueError("A solver ceiling amendment contains other parameter changes")
    if amendment["base_specification_sha256"] != plan["specification_sha256"]:
        raise ValueError("Solver amendment belongs to a different base analysis")
    if not amendment.get("approval_sha256") or int(amendment["max_iter"]) <= 0:
        raise ValueError("Missing approved solver amendment provenance")
    return dict(spec, max_iter=int(amendment["max_iter"])), amendment


class ConvergenceFailure(RuntimeError):
    """A fit failed the approved acceptance gate; do not discard the draw."""


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_windows(origin, fs, n_samples, *, tmin, tmax, window, step):
    """Return complete windows and their actual sample-center coordinates.

    Retain the historical integer window length (38 samples at 128 Hz).
    Round each requested start independently to avoid accumulated step error;
    30 ms is not an integer number of samples at 128 Hz. Never extrapolate.
    """
    width = int(float(window) * fs)
    if width < 1 or step <= 0 or tmax - tmin < window:
        raise ValueError("Invalid sliding-window specification")
    count = int(np.floor((tmax - tmin - window) / step + 1e-9)) + 1
    result = []
    for index in range(count):
        nominal_start = tmin + index * step
        first = int(np.rint((nominal_start - origin) * fs))
        last = first + width
        if first < 0 or last > n_samples:
            raise ValueError(f"Incomplete requested window {index}: {first}:{last}")
        start = origin + first / fs
        final_sample = origin + (last - 1) / fs
        result.append(dict(index=index, first=first, last=last,
            nominal_start=nominal_start, nominal_end=nominal_start + window,
            start=start, end_exclusive=origin + last / fs,
            time=(start + final_sample) / 2, n_samples=width))
    return result


def phase_window(origin, fs, n_samples, bounds):
    """Keep the original load_roi_data half-open, integer-index crop rule."""
    first, last = [int(fs * (float(t) - origin)) for t in bounds]
    if first < 0 or last > n_samples or last <= first:
        raise ValueError(f"Incomplete phase window {bounds}: {first}:{last}")
    return dict(first=first, last=last, start=origin + first/fs,
        end_exclusive=origin + last/fs,
        time=origin + (first + last - 1)/(2*fs), n_samples=last-first,
        nominal_start=float(bounds[0]), nominal_end=float(bounds[1]))


def pipeline(spec):
    return make_pipeline(
        Vectorizer(), StandardScaler(),
        PCA(n_components=float(spec["variance"]), random_state=42),
        LinearSVC(C=float(spec["C"]), random_state=42,
            max_iter=int(spec["max_iter"]), tol=float(spec["tol"])),
    )


def evaluate_labels(X, labels, keep, spec):
    """Evaluate a true or null label vector with exactly the same algorithm.

    Full-reference channels enter splitting/imputation. ``keep`` is applied
    only afterwards, making the 13/14-electrode comparison paired without
    changing the 203 input rows. The splitter never removes a row.
    """
    X, labels = np.asarray(X), np.asarray(labels)
    keep = np.asarray(keep, dtype=int)
    nr, nf = int(spec["n_repeats"]), int(spec["n_folds"])
    seed = int(spec["random_state"])
    classes = np.unique(labels)
    if X.ndim != 3 or len(X) != len(labels) or len(keep) < 1:
        raise ValueError("Expected aligned trials x channels x time")
    scores = np.empty((nr, nf))
    iterations = np.empty((nr, nf), dtype=np.int32)
    components = np.empty((nr, nf), dtype=np.int16)
    assignment = np.full((nr, len(labels)), 255, dtype=np.uint8)
    predictions = np.empty_like(labels)
    imputation_digest = hashlib.sha256()
    for repeat in range(nr):
        cv = MinimumNaNSplit(n_splits=nf, n_repeats=1, random_state=seed+repeat)
        folds = list(cv.split(X, labels))
        if len(folds) != nf:
            raise ValueError("CV did not produce the requested fold count")
        coverage = np.sort(np.concatenate([te for _, te in folds]))
        if not np.array_equal(coverage, np.arange(len(labels))):
            raise ValueError("CV did not hold out every fixed input row once")
        for fold, (train, test) in enumerate(folds):
            if (not np.array_equal(np.unique(labels[train]), classes) or
                    not np.array_equal(np.unique(labels[test]), classes)):
                raise ValueError("A stratified fold lost a class; do not omit it")
            a, b, ya, yb = sample_fold(X, labels, train, test,
                seed=seed+repeat+fold)
            imputation_digest.update(np.ascontiguousarray(a).tobytes())
            imputation_digest.update(np.ascontiguousarray(b).tobytes())
            model = pipeline(spec)
            model.fit(a[:, keep, :], ya)
            svc = model.named_steps["linearsvc"]
            iterations[repeat, fold] = int(svc.n_iter_)
            if svc.n_iter_ >= spec["max_iter"]:
                raise ConvergenceFailure(
                    f"repeat={repeat}, fold={fold}, n_iter={svc.n_iter_}, "
                    f"max_iter={spec['max_iter']}; no draw was discarded")
            if not (np.isfinite(svc.coef_).all() and
                    np.isfinite(svc.intercept_).all()):
                raise ConvergenceFailure("Nonfinite classifier coefficients")
            predicted = model.predict(b[:, keep, :])
            scores[repeat, fold] = balanced_accuracy_score(yb, predicted)
            components[repeat, fold] = model.named_steps["pca"].n_components_
            assignment[repeat, test] = fold
            if repeat == 0:
                predictions[test] = predicted
    return dict(scores=scores, iterations=iterations, components=components,
        fold_assignment=assignment, first_repeat_predictions=predictions,
        imputation_sha256=imputation_digest.hexdigest())


def inference(observed, null):
    """One overall statistic vs matched overall null statistics, plus repeat SE."""
    observed, null = np.asarray(observed), np.asarray(null)
    if observed.ndim != 2 or null.ndim != 3 or null.shape[1:] != observed.shape:
        raise ValueError("True and null statistics must cover the same repeats/folds")
    if not np.isfinite(observed).all() or not np.isfinite(null).all():
        raise ValueError("An incomplete/failed permutation cannot enter inference")
    repeat_means = observed.mean(axis=1)
    mean = float(repeat_means.mean())
    null_means = null.mean(axis=(1, 2))
    exceedances = int(np.count_nonzero(null_means >= observed.mean()))
    return dict(accuracy=repeat_means, accuracy_stable=mean,
        accuracy_se=float(repeat_means.std(ddof=1)/np.sqrt(len(repeat_means))),
        permutation_means=null_means, null_exceedances=exceedances,
        p_value=(exceedances+1)/(len(null_means)+1))
