#!/usr/bin/env python3
"""Time-resolved decoding with shrinkage LDA and a pooled out-of-fold AUC.

Same sliding-window design as ``run_decoding_resolved.py`` (train on t, test on
t), but retuned for pooled ROIs that carry few channels, where the production
``PCA + LinearSVC`` on balanced accuracy is too blunt to resolve anything:

===============  ==============================  ==============================
                 run_decoding_resolved.py        this module
===============  ==============================  ==============================
estimator        PCA(0.90) + LinearSVC           shrinkage LDA (lsqr)
features         all window samples flattened    5 time bins per channel
metric           balanced accuracy               AUC, macro OVR if multiclass
AUC pooling      n/a                             pooled out-of-fold
CV               MinimumNaNSplit                 ``--cv_scheme``
null             trial-level shuffle             word-level shuffle
===============  ==============================  ==============================

``--cv_scheme`` selects the fold assignment independently of the null. Binary
lexicality uses ``group``, where a word never straddles train and test, because
the claim there is that the Word/Nonword split generalises to unseen words.
Articulator uses ``stratified``, matching the production runs it is compared
against; the null stays at the word level either way, so item memorisation is
represented in the null rather than being credited to the observed score.

Outputs go to the ``(decode)(resolved)(lda){datatype}`` datatype so the
production ``(decode)(resolved){datatype}`` results are left untouched and the
two can be compared side by side.
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
from mne.decoding import Vectorizer
from mne_bids import BIDSPath
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.decoding.decoder import decode_permutation_auc_pooled
from src.decoding.pooled_io import load_pooled_roi
from src.decoding.run_decoding_resolved import cluster_correction
from src.paths import decoding_task_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
# Macro one-vs-rest AUC keeps chance at 0.5 for any number of classes.
CHANCE_AUC = 0.5

CV_SCHEMES = {
    "group": "StratifiedGroupKFold(condition)",
    "stratified": "StratifiedKFold",
}


class TimeBin(BaseEstimator, TransformerMixin):
    """Average a window into ``n_bins`` contiguous time bins.

    A 0.3 s window at 128 Hz is 38 samples, so a per-channel-per-sample feature
    space runs to hundreds of dimensions against ~130 trials. Binning to ~60 ms
    keeps the within-window time course but brings the feature count back to
    something shrinkage LDA can estimate a covariance for.
    """

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.shape[-1] < self.n_bins:
            raise ValueError(
                f"Window has {X.shape[-1]} samples, fewer than n_bins={self.n_bins}"
            )
        chunks = np.array_split(X, self.n_bins, axis=-1)
        return np.stack([chunk.mean(axis=-1) for chunk in chunks], axis=-1)


def build_transformer(n_bins: int):
    """Label-free feature preparation, so it can be fit once per training split."""
    return make_pipeline(TimeBin(n_bins=n_bins), Vectorizer(), StandardScaler())


def build_classifier():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def build_pipeline(n_bins: int):
    """The two halves composed, for callers that want a single estimator."""
    return make_pipeline(build_transformer(n_bins), build_classifier())


def build_cv(cv_scheme: str, n_folds: int):
    """Fold assignment only; the null's exchangeability unit is set separately.

    ``stratified`` is still handed the grouping vector at ``split`` time, where
    scikit-learn ignores it. Folds may then split a stimulus across train and
    test while the permutation null stays at the stimulus level.
    """
    if cv_scheme == "group":
        return StratifiedGroupKFold(
            n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED
        )
    if cv_scheme == "stratified":
        return StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED
        )
    raise ValueError(f"Unknown cv_scheme {cv_scheme!r}, expected {sorted(CV_SCHEMES)}")


def drop_label_trials(X, y, conditions, labels, drop_labels):
    """Remove trials whose string label is in ``drop_labels``.

    ``sequence_articulator`` assigns a catch-all ``"other"`` to any phoneme
    outside the four articulatory buckets, and the preparation step keeps it as
    a class. Filtering here rather than in the preparation step leaves the
    on-disk inputs, and the production results built from them, untouched.
    """
    if not drop_labels:
        return X, y, conditions, labels, 0
    keep = ~np.isin(labels, list(drop_labels))
    if not keep.any():
        raise ValueError(f"Dropping {sorted(drop_labels)} removed every trial")
    return X[keep], y[keep], conditions[keep], labels[keep], int((~keep).sum())


def main(
    bids_root,
    ref,
    subject,
    description,
    phase,
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
        raise ValueError(f"Unknown cv_scheme {cv_scheme!r}, expected {sorted(CV_SCHEMES)}")
    drop_labels = tuple(drop_labels or ())

    X, y, conditions, labels, channels, _trials, src_path = load_pooled_roi(
        bids_root, ref, subject, datatype, description, phase, band, tmin, tmax
    )
    logger.info("Loaded %s", src_path)

    X, y, conditions, labels, n_dropped = drop_label_trials(
        X, y, conditions, labels, drop_labels
    )
    if n_dropped:
        logger.info("Dropped %d trials labelled %s", n_dropped, sorted(drop_labels))

    class_counts = {label: int((labels == label).sum()) for label in sorted(set(labels))}
    logger.info(
        "X %s, y %s, %d channels, %d unique stimuli, classes %s",
        X.shape,
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
    window_samples = int(window * fs)

    time_points, aucs, baselines, p_perm = [], [], [], []
    run_t0 = _time.time()

    for time_end in np.arange(tmin + window, tmax + step, step):
        end_sample = int((time_end - tmin) * fs) + 1
        start_sample = end_sample - window_samples
        if start_sample < 0 or end_sample > X.shape[-1]:
            logger.warning("Window out of bounds at %.3fs, skipping", time_end)
            continue

        X_segment = X[..., start_sample:end_sample].copy()
        cv = build_cv(cv_scheme, n_folds)
        obs_auc, perm_aucs, p_value = decode_permutation_auc_pooled(
            X_segment,
            y,
            cv,
            classifier,
            transformer=transformer,
            groups=conditions,
            permute_groups=True,
            n_jobs=n_jobs,
            n_permutations=n_perm,
            random_state=RANDOM_SEED,
        )
        logger.info(
            "t=%.3fs  AUC=%.3f  p=%.4f  (%.1fs elapsed)",
            time_end,
            obs_auc,
            p_value,
            _time.time() - run_t0,
        )

        time_points.append(time_end)
        aucs.append(obs_auc)
        # Degenerate permutations come back as NaN; cluster_correction cannot
        # take them, and a dropped null draw is best replaced by chance.
        baselines.append(np.where(np.isfinite(perm_aucs), perm_aucs, CHANCE_AUC))
        p_perm.append(p_value)
        del X_segment

    if not time_points:
        raise ValueError("No sliding window fitted inside the requested bounds")

    time_points = np.asarray(time_points)
    aucs = np.asarray(aucs)
    p_perm = np.asarray(p_perm)
    baseline = np.stack(baselines, axis=1)  # (n_perm, n_time)

    mask, p_values = cluster_correction(aucs, baseline)

    save_path = BIDSPath(
        root=str(decoding_task_dir(str(src_path.task))),
        datatype="(decode)(resolved)(lda)" + str(datatype),
        subject=subject,
        suffix=band,
        processing=src_path.processing,
        description=src_path.description,
        recording=src_path.recording,
        extension=".h5",
        check=False,
    )
    save_path.mkdir(exist_ok=True)
    logger.info("Saving results to: %s", save_path)

    with h5py.File(save_path, "w") as stream:
        stream.create_dataset(name="auc", data=aucs)
        stream.create_dataset(name="baseline", data=baseline)
        stream.create_dataset(name="time", data=time_points)
        stream.create_dataset(name="mask", data=mask)
        stream.create_dataset(name="p_values", data=p_values)
        stream.create_dataset(name="p_perm", data=p_perm)
        stream.create_dataset(
            name="channel", data=np.array(channels, dtype=h5py.string_dtype())
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
            "roc_auc_pooled_oof" if len(class_counts) == 2 else "macro_ovr_auc_pooled_oof"
        )
        stream.attrs["cv_scheme"] = CV_SCHEMES[cv_scheme]
        stream.attrs["perm_scheme"] = "group_label_shuffle(condition)"
        stream.attrs["source"] = str(src_path)

    logger.info("Completed in %.2fs", _time.time() - run_t0)
    del X, baseline
    gc.collect()
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--bids_root", type=str,
                        default="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/",
                        help="Root directory of the BIDS dataset")
    parser.add_argument("--subject", type=str, default="STGl",
                        help="Pooled ROI to process")
    parser.add_argument("--ref", type=str, default="bipolar",
                        choices=["car", "bipolar"],
                        help="Reference scheme")
    parser.add_argument("--description", type=str, default="Repeat",
                        choices=["Repeat", "Decision", "Passive"],
                        help="Trial description")
    parser.add_argument("--phase", type=str, default="Stimulus",
                        choices=["Stimulus", "Delay", "Go", "Response"],
                        help="Epoch alignment")
    parser.add_argument("--band", type=str, default="highgamma",
                        help="Frequency band of the neural signal")
    parser.add_argument("--datatype", type=str, default="lexicality",
                        choices=["phoneme", "articulator", "structure",
                                 "word", "token", "lexicality"])
    parser.add_argument("--n_bins", type=int, default=5,
                        help="Time bins per sliding window")
    parser.add_argument("--window", type=float, default=0.3,
                        help="Window length in seconds")
    parser.add_argument("--step", type=float, default=0.03,
                        help="Step size in seconds")
    parser.add_argument("--n_perm", type=int, default=200,
                        help="Number of label permutations (production: 5000)")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument("--cv_scheme", type=str, default="group",
                        choices=sorted(CV_SCHEMES),
                        help="Fold assignment; the null stays word-level either way")
    parser.add_argument("--drop_labels", nargs="*", default=[],
                        help="String labels to exclude, e.g. 'other' for "
                             "PhonemeSequence articulator")
    parser.add_argument("--n_jobs", type=int, default=2,
                        help="Number of parallel jobs")
    parser.add_argument("--tmin", type=float, default=-0.5, help="tmin")
    parser.add_argument("--tmax", type=float, default=1.5, help="tmax")

    args = parser.parse_args()
    main(**vars(args))
