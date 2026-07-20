"""Cross-ROI window decoding with CCA alignment.

Train on one ROI (AIC) and test on another ROI within the same task, phase,
and condition, using a single phase-specific time window per run.

Uses CrossDecoder and cross_domain_permutation_scores from cross_decoder.py.
"""

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import argparse
import logging
import os
import sys
import time as _time

import h5py
import numpy as np
from mne_bids import BIDSPath
from mne.decoding import Vectorizer
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from ieeg.calc.oversample import MinimumNaNSplit

from src.decoding.cross_decoder import (
    CrossDecoder,
    _balance_datasets,
    cross_domain_permutation_scores,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

TASK_DEFAULTS = {
    "LexicalDelay": (
        "/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS",
        "lexicality",
    ),
    "PhonemeSequence": (
        "/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS",
        "articulator",
    ),
}


def get_phase_window(phase):
    """Phase windows aligned with run_decoding.py / grant aim2 window figures."""
    from src.decoding.run_decoding import PHASE_WINDOWS

    try:
        return PHASE_WINDOWS[phase]
    except KeyError as exc:
        raise ValueError(f"Unknown phase: {phase}") from exc


def load_roi_condition(
    bids_root,
    ref,
    task,
    roi,
    phase,
    description,
    band,
    datatype,
    recording=None,
):
    root = BIDSPath(
        root=os.path.join(bids_root, "derivatives", f"decoding({ref})"),
        datatype=datatype,
        suffix=band,
        subject=roi,
        description=description,
        processing=phase,
        task=task,
        extension=".h5",
        check=False,
    )
    if recording is not None:
        root = root.update(recording=str(recording))

    files = sorted(root.match(), key=lambda p: str(p.fpath))
    if not files:
        raise FileNotFoundError(
            f"No decoding file for task={task}, ROI={roi}, phase={phase}, "
            f"desc={description}, recording={recording}. Searched: {root.fpath}"
        )

    fpath = files[0]
    logger.info("Loading %s", fpath)

    with h5py.File(fpath, "r") as f:
        X = f["X"][()]
        y = f["y"][()]
        meta = {
            "fs": int(f.attrs["fs"]),
            "tmin": float(f.attrs["tmin"]),
            "tmax": float(f.attrs["tmax"]),
            "event_id": f.attrs.get("event_id", ""),
            "fpath": str(fpath),
        }

    logger.info("  Shape: %s, labels: %s", X.shape, np.unique(y))
    return X, y, meta


def crop_phase_window(X, meta, phase):
    fs = meta["fs"]
    data_tmin = meta["tmin"]
    n_times = X.shape[-1]
    data_tmax = data_tmin + (n_times / fs)

    window_tmin, window_tmax = get_phase_window(phase)
    window_tmin = max(window_tmin, data_tmin)
    window_tmax = min(window_tmax, data_tmax)

    start_sample = int(round((window_tmin - data_tmin) * fs))
    end_sample = int(round((window_tmax - data_tmin) * fs))
    X = X[..., start_sample:end_sample]
    return X, window_tmin, window_tmax


def main(
    task,
    bids_root,
    ref,
    train_roi,
    test_roi,
    phase,
    description,
    band,
    datatype,
    variance,
    n_components,
    n_perm,
    n_folds,
    n_jobs,
    recording,
):
    default_root, expected_dtype = TASK_DEFAULTS[task]
    if bids_root is None:
        bids_root = default_root
    if datatype is None:
        datatype = expected_dtype
    if datatype != expected_dtype:
        raise ValueError(
            f"task={task} expects datatype={expected_dtype}, got {datatype}"
        )
    if task == "PhonemeSequence" and recording is None:
        recording = "1"

    t0 = _time.time()
    logger.info("=== Cross-ROI WINDOW decoding (CCA) ===")
    logger.info(
        "task=%s train=%s test=%s phase=%s desc=%s dtype=%s recording=%s",
        task,
        train_roi,
        test_roi,
        phase,
        description,
        datatype,
        recording,
    )

    X1, y1, meta1 = load_roi_condition(
        bids_root, ref, task, train_roi, phase, description, band, datatype, recording=recording,
    )
    X2, y2, meta2 = load_roi_condition(
        bids_root, ref, task, test_roi, phase, description, band, datatype, recording=recording,
    )
    assert meta1["fs"] == meta2["fs"], "Sampling rate mismatch"

    X1, window_tmin, window_tmax = crop_phase_window(X1, meta1, phase)
    X2, _, _ = crop_phase_window(X2, meta2, phase)

    X1, X2, y1, y2 = _balance_datasets(X1, y1, X2, y2)

    n_ch1, n_ch2 = X1.shape[1], X2.shape[1]
    if n_ch1 == 0:
        logger.warning("Train ROI %s has 0 channels. Skipping.", train_roi)
        return
    if n_ch2 == 0:
        logger.warning("Test ROI %s has 0 channels. Skipping.", test_roi)
        return

    logger.info("  Channels: train=%d, test=%d", n_ch1, n_ch2)
    logger.info("  Balanced trials: %d", len(y1))
    logger.info("  Window: [%.3f, %.3f] s", window_tmin, window_tmax)

    estimator = make_pipeline(
        Vectorizer(),
        StandardScaler(),
        PCA(n_components=variance, random_state=42),
        LinearSVC(random_state=42, max_iter=10000),
    )
    cross_decoder = CrossDecoder(
        estimator=estimator,
        n_components=n_components,
        random_state=42,
    )
    cv = MinimumNaNSplit(n_splits=n_folds, n_repeats=1)

    obs_scores, perm_scores, p_value = cross_domain_permutation_scores(
        X1=X1, y1=y1, X2=X2, y2=y2,
        cv=cv, cross_decoder=cross_decoder,
        scoring="accuracy", n_permutations=n_perm, n_jobs=n_jobs, random_state=42,
    )

    elapsed = _time.time() - t0
    logger.info("  Done in %.1fs | acc=%.3f p=%.4f", elapsed, float(np.mean(obs_scores)), p_value)

    save_path = BIDSPath(
        root=f"./results/{task}(cross_roi)({ref})",
        datatype=f"(cross)(window){datatype}",
        subject=f"{train_roi}2{test_roi}",
        description=description,
        processing=phase,
        task=task,
        suffix=band,
        extension=".h5",
        check=False,
    )
    save_path.mkdir(exist_ok=True)

    with h5py.File(save_path.fpath, "w") as f:
        f.create_dataset("scores", data=obs_scores)
        f.create_dataset("baseline", data=perm_scores)
        f.create_dataset("p_value", data=p_value)
        f.attrs["task"] = task
        f.attrs["train_roi"] = train_roi
        f.attrs["test_roi"] = test_roi
        f.attrs["phase"] = phase
        f.attrs["description"] = description
        f.attrs["band"] = band
        f.attrs["datatype"] = datatype
        f.attrs["variance"] = variance
        f.attrs["n_components"] = n_components
        f.attrs["n_permutations"] = n_perm
        f.attrs["n_folds"] = n_folds
        f.attrs["fs"] = meta1["fs"]
        f.attrs["window_tmin"] = window_tmin
        f.attrs["window_tmax"] = window_tmax
        if recording is not None:
            f.attrs["recording"] = str(recording)

    logger.info("Saved to %s", save_path.fpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-ROI window decoding with CCA")
    parser.add_argument("--task", required=True, choices=["LexicalDelay", "PhonemeSequence"])
    parser.add_argument("--bids_root", default=None)
    parser.add_argument("--train_roi", default="AICl")
    parser.add_argument("--test_roi", required=True)
    parser.add_argument("--phase", required=True, choices=["Stimulus", "Delay", "Go", "Response"])
    parser.add_argument("--description", default="Repeat", choices=["Repeat", "Decision"])
    parser.add_argument("--ref", default="bipolar", choices=["bipolar", "car"])
    parser.add_argument("--band", default="highgamma")
    parser.add_argument("--datatype", default=None)
    parser.add_argument("--recording", default=None)
    parser.add_argument("--variance", type=float, default=0.80)
    parser.add_argument("--n_components", type=int, default=5)
    parser.add_argument("--n_perm", type=int, default=100)
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--n_jobs", type=int, default=40)
    main(**vars(parser.parse_args()))
