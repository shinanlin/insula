"""Cross-condition window decoding.

Train on one condition (e.g., Repeat), test on another (e.g., Decision)
for the same ROI and phase, using a specific time window.

Usage:
    python src/run_cross_condition_window.py \
        --bids_root /cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS \
        --roi AICl \
        --phase Delay \
        --train_on Repeat \
        --test_on Decision \
        --n_perm 100 --n_folds 10 --n_jobs 40
"""

import rootutils
path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import argparse
import logging
import sys
import os
import time as _time

import h5py
import numpy as np
from mne_bids import BIDSPath
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from mne.decoding import Vectorizer
from ieeg.calc.oversample import MinimumNaNSplit

from src.decoding.direct_cross_decoder import (
    DirectCrossDecoder,
    direct_cross_domain_permutation_scores,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def load_intersection_condition(bids_root, ref, roi, phase, description, band, datatype):
    """Load a single condition from the intersection dataset."""
    root = BIDSPath(
        root=os.path.join(bids_root, "derivatives", f"decoding(intersection)({ref})"),
        datatype=datatype,
        suffix=band,
        subject=roi,
        description=description,
        processing=phase,
        task="LexicalDelay",
        extension=".h5",
        check=False,
    )
    files = root.match()
    if not files:
        raise FileNotFoundError(
            f"No intersection file for ROI={roi}, phase={phase}, desc={description}. "
            f"Searched: {root.fpath}"
        )

    fpath = files[0]
    logger.info(f"Loading {fpath}")

    with h5py.File(fpath, "r") as f:
        X = f["X"][()]
        y = f["y"][()]
        channels = [
            ch.decode("utf-8") if isinstance(ch, bytes) else ch
            for ch in f["channel"][()]
        ]
        meta = {
            "fs": int(f.attrs["fs"]),
            "tmin": float(f.attrs["tmin"]),
            "tmax": float(f.attrs["tmax"]),
            "channels": channels,
            "event_id": f.attrs["event_id"],
        }

    return X, y, meta


def get_phase_window(phase):
    """Return the time window (start, end) for a given phase in seconds."""
    match phase:
        case 'Stimulus':
            tmin, tmax = 0.0, 0.75
        case 'Go':
            tmin, tmax = 0, 0.5
        case 'Response':
            tmin, tmax = -0.5, 1
        case 'Delay':
            tmin, tmax = 0.0, 1
        case _:
            raise ValueError(f"Unknown phase: {phase}")
    return tmin, tmax


def main(
    bids_root,
    ref,
    roi,
    phase,
    train_on,
    test_on,
    band,
    datatype,
    variance,
    n_perm,
    n_folds,
    n_jobs,
):
    t0 = _time.time()
    logger.info(f"=== Cross-condition WINDOW decoding ===")
    logger.info(f"ROI={roi}, Phase={phase}, Train={train_on}, Test={test_on}")

    # 1. Load both conditions from intersection dataset
    X_train, y_train, meta_train = load_intersection_condition(
        bids_root, ref, roi, phase, train_on, band, datatype
    )
    X_test, y_test, meta_test = load_intersection_condition(
        bids_root, ref, roi, phase, test_on, band, datatype
    )

    # Verify channels match
    assert meta_train["channels"] == meta_test["channels"], "Channel mismatch!"
    assert meta_train["fs"] == meta_test["fs"], "Sampling rate mismatch"

    fs = meta_train["fs"]
    data_tmin = meta_train["tmin"]
    n_times = X_train.shape[-1]
    data_tmax = data_tmin + (n_times / fs)

    # Extract time window for this phase
    window_tmin, window_tmax = get_phase_window(phase)
    
    # Clip window to data availability (just in case)
    window_tmin = max(window_tmin, data_tmin)
    window_tmax = min(window_tmax, data_tmax)
    
    start_sample = int(round((window_tmin - data_tmin) * fs))
    end_sample = int(round((window_tmax - data_tmin) * fs))
    
    X_train = X_train[..., start_sample:end_sample]
    X_test = X_test[..., start_sample:end_sample]

    # Align trial counts — CV splits must use the same indices for both
    n_min = min(len(y_train), len(y_test))
    if len(y_train) != len(y_test):
        logger.info(f"  Aligning trial counts: {len(y_train)} vs {len(y_test)} → {n_min}")
        X_train, y_train = X_train[:n_min], y_train[:n_min]
        X_test, y_test = X_test[:n_min], y_test[:n_min]

    logger.info(f"  Channels: {len(meta_train['channels'])}, "
                f"Trials (aligned): {min(len(y_train), len(y_test))}")
    logger.info(f"  Window: [{window_tmin:.3f}, {window_tmax:.3f}] seconds, fs={fs}")

    # 2. Build pipeline
    estimator = make_pipeline(
        Vectorizer(),
        StandardScaler(),
        PCA(n_components=variance, random_state=42),
        LinearSVC(random_state=42, max_iter=10000),
    )

    cross_decoder = DirectCrossDecoder(
        estimator=estimator,
        random_state=42,
    )

    cv = MinimumNaNSplit(n_splits=n_folds, n_repeats=5)

    # 3. Run window decoding
    logger.info(f"  Running window decoding: perm={n_perm}, folds={n_folds}, jobs={n_jobs}")

    obs_scores, perm_scores, p_value = direct_cross_domain_permutation_scores(
        X1=X_train,
        y1=y_train,
        X2=X_test,
        y2=y_test,
        cv=cv,
        cross_decoder=cross_decoder,
        scoring="accuracy",
        n_permutations=n_perm,
        n_jobs=n_jobs,
        random_state=42,
    )

    elapsed = _time.time() - t0
    obs_mean = np.mean(obs_scores)
    logger.info(f"  Done in {elapsed:.1f}s")
    logger.info(f"  Observed accuracy: {obs_mean:.3f}, p-value: {p_value:.4f}")

    # 4. Save results
    save_path = BIDSPath(
        root=f"./results/LexicalDelay(roi)({ref})",
        datatype=f"(cross)(window){datatype}",
        subject=roi,
        description=f"{train_on}2{test_on}",
        processing=phase,
        suffix=band,
        extension=".h5",
        check=False,
    )
    save_path.mkdir(exist_ok=True)

    with h5py.File(save_path.fpath, "w") as f:
        f.create_dataset("scores", data=obs_scores)
        f.create_dataset("baseline", data=perm_scores)
        f.create_dataset("p_value", data=p_value)

        f.attrs["roi"] = roi
        f.attrs["phase"] = phase
        f.attrs["train_on"] = train_on
        f.attrs["test_on"] = test_on
        f.attrs["band"] = band
        f.attrs["datatype"] = datatype
        f.attrs["variance"] = variance
        f.attrs["n_permutations"] = n_perm
        f.attrs["n_folds"] = n_folds
        f.attrs["fs"] = fs
        f.attrs["window_tmin"] = window_tmin
        f.attrs["window_tmax"] = window_tmax

    logger.info(f"Saved to {save_path.fpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-condition window decoding"
    )
    parser.add_argument("--bids_root", type=str,
                        default="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS")
    parser.add_argument("--roi", type=str, default="AICl",
                        help="ROI name (e.g., AICl, STGl)")
    parser.add_argument("--phase", type=str, default="Delay",
                        help="Phase (e.g., Delay, Response, Stimulus, Go)")
    parser.add_argument("--train_on", type=str, default="Repeat",
                        choices=["Repeat", "Decision"],
                        help="Condition to train on")
    parser.add_argument("--test_on", type=str, default="Decision",
                        choices=["Repeat", "Decision"],
                        help="Condition to test on")
    parser.add_argument("--ref", type=str, default="bipolar",
                        choices=["bipolar", "car"])
    parser.add_argument("--band", type=str, default="highgamma")
    parser.add_argument("--datatype", type=str, default="lexicality")
    parser.add_argument("--variance", type=float, default=0.85,
                        help="PCA variance retained")
    parser.add_argument("--n_perm", type=int, default=100,
                        help="Number of permutations")
    parser.add_argument("--n_folds", type=int, default=10,
                        help="Number of CV folds")
    parser.add_argument("--n_jobs", type=int, default=40,
                        help="Parallel jobs")
    args = parser.parse_args()
    main(**vars(args))
