"""Cross-condition generalized decoding (2D train-time × test-time).

Train on one condition (e.g., Repeat), test on another (e.g., Decision)
for the same ROI and phase. Uses DirectCrossDecoder from direct_cross_decoder.py
with pre-aligned electrode intersection datasets.

Usage:
    python src/run_cross_condition_generalized.py \
        --bids_root /cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS \
        --roi AICl \
        --phase Delay \
        --train_on Repeat \
        --test_on Decision \
        --window 0.2 --step 0.1 \
        --n_perm 200 --n_folds 10 --n_jobs 24
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
from sklearn.base import clone
from mne.decoding import Vectorizer
from ieeg.calc.oversample import MinimumNaNSplit

from src.decoding.direct_cross_decoder import (
    DirectCrossDecoder,
    direct_cross_domain_generalized_permutation_scores,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def load_intersection_condition(bids_root, ref, roi, phase, description, band, datatype):
    """Load a single condition from the intersection dataset.
    
    Parameters
    ----------
    bids_root : str
        BIDS root directory.
    ref : str
        Reference type (e.g., 'bipolar').
    roi : str
        ROI name (e.g., 'AICl').
    phase : str
        Phase name (e.g., 'Delay', 'Response').
    description : str
        Condition name (e.g., 'Repeat', 'Decision').
    band : str
        Frequency band (e.g., 'highgamma').
    datatype : str
        Data type (e.g., 'lexicality').
        
    Returns
    -------
    X : ndarray, shape (n_trials, n_channels, n_times)
    y : ndarray, shape (n_trials,)
    meta : dict with keys 'fs', 'tmin', 'tmax', 'channels', 'event_id'
    """
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

    logger.info(f"  Shape: {X.shape}, labels: {np.unique(y)}, channels: {len(channels)}")
    return X, y, meta


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
    window,
    step,
    n_perm,
    n_folds,
    n_jobs,
):
    t0 = _time.time()
    logger.info(f"=== Cross-condition generalized decoding ===")
    logger.info(f"ROI={roi}, Phase={phase}, Train={train_on}, Test={test_on}")

    # 1. Load both conditions from intersection dataset
    X_train, y_train, meta_train = load_intersection_condition(
        bids_root, ref, roi, phase, train_on, band, datatype
    )
    X_test, y_test, meta_test = load_intersection_condition(
        bids_root, ref, roi, phase, test_on, band, datatype
    )

    # Verify channels match (they should, by construction)
    assert meta_train["channels"] == meta_test["channels"], (
        f"Channel mismatch! Train: {len(meta_train['channels'])}, "
        f"Test: {len(meta_test['channels'])}. "
        "This should not happen with intersection datasets."
    )
    assert meta_train["fs"] == meta_test["fs"], "Sampling rate mismatch"

    fs = meta_train["fs"]
    tmin = meta_train["tmin"]
    # Compute effective tmax from actual data dimensions.
    # The h5 tmax can exceed what arange+window can safely slice,
    # because direct_cross_domain_generalized_permutation_scores uses
    # arange(tmin+window, tmax+step, step) which may generate time points
    # beyond the data length. The inner test loop has no bounds check,
    # so we clip tmax to the actual data range here.
    n_times = X_train.shape[-1]
    effective_tmax = tmin + (n_times / fs)
    tmax = effective_tmax
    logger.info(f"  Effective tmax (from data): {tmax:.4f} "
                f"(h5 tmax: {meta_train['tmax']:.4f})")

    # Align trial counts — CV splits must use the same indices for both
    n_min = min(len(y_train), len(y_test))
    if len(y_train) != len(y_test):
        logger.info(f"  Aligning trial counts: {len(y_train)} vs {len(y_test)} → {n_min}")
        X_train, y_train = X_train[:n_min], y_train[:n_min]
        X_test, y_test = X_test[:n_min], y_test[:n_min]

    logger.info(f"  Channels: {len(meta_train['channels'])}, "
                f"Trials (aligned): {len(y_train)}")
    logger.info(f"  Time range: [{tmin}, {tmax}], fs={fs}")

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

    cv = MinimumNaNSplit(n_splits=n_folds, n_repeats=1)

    # 3. Run generalized decoding
    logger.info(f"  Running 2D generalized decoding: window={window}s, step={step}s, "
                f"perm={n_perm}, folds={n_folds}")

    obs_scores, perm_scores, pvals_fdr = direct_cross_domain_generalized_permutation_scores(
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
        window=window,
        step=step,
        fs=fs,
        train_tmin=tmin,
        train_tmax=tmax,
        test_tmin=tmin,
        test_tmax=tmax,
    )

    elapsed = _time.time() - t0
    logger.info(f"  Done in {elapsed:.1f}s")
    logger.info(f"  Observed shape: {obs_scores.shape}, "
                f"mean accuracy: {obs_scores.mean():.3f}")
    logger.info(f"  Significant cells (FDR<0.05): {(pvals_fdr < 0.05).sum()} / {pvals_fdr.size}")

    # 4. Compute time axes for saving
    train_time = np.arange(tmin + window, tmax + step, step)
    test_time = np.arange(tmin + window, tmax + step, step)

    # 5. Save results
    save_path = BIDSPath(
        root=f"./results/LexicalDelay(roi)({ref})",
        datatype=f"(cross)(generalized){datatype}",
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
        f.create_dataset("p_values", data=pvals_fdr)
        f.create_dataset("train_time", data=train_time)
        f.create_dataset("test_time", data=test_time)

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
        f.attrs["tmin"] = tmin
        f.attrs["tmax"] = tmax
        f.attrs["window"] = window
        f.attrs["step"] = step

    logger.info(f"Saved to {save_path.fpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-condition generalized decoding (2D train-time × test-time)"
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
    parser.add_argument("--window", type=float, default=0.3,
                        help="Sliding window length (seconds)")
    parser.add_argument("--step", type=float, default=0.03,
                        help="Sliding window step (seconds)")
    parser.add_argument("--n_perm", type=int, default=200,
                        help="Number of permutations")
    parser.add_argument("--n_folds", type=int, default=10,
                        help="Number of CV folds")
    parser.add_argument("--n_jobs", type=int, default=24,
                        help="Parallel jobs")
    args = parser.parse_args()
    main(**vars(args))
