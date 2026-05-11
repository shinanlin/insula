"""Cross-ROI generalized decoding with CCA alignment.

Train on one ROI and test on another ROI within the same phase and condition.
Uses CrossDecoder from cross_decoder.py, so decoding follows:
CCA alignment -> Vectorizer/StandardScaler -> PCA -> LinearSVC.
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

from src.cross_decoder import (
    CrossDecoder,
    _balance_datasets,
    cross_domain_generalized_permutation_scores,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def load_roi_condition(bids_root, ref, roi, phase, description, band, datatype):
    root = BIDSPath(
        root=os.path.join(bids_root, "derivatives", f"decoding({ref})"),
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
            f"No decoding file for ROI={roi}, phase={phase}, desc={description}. "
            f"Searched: {root.fpath}"
        )

    fpath = files[0]
    logger.info(f"Loading {fpath}")

    with h5py.File(fpath, "r") as f:
        X = f["X"][()]
        y = f["y"][()]
        meta = {
            "fs": int(f.attrs["fs"]),
            "tmin": float(f.attrs["tmin"]),
            "tmax": float(f.attrs["tmax"]),
            "event_id": f.attrs.get("event_id", ""),
        }

    logger.info(f"  Shape: {X.shape}, labels: {np.unique(y)}")
    return X, y, meta


def main(
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
    window,
    step,
    n_perm,
    n_folds,
    n_jobs,
):
    t0 = _time.time()
    logger.info("=== Cross-ROI generalized decoding ===")
    logger.info(f"Train ROI={train_roi}, Test ROI={test_roi}, Phase={phase}, Description={description}")

    X1, y1, meta1 = load_roi_condition(
        bids_root, ref, train_roi, phase, description, band, datatype
    )
    X2, y2, meta2 = load_roi_condition(
        bids_root, ref, test_roi, phase, description, band, datatype
    )
    assert meta1["fs"] == meta2["fs"], "Sampling rate mismatch"

    X1, X2, y1, y2 = _balance_datasets(X1, y1, X2, y2)

    # Check for zero channels (missing electrode coverage)
    n_ch1, n_ch2 = X1.shape[1], X2.shape[1]
    if n_ch1 == 0:
        logger.warning(f"Train ROI {train_roi} has 0 channels for phase={phase}, desc={description}. Skipping.")
        return
    if n_ch2 == 0:
        logger.warning(f"Test ROI {test_roi} has 0 channels for phase={phase}, desc={description}. Skipping.")
        return
    logger.info(f"  Channels: train={n_ch1}, test={n_ch2}")

    fs = meta1["fs"]
    tmin = meta1["tmin"]
    n_times = min(X1.shape[-1], X2.shape[-1])
    X1 = X1[..., :n_times]
    X2 = X2[..., :n_times]
    tmax = tmin + (n_times / fs)

    logger.info(f"  Balanced trials: {len(y1)}")
    logger.info(f"  Time range: [{tmin}, {tmax}], fs={fs}")

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

    logger.info(f"  Running 2D generalized decoding: window={window}s, step={step}s, "
                f"perm={n_perm}, folds={n_folds}")

    obs_scores, perm_scores, pvals_fdr = cross_domain_generalized_permutation_scores(
        X1=X1,
        y1=y1,
        X2=X2,
        y2=y2,
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
    logger.info(f"  Observed shape: {obs_scores.shape}, mean accuracy: {obs_scores.mean():.3f}")
    logger.info(f"  Significant cells (FDR<0.05): {(pvals_fdr < 0.05).sum()} / {pvals_fdr.size}")

    train_time = np.arange(tmin + window, tmax + step, step)
    test_time = np.arange(tmin + window, tmax + step, step)

    save_path = BIDSPath(
        root=f"./results/LexicalDelay(cross_roi)({ref})",
        datatype=f"(cross)(generalized){datatype}",
        subject=f"{train_roi}2{test_roi}",
        description=description,
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
        f.attrs["fs"] = fs
        f.attrs["tmin"] = tmin
        f.attrs["tmax"] = tmax
        f.attrs["window"] = window
        f.attrs["step"] = step

    logger.info(f"Saved to {save_path.fpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-ROI generalized decoding with CCA alignment"
    )
    parser.add_argument("--bids_root", type=str,
                        default="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS")
    parser.add_argument("--train_roi", type=str, default="AICl")
    parser.add_argument("--test_roi", type=str, default="SMCl")
    parser.add_argument("--phase", type=str, default="Delay")
    parser.add_argument("--description", type=str, default="Repeat",
                        choices=["Repeat", "Decision"])
    parser.add_argument("--ref", type=str, default="bipolar",
                        choices=["bipolar", "car"])
    parser.add_argument("--band", type=str, default="highgamma")
    parser.add_argument("--datatype", type=str, default="lexicality")
    parser.add_argument("--variance", type=float, default=0.80)
    parser.add_argument("--n_components", type=int, default=5)
    parser.add_argument("--window", type=float, default=0.2)
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument("--n_perm", type=int, default=100)
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--n_jobs", type=int, default=40)
    args = parser.parse_args()
    main(**vars(args))
