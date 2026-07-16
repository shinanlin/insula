#!/usr/bin/env python3
"""CLI for multi-block ridge encoding (semantic / phon / acoustic)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import rootutils

rootutils.setup_root(
    __file__,
    indicator=".project-root",
    pythonpath=True,
    cwd=True,
)

from src.paths import RESULTS_ROOT
from src.semantic.design_matrix import DEFAULT_BIDS_ROOT, load_trial_design_multi
from src.semantic.load_stimulus_features import DEFAULT_FEATURES_H5
from src.semantic.ridge_encode_multi import (
    MODEL_SPECS,
    ridge_encode_multi_block,
    ridge_encode_multi_with_significance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = RESULTS_ROOT / "semantic" / "LexicalDelay"


def output_filename(subject: str, phase: str, description: str, model: str) -> str:
    subject = subject.replace("sub-", "")
    return (
        f"sub-{subject}_task-LexicalDelay_proc-{phase}_desc-{description}"
        f"_ridge_{model}.h5"
    )


def save_result(
    out_path: Path,
    result,
    design,
    *,
    random_state: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("r", data=result.r_map, compression="gzip")
        f.create_dataset("times", data=design.times)
        ch_bytes = [c.encode("utf-8") for c in design.ch_names]
        f.create_dataset("channel", data=np.asarray(ch_bytes, dtype="S"))
        f.create_dataset("tokens", data=design.tokens.astype("S"))
        f.create_dataset("groups", data=design.groups.astype("S"))

        if result.n_perm > 0 and result.r_null is not None:
            f.create_dataset("r_null", data=result.r_null, compression="gzip")
            f.create_dataset("baseline", data=result.r_null, compression="gzip")
            f.create_dataset("mask", data=result.mask.astype(np.uint8))
            f.create_dataset("p_values", data=result.p_values, compression="gzip")

        f.attrs["subject"] = design.subject
        f.attrs["phase"] = design.phase
        f.attrs["description"] = design.description
        f.attrs["tmin"] = design.tmin
        f.attrs["tmax"] = design.tmax
        f.attrs["model"] = result.model
        f.attrs["feature_blocks"] = json.dumps(list(result.feature_blocks))
        f.attrs["k_pca_per_block"] = json.dumps(result.k_pca_per_block)
        f.attrs["perm_shuffled_block"] = (
            result.perm_shuffled_block if result.perm_shuffled_block else ""
        )
        f.attrs["alpha"] = result.alpha
        f.attrs["n_splits"] = result.n_splits
        f.attrs["n_trials"] = int(design.Y.shape[0])
        f.attrs["n_tokens"] = design.n_unique_tokens
        f.attrs["n_channels"] = design.n_channels
        f.attrs["n_times"] = design.n_times
        f.attrs["random_state"] = random_state
        f.attrs["mean_abs_r"] = float(np.nanmean(np.abs(result.r_flat)))
        f.attrs["max_abs_r"] = float(np.nanmax(np.abs(result.r_flat)))
        f.attrs["n_perm"] = int(result.n_perm)
        if result.n_perm > 0:
            f.attrs["p_thresh"] = float(result.p_thresh)
            f.attrs["significance_method"] = "channel_time_cluster"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--phase", type=str, default="Delay")
    parser.add_argument("--description", type=str, default="Decision")
    parser.add_argument(
        "--model",
        type=str,
        default="full_perm_semantic",
        choices=sorted(MODEL_SPECS.keys()),
    )
    parser.add_argument("--tmin", type=float, default=-0.5)
    parser.add_argument("--tmax", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--bids_root", type=str, default=str(DEFAULT_BIDS_ROOT))
    parser.add_argument("--features_h5", type=str, default=str(DEFAULT_FEATURES_H5))
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--n_perm", type=int, default=500)
    parser.add_argument("--p_thresh", type=float, default=0.05)
    parser.add_argument("--n_jobs", type=int, default=2)
    args = parser.parse_args()

    subject = args.subject.replace("sub-", "")
    design = load_trial_design_multi(
        subject=subject,
        phase=args.phase,
        description=args.description,
        tmin=args.tmin,
        tmax=args.tmax,
        bids_root=args.bids_root,
        features_h5=args.features_h5,
    )
    logger.info(
        "sub-%s %s/%s model=%s: %d trials, %d tokens, %d ch, blocks=%s",
        subject,
        args.phase,
        args.description,
        args.model,
        design.Y.shape[0],
        design.n_unique_tokens,
        design.n_channels,
        list(MODEL_SPECS[args.model]["blocks"]),
    )

    if args.n_perm > 0:
        result = ridge_encode_multi_with_significance(
            design,
            model=args.model,
            alpha=args.alpha,
            n_splits=args.n_splits,
            random_state=args.random_state,
            n_perm=args.n_perm,
            p_thresh=args.p_thresh,
            n_jobs=args.n_jobs,
        )
    else:
        result = ridge_encode_multi_block(
            design,
            model=args.model,
            alpha=args.alpha,
            n_splits=args.n_splits,
            random_state=args.random_state,
            shuffle_within_folds=False,
        )

    out_dir = Path(args.out_dir) / f"sub-{subject}"
    out_path = out_dir / output_filename(
        subject, args.phase, args.description, args.model
    )
    save_result(out_path, result, design, random_state=args.random_state)

    msg = (
        f"Wrote {out_path} (mean|r|={float(np.nanmean(np.abs(result.r_flat))):.4f}, "
        f"max|r|={float(np.nanmax(np.abs(result.r_flat))):.4f}"
    )
    if args.n_perm > 0 and result.mask is not None:
        msg += f", sig_ch={int(result.mask.any(axis=1).sum())}/{result.mask.shape[0]}"
    logger.info(msg + ")")


if __name__ == "__main__":
    main()
