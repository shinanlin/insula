"""Batch-generate interactive 3D xcorr viewers for all qualifying subjects.

A subject qualifies if it has both Insula and IFG electrodes in the same
hemisphere AND has epoch data for the requested phase/desc/band.

Usage:
    python src/batch_xcorr_viewer.py --phase Response --desc Repeat
"""

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path

import pandas as pd

from src.xcorr.generate_xcorr_viewer import (
    RECON_DIR,
    classify_channels,
    filter_same_hemisphere,
    generate_viewer,
    load_parcellation,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
DATASETS = {
    "LexicalDecRepDelay": "/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/",
    "Phoneme_sequencing": "/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/",
    "LexicalDecRepNoDelay": "/cwork/ns458/BIDS-1.0_LexicalDecRepNoDelay/BIDS/",
    "PictureNaming": "/cwork/ns458/BIDS-1.3_PictureNaming/BIDS/",
    "SentenceRep": "/cwork/ns458/BIDS-1.4_SentenceRep/BIDS/",
}


def find_qualifying_subjects(phase: str, desc: str, band: str = "highgamma"):
    """Find (subject, bids_root) pairs that have both Insula+IFG in same hemi
    and have epoch data for the given phase/desc."""
    from mne_bids import BIDSPath

    results = []
    seen = set()

    for task_name, bids_root in DATASETS.items():
        parc_dir = Path(bids_root) / "derivatives" / "parcellation"
        if not parc_dir.exists():
            continue
        for sub_dir in sorted(parc_dir.glob("sub-*")):
            subject = sub_dir.name.replace("sub-", "")
            if subject in seen:
                continue

            try:
                parc = load_parcellation(bids_root, subject)
            except FileNotFoundError:
                continue

            ins, ifg, stg, hg = classify_channels(parc)
            other_roi_count = len(ifg) + len(stg) + len(hg)
            if not ins or other_roi_count == 0:
                continue
            ins, ifg, stg, hg = filter_same_hemisphere(ins, ifg, stg, hg, parc)
            other_roi_count = len(ifg) + len(stg) + len(hg)
            if not ins or other_roi_count == 0:
                continue

            # Check epoch file exists
            ep = BIDSPath(
                root=os.path.join(bids_root, "derivatives", "epoch(bipolar)"),
                datatype="epoch(band)(zscore)",
                subject=subject,
                suffix=band,
                processing=phase,
                extension=".h5",
                check=False,
            )
            matches = [m for m in ep.match() if m.description == desc]
            if not matches:
                continue

            seen.add(subject)
            results.append((subject, bids_root, task_name))
            logger.info(
                f"  Qualified: {subject} ({task_name}) "
                f"INS={len(ins)} IFG={len(ifg)} STG={len(stg)} HG={len(hg)}"
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch generate xcorr viewers")
    parser.add_argument("--phase", default="Response")
    parser.add_argument("--desc", default="Repeat")
    parser.add_argument("--band", default="highgamma")
    parser.add_argument("--recon_dir", default=RECON_DIR)
    parser.add_argument(
        "--output_dir", default="viz/3d_xcorr",
        help="Directory for output HTML files",
    )
    parser.add_argument("--max_lag_s", type=float, default=1.0)
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip subjects that already have an HTML file")
    args = parser.parse_args()

    logger.info("Scanning datasets for qualifying subjects...")
    subjects = find_qualifying_subjects(args.phase, args.desc, args.band)
    logger.info(f"Found {len(subjects)} qualifying subjects\n")

    os.makedirs(args.output_dir, exist_ok=True)
    success, failed = [], []

    skipped = 0
    for i, (subject, bids_root, task_name) in enumerate(subjects, 1):
        out_path = os.path.join(
            args.output_dir, f"{subject}_{args.phase}_{args.desc}.html"
        )
        if args.skip_existing and os.path.exists(out_path):
            skipped += 1
            continue
        logger.info(f"[{i}/{len(subjects)}] {subject} ({task_name})")
        try:
            generate_viewer(
                bids_root=bids_root,
                subject=subject,
                phase=args.phase,
                desc=args.desc,
                band=args.band,
                recon_dir=args.recon_dir,
                output=out_path,
                max_lag_s=args.max_lag_s,
            )
            success.append(subject)
        except Exception as e:
            logger.error(f"  FAILED {subject}: {e}")
            traceback.print_exc()
            failed.append(subject)

    # Summary
    logger.info("\n" + "=" * 60)
    if skipped:
        logger.info(f"SKIPPED: {skipped} (already exist)")
    logger.info(f"SUCCESS: {len(success)}/{len(subjects) - skipped}")
    if failed:
        logger.info(f"FAILED:  {', '.join(failed)}")
    logger.info(f"Output:  {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
