#!/usr/bin/env python3
"""Discover the MAPER subject union across project BIDS datasets.

Matches the five BIDS roots used in src/hga/package_highgamma.py. A subject is included
if it has a bipolar parcellation CSV in any dataset and FreeSurfer recon
(orig.mgz + brainmask.mgz) under ECoG_Recon.
"""

from __future__ import annotations

import argparse
from pathlib import Path

BIDS_ROOTS = {
    "LexicalDecRepDelay": Path("/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS"),
    "LexicalDecRepNoDelay": Path("/cwork/ns458/BIDS-1.0_LexicalDecRepNoDelay/BIDS"),
    "Phoneme_sequencing": Path("/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS"),
    "PictureNaming": Path("/cwork/ns458/BIDS-1.3_PictureNaming/BIDS"),
    "SentenceRep": Path("/cwork/ns458/BIDS-1.4_SentenceRep/BIDS"),
}

DEFAULT_RECON_ROOT = Path("/cwork/ns458/ECoG_Recon")


def recon_dir(recon_root: Path, subject: str) -> Path:
    return recon_root / f"D{int(subject.lstrip('D0'))}"


def has_parcellation(bids_root: Path, subject: str) -> bool:
    csv = (
        bids_root
        / "derivatives"
        / "parcellation"
        / f"sub-{subject}"
        / "bipolar"
        / f"sub-{subject}_aparc2009s.csv"
    )
    return csv.is_file()


def has_recon(recon_root: Path, subject: str) -> bool:
    mri = recon_dir(recon_root, subject) / "mri"
    return (mri / "orig.mgz").is_file() and (mri / "brainmask.mgz").is_file()


def discover(recon_root: Path) -> list[str]:
    subjects: set[str] = set()
    for bids_root in BIDS_ROOTS.values():
        parc_dir = bids_root / "derivatives" / "parcellation"
        if not parc_dir.is_dir():
            continue
        for sub_dir in parc_dir.glob("sub-*"):
            subject = sub_dir.name.replace("sub-", "")
            if has_parcellation(bids_root, subject):
                subjects.add(subject)

    ready = sorted(s for s in subjects if has_recon(recon_root, s))
    missing_recon = sorted(s for s in subjects if s not in ready)
    if missing_recon:
        raise SystemExit(
            f"Subjects with parcellation but missing recon: {missing_recon}"
        )
    return ready


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recon-root", type=Path, default=DEFAULT_RECON_ROOT)
    parser.add_argument(
        "--write",
        type=Path,
        help="Optional path to write one subject ID per line.",
    )
    args = parser.parse_args()

    subjects = discover(args.recon_root)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text("\n".join(subjects) + "\n")
        print(f"Wrote {len(subjects)} subjects to {args.write}")

    for subject in subjects:
        print(subject)


if __name__ == "__main__":
    main()
