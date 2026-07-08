#!/usr/bin/env python3
"""Pick the canonical bipolar parcellation CSV for a subject across BIDS datasets."""

from __future__ import annotations

from pathlib import Path

BIDS_ROOTS = {
    "LexicalDecRepDelay": Path("/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS"),
    "LexicalDecRepNoDelay": Path("/cwork/ns458/BIDS-1.0_LexicalDecRepNoDelay/BIDS"),
    "Phoneme_sequencing": Path("/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS"),
    "PictureNaming": Path("/cwork/ns458/BIDS-1.3_PictureNaming/BIDS"),
    "SentenceRep": Path("/cwork/ns458/BIDS-1.4_SentenceRep/BIDS"),
}

PRIORITY = [
    "LexicalDecRepDelay",
    "Phoneme_sequencing",
    "LexicalDecRepNoDelay",
    "SentenceRep",
    "PictureNaming",
]


def parcellation_csv(bids_root: Path, subject: str) -> Path:
    return (
        bids_root
        / "derivatives"
        / "parcellation"
        / f"sub-{subject}"
        / "bipolar"
        / f"sub-{subject}_aparc2009s.csv"
    )


def find_parcellation(subject: str) -> Path:
    for name in PRIORITY:
        path = parcellation_csv(BIDS_ROOTS[name], subject)
        if path.is_file():
            return path
    raise FileNotFoundError(f"No bipolar parcellation CSV found for {subject}")
