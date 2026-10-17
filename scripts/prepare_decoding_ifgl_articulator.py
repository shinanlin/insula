#!/usr/bin/env python3
"""Run the external PhonemeSequence preparation for IFGl articulator only."""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path


DEFAULT_SOURCE = Path(
    "/hpc/group/coganlab/nanlinshi/seeg-preprocessing-worktrees/"
    "phoneme_seq/task/prepare_decoding_dataset.py"
)
PHASES = frozenset({"Stimulus", "Delay", "Response"})


def load_external(path: Path):
    spec = importlib.util.spec_from_file_location("phoneme_seq_prepare_decoding", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load external preparation script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids-root", required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()

    prepare = load_external(args.source)
    prepare.HAMMERS_ROI_GROUPS["IFG"] = ["IFG"]

    original_get_roi_channels = prepare.get_roi_channels

    def get_ifgl_channels(parc, roi, hemi):
        if roi != "IFG" or hemi != "L":
            return [], []
        return original_get_roi_channels(parc, roi, hemi)

    prepare.get_roi_channels = get_ifgl_channels

    original_match = prepare.BIDSPath.match

    def match_required_epochs(path, *match_args, **match_kwargs):
        matches = original_match(path, *match_args, **match_kwargs)
        if "epoch(bipolar)" not in str(path.root):
            return matches
        return [
            candidate
            for candidate in matches
            if candidate.description == "Repeat" and candidate.processing in PHASES
        ]

    prepare.BIDSPath.match = match_required_epochs
    logging.getLogger(__name__).info(
        "Preparing IFGl articulator for Repeat phases: %s", sorted(PHASES)
    )
    prepare.main(
        bids_root=args.bids_root,
        band="highgamma",
        task="articulator",
        reference="bipolar",
        rois=["IFG"],
        atlas="hammers",
        input_datatype="epoch(band)(zscore)",
    )


if __name__ == "__main__":
    main()
