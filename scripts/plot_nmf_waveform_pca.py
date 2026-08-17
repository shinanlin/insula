#!/usr/bin/env python3
"""PCA scatter of concat-NMF waveforms, colored by frozen cluster labels."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.nmf.waveform_analysis import TASKS
from src.nmf.waveform_pca import run
from src.paths import RESULTS_ROOT, nmf_assignments_path, nmf_exclude_channels_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument(
        "--exclude-subject",
        action="append",
        default=["D0121"],
        help="Repeatable; D0121 matches the canonical concat-NMF fit",
    )
    parser.add_argument(
        "--exclude-channels-file",
        type=Path,
        default=None,
        help=f"Default: {nmf_exclude_channels_path()}",
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=None,
        help=f"Default: {nmf_assignments_path()}",
    )
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--scores-csv", type=Path, default=None)
    parser.add_argument("--meta-json", type=Path, default=None)
    parser.add_argument(
        "--svg",
        type=Path,
        default=None,
        help="Default: img/nmf/waveform_pca.svg",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_components < 2:
        raise SystemExit("--n-components must be >= 2")
    result = run(
        results_root=args.results_root,
        tasks=tuple(args.tasks),
        exclude_subjects=set(args.exclude_subject),
        exclude_channels_file=args.exclude_channels_file,
        assignments_path=args.assignments,
        n_components=args.n_components,
        scores_path=args.scores_csv,
        meta_path=args.meta_json,
        svg_path=args.svg,
    )
    print(
        f"Done. n_shared={result['n_shared']} "
        f"explained={result['explained_variance_ratio']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
