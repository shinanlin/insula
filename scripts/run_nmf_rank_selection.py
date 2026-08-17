#!/usr/bin/env python3
"""Electrode bootstrap consensus rank selection for concat-NMF."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.nmf.rank_selection import (
    DEFAULT_B,
    DEFAULT_K_MAX,
    DEFAULT_K_MIN,
    DEFAULT_MAX_ITER,
    DEFAULT_RANDOM_STATE,
    DEFAULT_ROW_FRAC,
    run,
)
from src.nmf.waveform_analysis import TASKS
from src.paths import RESULTS_ROOT, nmf_exclude_channels_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument(
        "--exclude-subject",
        action="append",
        default=["D0121"],
    )
    parser.add_argument(
        "--exclude-channels-file",
        type=Path,
        default=None,
        help=f"Default: {nmf_exclude_channels_path()}",
    )
    parser.add_argument("--k-min", type=int, default=DEFAULT_K_MIN)
    parser.add_argument("--k-max", type=int, default=DEFAULT_K_MAX)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_B)
    parser.add_argument("--row-frac", type=float, default=DEFAULT_ROW_FRAC)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--images-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k_min < 2 or args.k_max < args.k_min:
        raise SystemExit("--k-min/--k-max must satisfy 2 <= k_min <= k_max")
    decision = run(
        results_root=args.results_root,
        tasks=tuple(args.tasks),
        exclude_subjects=set(args.exclude_subject),
        exclude_channels_file=args.exclude_channels_file,
        k_min=args.k_min,
        k_max=args.k_max,
        n_boot=args.n_boot,
        row_frac=args.row_frac,
        max_iter=args.max_iter,
        random_state=args.random_state,
        results_dir=args.results_dir,
        images_dir=args.images_dir,
    )
    print(f"Done. chosen_k={decision['k']}", flush=True)


if __name__ == "__main__":
    main()
