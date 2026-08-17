#!/usr/bin/env python3
"""Compute PCA scree and PC-space clustering tables (no figures)."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.nmf.pc_clustering import (
    DEFAULT_K_MAX,
    DEFAULT_K_MIN,
    DEFAULT_N_ITER,
    DEFAULT_N_SCREE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_VARIANCE_THRESHOLD,
    run,
)
from src.nmf.waveform_analysis import TASKS
from src.paths import RESULTS_ROOT, nmf_assignments_path, nmf_exclude_channels_path


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
    parser.add_argument(
        "--assignments",
        type=Path,
        default=None,
        help=f"Default: {nmf_assignments_path()}",
    )
    parser.add_argument("--n-scree", type=int, default=DEFAULT_N_SCREE)
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=DEFAULT_VARIANCE_THRESHOLD,
    )
    parser.add_argument("--k-min", type=int, default=DEFAULT_K_MIN)
    parser.add_argument("--k-max", type=int, default=DEFAULT_K_MAX)
    parser.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k_min < 2 or args.k_max < args.k_min:
        raise SystemExit("--k-min/--k-max must satisfy 2 <= k_min <= k_max")
    result = run(
        results_root=args.results_root,
        tasks=tuple(args.tasks),
        exclude_subjects=set(args.exclude_subject),
        exclude_channels_file=args.exclude_channels_file,
        assignments_path=args.assignments,
        n_scree=args.n_scree,
        variance_threshold=args.variance_threshold,
        k_min=args.k_min,
        k_max=args.k_max,
        n_iter=args.n_iter,
        random_state=args.random_state,
        results_dir=args.results_dir,
    )
    print(
        f"Done. n_electrodes={result['n_electrodes']} "
        f"n_embedding_pcs={result['n_embedding_pcs']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
