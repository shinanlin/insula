#!/usr/bin/env python3
"""CLI entry point for insula functional NMF waveform analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

from src.nmf.waveform_analysis import TASKS, run
from src.paths import RESULTS_ROOT, nmf_results_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stimulus-only insula NMF with held-out phase validation."
    )
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Default: results/nmf for k=2, results/nmf/k{k} otherwise",
    )
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument(
        "--exclude-subject",
        action="append",
        default=["D0121"],
        help="Repeatable; D0121 matches vizpub/fig2.ipynb",
    )
    parser.add_argument("--k", type=int, default=2, choices=(2, 3))
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--n-permutations", type=int, default=10_000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = (
            nmf_results_dir() if args.k == 2 else nmf_results_dir() / f"k{args.k}"
        )
    return args


def main() -> None:
    matplotlib.use("Agg")
    run(parse_args())


if __name__ == "__main__":
    main()
