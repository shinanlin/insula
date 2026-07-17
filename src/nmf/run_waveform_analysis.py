#!/usr/bin/env python3
"""CLI entry point for insula functional NMF waveform analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

from src.nmf.waveform_analysis import TASKS, run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stimulus-only insula NMF with held-out phase validation."
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("tmp/nmf_corrected")
    )
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument(
        "--exclude-subject",
        action="append",
        default=["D0121"],
        help="Repeatable; D0121 matches vizpub/fig2.ipynb",
    )
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--n-permutations", type=int, default=10_000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    matplotlib.use("Agg")
    run(parse_args())


if __name__ == "__main__":
    main()
