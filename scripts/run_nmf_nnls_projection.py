#!/usr/bin/env python3
"""CLI: fixed-W NNLS projection of single-trial HGA (pilot LexicalDelay)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nmf.nnls_projection import PHASES, run_nnls_projection  # noqa: E402
from src.paths import nmf_assignments_path, nmf_nnls_dir  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        default="LexicalDelay",
        help="BIDS task name (default: LexicalDelay)",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        default=list(PHASES),
        help=f"Phases to project (default: {' '.join(PHASES)})",
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=None,
        help="channel_assignments.csv (default: results/nmf/channel_assignments.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root (default: results/nmf/nnls_projection/)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    manifest = run_nnls_projection(
        task=args.task,
        phases=args.phases,
        assignments_path=args.assignments or nmf_assignments_path(),
        output_dir=args.output_dir or nmf_nnls_dir(),
    )
    print(f"Wrote manifest: {manifest['output_dir']}/manifest.json")
    print(f"Overview SVG: {manifest['svg']}")
    for phase, detail in manifest["phases_detail"].items():
        print(
            f"  {phase}: n_trials={detail['n_trials']} "
            f"subjects={detail['n_subjects_used']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
