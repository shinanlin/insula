#!/usr/bin/env python3
"""CLI: project whole-brain packaged HGA onto frozen Insula NMF templates."""

from __future__ import annotations

import argparse
import logging

import rootutils

rootutils.setup_root(
    __file__,
    indicator=".project-root",
    pythonpath=True,
    cwd=True,
)

from src.nmf.waveform_analysis import TASKS
from src.nmf.whole_brain_projection import (
    DEFAULT_EXCLUDE_SUBJECTS,
    PHASES,
    SIGNIFICANCE_MODES,
    discover_cohort_subjects,
    run_whole_brain_projection,
)
from src.paths import nmf_assignments_path, nmf_wholebrain_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Subject IDs (default: all Repeat HGA subjects minus --exclude-subject)",
    )
    parser.add_argument(
        "--exclude-subject",
        nargs="*",
        default=list(DEFAULT_EXCLUDE_SUBJECTS),
        help="Subjects to drop (default: D0121)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASKS),
        help=f"Tasks to pool per electrode (default: {' '.join(TASKS)})",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        default=list(PHASES),
        help=f"Phases (default: {' '.join(PHASES)})",
    )
    parser.add_argument(
        "--significance-mode",
        choices=list(SIGNIFICANCE_MODES),
        default="mask-any",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.95,
        help="Per-phase non-NaN fraction required (default: 0.95)",
    )
    parser.add_argument(
        "--min-tasks",
        type=int,
        default=1,
        help="Minimum tasks in which a channel must appear (default: 1)",
    )
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument(
        "--pilot-min-explained-energy",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--assignments",
        type=str,
        default=None,
        help="channel_assignments.csv for in_discovery (default: canonical)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output root (default: results/nmf/whole_brain_projection/)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip per-subject reconstruction SVGs",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    subjects = args.subjects
    if subjects is None:
        subjects = list(
            discover_cohort_subjects(
                args.tasks,
                exclude_subjects=args.exclude_subject,
            )
        )
        logging.info(
            "Discovered %d cohort subjects (excluded %s)",
            len(subjects),
            args.exclude_subject,
        )
    manifest = run_whole_brain_projection(
        subjects,
        tasks=args.tasks,
        phases=args.phases,
        significance_mode=args.significance_mode,
        min_coverage=args.min_coverage,
        min_tasks=args.min_tasks,
        ridge=args.ridge,
        pilot_min_explained_energy=args.pilot_min_explained_energy,
        assignments_path=args.assignments or nmf_assignments_path(),
        output_dir=args.output_dir or nmf_wholebrain_dir(),
        make_plots=not args.no_plots,
        exclude_subjects=args.exclude_subject,
    )
    print(f"Wrote manifest: {manifest['output_dir']}/manifest.json")
    print(f"Combined table: {manifest['combined_table']}")
    print(
        f"subjects used={manifest['n_subjects_used']} "
        f"skipped={manifest['n_subjects_skipped']} "
        f"electrodes={manifest['n_electrodes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
