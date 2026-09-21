#!/usr/bin/env python3
"""Build subject-level soft NNLS/OAEC embedding tables for the notebook."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.nmf.oaec_soft_embedding import (
    DEFAULT_COMPONENTS,
    attach_soft_target_weights,
    collapse_insula_seeds,
    condition_network_embedding,
    group_sign_flip_inference,
    load_oaec_pair_tables,
    prepare_oaec_edges,
    prepare_soft_projection,
    subject_embedding,
)


PAIR_COLUMNS = (
    "source",
    "target",
    "source_is_seed",
    "target_is_seed",
    "source_effective",
    "target_effective",
    "metric",
    "stat",
    "null_mean",
    "null_std",
    "p_uncorrected",
    "q_fdr",
    "p_fwer_maxstat",
    "qc_pass",
    "dataset",
    "subject",
    "task",
    "phase",
    "description",
    "recording",
    "run",
    "acquisition",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--connectivity-root",
        type=Path,
        default=PROJECT / "results" / "connectivity",
    )
    parser.add_argument(
        "--projection",
        type=Path,
        default=(
            PROJECT
            / "results"
            / "nmf"
            / "whole_brain_projection"
            / "electrode_projection.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT
            / "results"
            / "nmf"
            / "whole_brain_projection"
            / "soft_oaec_embedding"
        ),
    )
    parser.add_argument("--description", default="Repeat")
    parser.add_argument("--min-explained-energy", type=float, default=0.0)
    parser.add_argument("--min-targets", type=int, default=1)
    parser.add_argument("--require-target-effective", action="store_true")
    parser.add_argument("--n-resamples", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    projection = prepare_soft_projection(
        pd.read_csv(args.projection),
        min_explained_energy=args.min_explained_energy,
    )
    raw = load_oaec_pair_tables(
        args.connectivity_root,
        columns=PAIR_COLUMNS,
    )
    edges = prepare_oaec_edges(
        raw,
        description=args.description,
        require_target_effective=args.require_target_effective,
    )
    weighted = attach_soft_target_weights(edges, projection)
    targets = collapse_insula_seeds(weighted)
    conditions = condition_network_embedding(
        targets,
        min_targets=args.min_targets,
    )
    subjects = subject_embedding(conditions, keep_phase=False)
    subject_phases = subject_embedding(conditions, keep_phase=True)
    inference_parts = []
    for value in ("embedding", "network_selectivity"):
        inference_parts.append(
            group_sign_flip_inference(
                subjects,
                value_column=value,
                n_permutations=args.n_resamples,
                n_bootstrap=args.n_resamples,
            )
        )
    inference = pd.concat(inference_parts, ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "soft_projection.csv.gz": projection,
        "target_level.csv.gz": targets,
        "condition_network.csv.gz": conditions,
        "subject_overall.csv": subjects,
        "subject_phase.csv": subject_phases,
        "group_inference.csv": inference,
    }
    for name, frame in tables.items():
        path = args.output_dir / name
        frame.to_csv(path, index=False)
        print(f"wrote {path} ({len(frame):,} rows)")
    print(
        f"raw={len(raw):,} filtered_edges={len(edges):,} "
        f"soft_edges={len(weighted):,} subjects={subjects['subject'].nunique()}"
    )
    print(inference.to_string(index=False))


if __name__ == "__main__":
    main()
