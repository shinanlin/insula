#!/usr/bin/env python3
"""Summarize validated task-specific MAPER parcellation derivatives."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def counts(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return (
        table.groupby(columns, dropna=False).size()
        .rename("channels").reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, sep="\t")
    ready = manifest[manifest["status"] == "ready"]
    tables = []
    for row in ready.itertuples(index=False):
        table = pd.read_csv(row.output)
        table.insert(0, "maper_task", row.task)
        tables.append(table)
    cohort = pd.concat(tables, ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {
        "status": counts(cohort, ["maper_insula_status"]),
        "ap": counts(cohort, ["maper_ap_consensus", "maper_ap_mix"]),
        "agreement": counts(cohort, ["maper_atlas_agreement"]),
        "by_task": counts(
            cohort,
            ["maper_task", "maper_insula_status", "maper_atlas_agreement"],
        ),
    }
    for name, summary in summaries.items():
        summary.to_csv(args.output_dir / f"maper_cohort_{name}.csv", index=False)

    missing = manifest[manifest["status"] != "ready"].copy()
    missing.to_csv(args.output_dir / "maper_cohort_missing.tsv", sep="\t", index=False)
    overview = pd.DataFrame([{
        "manifest_combinations": len(manifest),
        "ready_combinations": len(ready),
        "missing_combinations": len(missing),
        "channels": len(cohort),
    }])
    overview.to_csv(args.output_dir / "maper_cohort_overview.csv", index=False)
    print(overview.to_string(index=False))
    print(f"Wrote cohort summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
