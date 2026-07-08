#!/usr/bin/env python3
"""Validate task-specific MAPER derivatives against source aparc tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


LOCATIONS = {"contact_1", "center", "contact_2"}
INSULA_STATUSES = {"core", "partial", "none"}
AGREEMENT_LABELS = {
    "concordant_insula", "maper_only", "aparc_only", "concordant_noninsula",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--failure-manifest", type=Path,
        help="Write only failed ready rows as a retry manifest.",
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, sep="\t")
    ready = manifest[manifest["status"] == "ready"]
    keys: list[tuple[str, str, str, str]] = []
    failures: list[str] = []
    failed_indices: list[int] = []
    total_rows = 0

    for manifest_index, row in ready.iterrows():
        try:
            source = pd.read_csv(row["parcellation_csv"])
            output = pd.read_csv(row["output"])
            sensitivity = pd.read_csv(row["sensitivity_output"])
            if len(source) != len(output):
                raise AssertionError(f"row count {len(output)} != {len(source)}")
            assert_frame_equal(
                output[source.columns], source,
                check_dtype=False, check_exact=False, rtol=0, atol=1e-12,
            )
            if output["name"].duplicated().any():
                raise AssertionError("duplicate channel names")
            if set(output["maper_insula_status"].dropna()) - INSULA_STATUSES:
                raise AssertionError("invalid maper_insula_status")
            if set(output["maper_atlas_agreement"].dropna()) - AGREEMENT_LABELS:
                raise AssertionError("invalid maper_atlas_agreement")
            expected_status = output["maper_insula_points"].map(
                lambda value: "core" if value == 3 else "partial" if value > 0 else "none"
            )
            if not output["maper_insula_status"].equals(expected_status):
                raise AssertionError("insula status disagrees with exact-point count")
            if len(sensitivity) != 3 * len(output):
                raise AssertionError("sensitivity table is not 3 rows/channel")
            if set(sensitivity["location"]) != LOCATIONS:
                raise AssertionError("sensitivity locations incomplete")
            if sensitivity.duplicated(["task", "subject", "reference", "name", "location"]).any():
                raise AssertionError("duplicate sensitivity key")

            vote_columns = [
                column for column in output
                if column.endswith("_vote_fraction")
            ]
            vote_values = output[vote_columns].to_numpy(float)
            if not np.isfinite(vote_values).all():
                raise AssertionError("non-finite vote fraction")
            if not np.allclose(vote_values * 30, np.rint(vote_values * 30), atol=1e-8):
                raise AssertionError("vote fraction is not an integer multiple of 1/30")
            sphere_columns = [
                "sphere_insula_fraction", "sphere_winner_fraction_within_insula"
            ]
            sphere_values = sensitivity[sphere_columns].to_numpy(float)
            if not np.isfinite(sphere_values).all():
                raise AssertionError("non-finite sphere fraction")
            if np.any((sphere_values < 0) | (sphere_values > 1)):
                raise AssertionError("sphere fraction outside [0,1]")

            keys.extend(
                (row["task"], row["subject"], "bipolar", str(name))
                for name in output["name"]
            )
            total_rows += len(output)
        except Exception as error:  # aggregate all subjects before failing
            failed_indices.append(manifest_index)
            failures.append(f"{row['task']}/{row['subject']}: {error}")

    if len(keys) != len(set(keys)):
        failures.append("duplicate task+subject+reference+name key across outputs")
    if args.failure_manifest:
        retry = manifest.loc[failed_indices].copy()
        args.failure_manifest.parent.mkdir(parents=True, exist_ok=True)
        retry.to_csv(args.failure_manifest, sep="\t", index=False)
        print(f"Wrote retry manifest: {args.failure_manifest} rows={len(retry)}")
    if failures:
        raise SystemExit("Validation failed:\n" + "\n".join(failures))

    print(
        f"PASS combinations={len(ready)} rows={total_rows} "
        f"unique_keys={len(keys)} missing={len(manifest) - len(ready)}")
    print(manifest["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
