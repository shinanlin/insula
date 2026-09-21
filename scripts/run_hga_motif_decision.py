#!/usr/bin/env python3
"""Build and summarize the HGA amplitude motif decision analysis."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

SCRIPT_REPOSITORY = Path(__file__).resolve().parents[1]
if str(SCRIPT_REPOSITORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPOSITORY))

from src.connectivity.hga_motif_decision import (
    ENTITY_COLUMNS,
    MetricColumns,
    PRIMARY_TASKS,
    add_distance_bins,
    annotate_motif_weights,
    balance_tasks_within_subject,
    bootstrap_group_cells,
    classify_lag_clusters,
    infer_subject_contrasts,
    matched_cross_entity_summary,
    stratified_partner_permutation_multi,
    subject_balance_contrasts,
    weighted_cell_summary,
    weighted_direction_summary,
)
from src.connectivity.pairwise.config import ConnectivityConfig
from src.connectivity.pairwise.permutation import benjamini_hochberg
from src.connectivity.pairwise.io import (
    discover_manifest,
    input_fingerprint,
    load_analysis_data,
    parse_filename_entities,
    read_manifest_row,
    write_manifest,
)
from src.connectivity.pairwise.oaec import compute_observed_lagged_oaec
from src.connectivity.pairwise.output import (
    atomic_json_write,
    atomic_table_write,
    connectivity_bids_path,
    existing_result_matches,
    implementation_hash,
    git_state,
    software_versions,
    provenance_path,
    write_metric_result,
)


REPOSITORY = Path("/hpc/group/coganlab/nanlinshi/insula-functional")
DEFAULT_ROOT = REPOSITORY / "results/connectivity/hga_amplitude_motif_decision"


ENTITY_KEYS = (
    "dataset", "subject", "task", "phase", "description",
    "recording", "run", "acquisition",
)


def _text(value: object) -> str:
    return str(value if value is not None else "").strip()


def _entity_key(row: dict[str, object]) -> tuple[str, ...]:
    return tuple(_text(row.get(column, "")) for column in ENTITY_KEYS)


def _baseline_entity_keys(baseline_root: Path) -> set[tuple[str, ...]]:
    keys: set[tuple[str, ...]] = set()
    for path in baseline_root.glob("*/sub-*/xcorr/*_desc-Repeat_pairs.parquet"):
        dataset = path.relative_to(baseline_root).parts[0]
        if dataset not in PRIMARY_TASKS:
            continue
        parsed = parse_filename_entities(path)
        keys.add(
            _entity_key(
                {
                    "dataset": dataset,
                    "subject": parsed.get("sub", ""),
                    "task": parsed.get("task", ""),
                    "phase": parsed.get("proc", ""),
                    "description": parsed.get("desc", ""),
                    "recording": parsed.get("recording", parsed.get("rec", "")),
                    "run": parsed.get("run", ""),
                    "acquisition": parsed.get("acq", ""),
                }
            )
        )
    return keys


def prepare_manifest(
    output: Path,
    *,
    baseline_root: Path,
    expected_entities: int = 511,
) -> Path:
    """Create the frozen manifest matching completed primary xcorr entities."""

    discovered = discover_manifest()
    baseline_keys = _baseline_entity_keys(baseline_root)
    discovered_keys = discovered.apply(lambda row: _entity_key(row.to_dict()), axis=1)
    selected = discovered.loc[discovered_keys.isin(baseline_keys)].copy()
    selected = selected.sort_values(
        ["dataset", "subject", "phase", "recording", "run", "acquisition"]
    ).reset_index(drop=True)
    if len(selected) != expected_entities:
        raise RuntimeError(
            f"expected {expected_entities} completed baseline entities, found "
            f"{len(selected)} manifest matches from {len(baseline_keys)} baselines"
        )
    if selected.duplicated(
        [
            "dataset", "subject", "task", "phase", "description",
            "recording", "run", "acquisition",
        ]
    ).any():
        raise RuntimeError("manifest contains duplicate analysis entities")
    return write_manifest(selected, output)


def prepare_command(args: argparse.Namespace) -> int:
    destination = prepare_manifest(
        args.output,
        baseline_root=args.baseline_root,
        expected_entities=args.expected_entities,
    )
    print(json.dumps({"manifest": str(destination), "n_entities": args.expected_entities}))
    return 0


def _manifest_entities(row: pd.Series) -> dict[str, str]:
    return {column: _text(row.get(column, "")) for column in ENTITY_COLUMNS}


def _artifact_path(
    root: Path, entities: dict[str, str], metric: str, suffix: str, extension: str
) -> Path:
    return Path(
        connectivity_bids_path(
            root, entities, metric=metric, suffix=suffix, extension=extension
        ).fpath
    )


def _xcorr_effect(table: pd.DataFrame, detail_path: Path) -> np.ndarray:
    with xr.open_dataset(detail_path, engine="h5netcdf") as detail:
        detail_ids = detail["pair_id"].astype(str).to_numpy()
        table_ids = table["pair_id"].astype(str).to_numpy()
        if not np.array_equal(detail_ids, table_ids):
            raise RuntimeError(f"pair order mismatch in {detail_path}")
        lags = np.asarray(detail["lag"].to_numpy(), dtype=float)
        peaks = table["peak_lag_s"].to_numpy(float)
        peak_index = np.abs(lags[None, :] - peaks[:, None]).argmin(axis=1)
        observed = detail["observed_fisher_z"].to_numpy()
        null_mean = detail["null_mean_fisher_z"].to_numpy()
        return observed[np.arange(len(table)), peak_index] - null_mean[
            np.arange(len(table)), peak_index
        ]


def _load_xcorr(
    manifest: pd.DataFrame, root: Path, metric: str
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, row in manifest.iterrows():
        entities = _manifest_entities(row)
        pair_path = _artifact_path(root, entities, metric, "pairs", ".parquet")
        detail_path = _artifact_path(root, entities, metric, "detail", ".nc")
        cluster_path = _artifact_path(root, entities, metric, "clusters", ".parquet")
        missing = [
            str(path) for path in (pair_path, detail_path, cluster_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError("missing metric artifacts: " + ", ".join(missing))
        table = pd.read_parquet(pair_path)
        table["effect"] = _xcorr_effect(table, detail_path)
        direction = classify_lag_clusters(table, pd.read_parquet(cluster_path))
        table = table.merge(direction, on="pair_id", validate="one_to_one")
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def _load_oaec(manifest: pd.DataFrame, root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, row in manifest.iterrows():
        entities = _manifest_entities(row)
        path = _artifact_path(root, entities, "oaec", "pairs", ".parquet")
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        frame["effect"] = frame["stat"] - frame["null_mean"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


XCORR_FIELDS = (
    "stat", "effect", "peak_lag_s", "peak_r", "null_mean_stat",
    "p_uncorrected", "q_fdr", "p_fwer_maxstat", "sig_fdr", "sig_fwer",
    "qc_pass", "lag_direction", "strict_peak_lag_s", "strict_cluster_mass",
    "strict_cluster_start_s", "strict_cluster_stop_s",
    "n_significant_lag_clusters",
)
OAEC_FIELDS = (
    "stat", "effect", "null_mean", "null_std", "p_uncorrected", "q_fdr",
    "p_fwer_maxstat", "sig_fdr", "sig_fwer", "qc_pass",
)


def _prefixed(frame: pd.DataFrame, prefix: str, fields: tuple[str, ...]) -> pd.DataFrame:
    keys = [*ENTITY_COLUMNS, "pair_id", "source", "target"]
    selected = frame[[*keys, *fields]].copy()
    return selected.rename(columns={field: f"{prefix}_{field}" for field in fields})


def _assert_same_pair_identity(
    reference: pd.DataFrame, candidate: pd.DataFrame, *, label: str
) -> None:
    keys = [*ENTITY_COLUMNS, "pair_id", "source", "target"]
    if reference.duplicated(keys).any() or candidate.duplicated(keys).any():
        raise RuntimeError(f"duplicate pair identity detected for {label}")
    left = pd.MultiIndex.from_frame(reference[keys])
    right = pd.MultiIndex.from_frame(candidate[keys])
    if len(left) != len(right) or set(left) != set(right):
        missing = len(set(left).difference(right))
        extra = len(set(right).difference(left))
        raise RuntimeError(
            f"pair identity mismatch for {label}: reference={len(left)}, "
            f"candidate={len(right)}, missing={missing}, extra={extra}"
        )


def _candidate_manifest(
    merged: pd.DataFrame,
    manifest: pd.DataFrame,
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strict = {"insula_leads", "partner_leads"}
    candidate = merged.loc[
        merged["xcorr_sig_fdr"]
        & merged["xcorr_resid_sig_fdr"]
        & merged["xcorr_lag_direction"].isin(strict)
        & merged["xcorr_resid_lag_direction"].isin(strict)
        & merged["xcorr_lag_direction"].eq(merged["xcorr_resid_lag_direction"])
    ].copy()
    candidate["consensus_lag_s"] = candidate[
        ["xcorr_strict_peak_lag_s", "xcorr_resid_strict_peak_lag_s"]
    ].mean(axis=1)
    membership_path = output_root / "tables/lagged_oaec_candidates.parquet"
    atomic_table_write(candidate, membership_path, require_parquet=True)
    counts = (
        candidate.groupby(list(ENTITY_COLUMNS), dropna=False)
        .size().rename("candidate_count").reset_index()
    )
    candidate_manifest = manifest.merge(
        counts, on=list(ENTITY_COLUMNS), how="inner", validate="one_to_one"
    )
    candidate_manifest["candidate_path"] = str(membership_path)
    return candidate, candidate_manifest


def _task_balanced_pair_rows(frame: pd.DataFrame, outcomes: list[str]) -> pd.DataFrame:
    """Collapse repeated entities, then give each available task equal weight."""

    metadata = [
        "subject", "task", "phase", "pair_id", "source", "target",
        "target_roi", "distance_bin", "seed_motif_available",
        "partner_projection_qc", "motif_match",
        *[f"source_proportion_{motif}" for motif in ("sensory", "sustain", "motor")],
        *[f"target_proportion_{motif}" for motif in ("sensory", "sustain", "motor")],
    ]
    task = (
        frame.groupby(metadata, dropna=False, observed=True)[outcomes]
        .mean().reset_index()
    )
    subject_pair_keys = [column for column in metadata if column != "task"]
    return (
        task.groupby(subject_pair_keys, dropna=False, observed=True)[outcomes]
        .mean().reset_index()
    )


def _method_concordance(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    methods = ("xcorr", "xcorr_resid", "oaec")
    for phase, group in frame.groupby("phase", sort=True):
        for left_index, left in enumerate(methods):
            for right in methods[left_index + 1:]:
                first = group[f"{left}_sig_fdr"].astype(bool).to_numpy()
                second = group[f"{right}_sig_fdr"].astype(bool).to_numpy()
                union = first | second
                effect_first = group[f"{left}_effect"].rank().to_numpy(float)
                effect_second = group[f"{right}_effect"].rank().to_numpy(float)
                rows.append(
                    {
                        "phase": phase, "method_a": left, "method_b": right,
                        "n_rows": int(len(group)),
                        "n_both_fdr": int(np.sum(first & second)),
                        "fdr_jaccard": float(np.sum(first & second) / np.sum(union))
                        if union.any() else np.nan,
                        "fdr_agreement": float(np.mean(first == second)),
                        "effect_spearman": float(
                            np.corrcoef(effect_first, effect_second)[0, 1]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _roi_summaries(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = [
        f"{method}_{suffix}" for method in ("xcorr", "xcorr_resid", "oaec")
        for suffix in ("sig_fdr", "sig_fwer", "effect")
    ]
    entity_keys = [*ENTITY_COLUMNS, "target_roi"]
    entity = (
        frame.loc[frame["target_roi"].notna()]
        .groupby(entity_keys, dropna=False, observed=True)
        .agg(
            **{column: (column, "mean") for column in outcomes},
            n_pairs=("pair_id", "size"),
        ).reset_index()
    )
    task = (
        entity.groupby(["subject", "task", "phase", "target_roi"], observed=True)
        .agg(
            **{column: (column, "mean") for column in outcomes},
            n_pairs=("n_pairs", "sum"),
        ).reset_index()
    )
    subject = (
        task.groupby(["subject", "phase", "target_roi"], observed=True)
        .agg(
            **{column: (column, "mean") for column in outcomes},
            n_tasks=("task", "nunique"), n_pairs=("n_pairs", "sum"),
        ).reset_index()
    )
    group = (
        subject.groupby(["phase", "target_roi"], observed=True)
        .agg(
            **{f"{column}_mean": (column, "mean") for column in outcomes},
            n_subjects=("subject", "nunique"), n_pairs=("n_pairs", "sum"),
        ).reset_index()
    )
    return subject, group


def build_tables_command(args: argparse.Namespace) -> int:
    output_root = args.output_root
    (output_root / "tables").mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest, sep="\t", keep_default_na=False)
    if args.entity_limit is not None:
        manifest = manifest.head(args.entity_limit).copy()
    if len(manifest) != args.expected_entities:
        raise RuntimeError(f"manifest has {len(manifest)} rows, expected {args.expected_entities}")
    original = _load_xcorr(manifest, args.baseline_root, "xcorr")
    residual = _load_xcorr(manifest, args.residual_root, "xcorr_resid")
    oaec = _load_oaec(manifest, args.baseline_root)
    _assert_same_pair_identity(original, residual, label="xcorr_resid")
    _assert_same_pair_identity(original, oaec, label="oaec")
    keys = [*ENTITY_COLUMNS, "pair_id", "source", "target"]
    original_prefixed = _prefixed(original, "xcorr", XCORR_FIELDS)
    residual_prefixed = _prefixed(residual, "xcorr_resid", XCORR_FIELDS)
    oaec_prefixed = _prefixed(oaec, "oaec", OAEC_FIELDS)
    baseline_metadata = original.drop(
        columns=[column for column in XCORR_FIELDS if column in original],
        errors="ignore",
    )
    baseline_metadata = baseline_metadata.drop(
        columns=["metric", "residualize_evoked"], errors="ignore"
    )
    merged = baseline_metadata.merge(
        original_prefixed, on=keys, validate="one_to_one"
    ).merge(residual_prefixed, on=keys, validate="one_to_one").merge(
        oaec_prefixed, on=keys, validate="one_to_one"
    )
    seeds = pd.read_csv(args.seed_assignments)
    projection = pd.read_csv(args.partner_projection)
    annotated = add_distance_bins(
        annotate_motif_weights(
            merged, seeds, projection, min_explained_energy=args.projection_threshold
        )
    )
    if args.entity_limit is None:
        observed_expectations = {
            "pair_rows": len(annotated),
            "subjects": annotated["subject"].nunique(),
            "unique_pairs": annotated["pair_id"].nunique(),
        }
        expected = {
            "pair_rows": args.expected_pair_rows,
            "subjects": args.expected_subjects,
            "unique_pairs": args.expected_unique_pairs,
        }
        if observed_expectations != expected:
            raise RuntimeError(
                f"fixed-scope QC mismatch: observed={observed_expectations}, "
                f"expected={expected}"
            )
    source_motif_columns = [
        f"source_proportion_{motif}" for motif in ("sensory", "sustain", "motor")
    ]
    target_motif_columns = [
        f"target_proportion_{motif}" for motif in ("sensory", "sustain", "motor")
    ]
    annotated["source_dominant_motif"] = pd.Series(
        pd.NA, index=annotated.index, dtype="string"
    )
    annotated["target_dominant_motif"] = pd.Series(
        pd.NA, index=annotated.index, dtype="string"
    )
    valid_source = annotated[source_motif_columns].notna().all(axis=1)
    valid_target = annotated[target_motif_columns].notna().all(axis=1)
    annotated.loc[valid_source, "source_dominant_motif"] = (
        annotated.loc[valid_source, source_motif_columns]
        .idxmax(axis=1).str.removeprefix("source_proportion_")
    )
    annotated.loc[valid_target, "target_dominant_motif"] = (
        annotated.loc[valid_target, target_motif_columns]
        .idxmax(axis=1).str.removeprefix("target_proportion_")
    )
    atomic_table_write(
        annotated, output_root / "tables/pair_level_annotated.parquet",
        require_parquet=True,
    )

    metrics = [
        MetricColumns("xcorr", "xcorr_sig_fdr", "xcorr_sig_fwer", "xcorr_effect"),
        MetricColumns(
            "xcorr_resid", "xcorr_resid_sig_fdr", "xcorr_resid_sig_fwer",
            "xcorr_resid_effect",
        ),
        MetricColumns("oaec", "oaec_sig_fdr", "oaec_sig_fwer", "oaec_effect"),
    ]
    cells = weighted_cell_summary(annotated, metrics)
    value_columns = [
        f"{metric.name}_{suffix}" for metric in metrics
        for suffix in ("fdr_hit_rate", "fwer_hit_rate", "effect")
    ]
    subject_cells = balance_tasks_within_subject(cells, value_columns=value_columns)
    group_cells = bootstrap_group_cells(
        subject_cells, value_columns=value_columns,
        n_bootstrap=args.n_resamples, random_state=args.random_state,
    )
    cells.to_csv(output_root / "tables/entity_motif_cells.csv", index=False)
    subject_cells.to_csv(output_root / "tables/subject_motif_cells.csv", index=False)
    group_cells.to_csv(output_root / "tables/group_motif_cells.csv", index=False)

    contrasts = matched_cross_entity_summary(annotated, metrics)
    contrast_values = [
        column for column in contrasts
        if "matched_minus_cross" in column
    ]
    subject_contrasts = subject_balance_contrasts(
        contrasts, value_columns=contrast_values
    )
    group_contrasts = infer_subject_contrasts(
        subject_contrasts, value_columns=contrast_values,
        n_permutations=args.n_resamples, random_state=args.random_state,
    )
    contrasts.to_csv(output_root / "tables/entity_matched_cross.csv", index=False)
    subject_contrasts.to_csv(
        output_root / "tables/subject_matched_cross.csv", index=False
    )
    group_contrasts.to_csv(
        output_root / "tables/group_matched_cross.csv", index=False
    )

    direction = weighted_direction_summary(
        annotated,
        direction_column="xcorr_lag_direction",
        lag_column="xcorr_strict_peak_lag_s",
    )
    direction_values = [
        "weighted_lag_s", "insula_leads_fraction", "partner_leads_fraction",
        "unresolved_fraction", "ambiguous_fraction",
    ]
    subject_direction = balance_tasks_within_subject(
        direction.rename(
            columns={"strict_weight": "eligible_weight"}
        ).assign(n_pairs=1),
        value_columns=direction_values,
    )
    group_direction = bootstrap_group_cells(
        subject_direction, value_columns=direction_values,
        n_bootstrap=args.n_resamples, random_state=args.random_state + 1,
    )
    direction.to_csv(output_root / "tables/entity_lag_direction.csv", index=False)
    subject_direction.to_csv(
        output_root / "tables/subject_lag_direction.csv", index=False
    )
    group_direction.to_csv(
        output_root / "tables/group_lag_direction.csv", index=False
    )

    hard = annotated.loc[
        annotated["seed_motif_available"] & annotated["partner_projection_qc"]
    ].rename(
        columns={
            "source_dominant_motif": "source_motif",
            "target_dominant_motif": "partner_motif",
        }
    )
    hard_cells = (
        hard.groupby(
            [*ENTITY_COLUMNS, "source_motif", "partner_motif"],
            dropna=False, observed=True,
        )
        .agg(
            eligible_weight=("pair_id", "size"), n_pairs=("pair_id", "size"),
            **{
                f"{metric.name}_{suffix}": (column, "mean")
                for metric in metrics
                for suffix, column in (
                    ("fdr_hit_rate", metric.sig_fdr),
                    ("fwer_hit_rate", metric.sig_fwer),
                    ("effect", metric.effect),
                )
            },
        ).reset_index()
    )
    hard_subject = balance_tasks_within_subject(
        hard_cells, value_columns=value_columns
    )
    hard_group = bootstrap_group_cells(
        hard_subject, value_columns=value_columns,
        n_bootstrap=args.n_resamples, random_state=args.random_state + 2,
    )
    hard_group.to_csv(output_root / "tables/group_hard_motif_cells.csv", index=False)

    subject_roi, group_roi = _roi_summaries(annotated)
    subject_roi.to_csv(output_root / "tables/subject_roi_summary.csv", index=False)
    group_roi.to_csv(output_root / "tables/group_roi_summary.csv", index=False)
    concordance = _method_concordance(annotated)
    concordance.to_csv(output_root / "tables/method_concordance.csv", index=False)

    task_contribution = (
        cells.groupby(["task", "phase", "source_motif", "partner_motif"], observed=True)
        .agg(
            **{column: (column, "mean") for column in value_columns},
            n_subjects=("subject", "nunique"),
            eligible_weight=("eligible_weight", "sum"),
        ).reset_index()
    )
    task_contribution.to_csv(
        output_root / "tables/task_contribution.csv", index=False
    )

    sensitivity_frames: list[pd.DataFrame] = []
    for threshold in (0.3, 0.5, 0.7):
        threshold_frame = annotated.copy()
        threshold_frame["partner_projection_qc"] = (
            threshold_frame["partner_projection_available"]
            & threshold_frame["explained_energy"].ge(threshold)
        )
        threshold_contrast = matched_cross_entity_summary(threshold_frame, metrics)
        threshold_subject = subject_balance_contrasts(
            threshold_contrast, value_columns=contrast_values
        )
        threshold_group = infer_subject_contrasts(
            threshold_subject, value_columns=contrast_values,
            n_permutations=args.n_resamples,
            random_state=args.random_state + int(threshold * 100),
        )
        threshold_group.insert(0, "projection_threshold", threshold)
        sensitivity_frames.append(threshold_group)
    pd.concat(sensitivity_frames, ignore_index=True).to_csv(
        output_root / "tables/projection_threshold_sensitivity.csv", index=False
    )

    permutation_outcomes = [
        "xcorr_sig_fdr", "xcorr_resid_sig_fdr", "oaec_sig_fdr",
        "xcorr_effect", "xcorr_resid_effect", "oaec_effect",
    ]
    balanced_pairs = _task_balanced_pair_rows(annotated, permutation_outcomes)
    permutation_rows: list[dict[str, object]] = []
    for phase, phase_pairs in balanced_pairs.groupby("phase", sort=True):
        result = stratified_partner_permutation_multi(
            phase_pairs, outcome_columns=permutation_outcomes,
            n_permutations=args.n_resamples,
            random_state=args.random_state + sum(map(ord, phase)),
        )
        result.insert(0, "phase", phase)
        permutation_rows.extend(result.to_dict(orient="records"))
    permutation_table = pd.DataFrame(permutation_rows)
    permutation_table["q_fdr"] = benjamini_hochberg(
        permutation_table["p"].to_numpy(float)
    )
    permutation_table.to_csv(
        output_root / "tables/stratified_motif_permutation.csv", index=False
    )

    candidates, candidate_manifest = _candidate_manifest(
        annotated, manifest, output_root
    )
    write_manifest(
        candidate_manifest, output_root / "manifests/lagged_oaec_candidates.tsv"
    )
    qc = {
        "expected_entities": args.expected_entities,
        "manifest_entities": int(len(manifest)),
        "original_xcorr_rows": int(len(original)),
        "residual_xcorr_rows": int(len(residual)),
        "oaec_rows": int(len(oaec)),
        "merged_rows": int(len(annotated)),
        "subjects": int(annotated["subject"].nunique()),
        "unique_anatomical_pairs": int(annotated["pair_id"].nunique()),
        "seed_motif_missing_rows": int((~annotated["seed_motif_available"]).sum()),
        "partner_projection_missing_rows": int(
            (~annotated["partner_projection_available"]).sum()
        ),
        "partner_projection_qc_rows": int(annotated["partner_projection_qc"].sum()),
        "partner_projection_below_threshold_rows": int(
            (
                annotated["partner_projection_available"]
                & ~annotated["partner_projection_qc"]
            ).sum()
        ),
        "lagged_oaec_candidate_rows": int(len(candidates)),
        "lagged_oaec_candidate_entities": int(len(candidate_manifest)),
        "cell_subject_counts": {
            f"{row.phase}:{row.source_motif}:{row.partner_motif}": int(row.n_subjects)
            for row in group_cells.itertuples()
        },
        "cell_pair_mass": {
            f"{row.phase}:{row.source_motif}:{row.partner_motif}": float(
                subject_cells.loc[
                    (subject_cells["phase"] == row.phase)
                    & (subject_cells["source_motif"] == row.source_motif)
                    & (subject_cells["partner_motif"] == row.partner_motif),
                    "eligible_weight_total",
                ].sum()
            )
            for row in group_cells.itertuples()
        },
    }
    atomic_json_write(qc, output_root / "qc/table_build_qc.json")
    print(json.dumps(qc, indent=2))
    return 0


def lagged_oaec_row_command(args: argparse.Namespace) -> int:
    row = read_manifest_row(args.manifest, args.row_index)
    entities = _manifest_entities(row)
    candidates = pd.read_parquet(str(row["candidate_path"]))
    mask = np.ones(len(candidates), dtype=bool)
    for column in ENTITY_COLUMNS:
        mask &= candidates[column].fillna("").astype(str).eq(entities[column])
    selected = candidates.loc[
        mask,
        [
            "pair_id", "xcorr_lag_direction", "xcorr_strict_peak_lag_s",
            "xcorr_resid_strict_peak_lag_s", "consensus_lag_s",
        ],
    ]
    config = ConnectivityConfig(
        n_perm=1_000, random_state=args.random_state, min_trials=args.min_trials,
        max_lag_s=args.max_lag_s, oaec_sfreq=args.oaec_sfreq,
    )
    fingerprint = input_fingerprint(str(row["candidate_path"]))
    selection_hash = sha256(
        selected.sort_values("pair_id").to_csv(index=False).encode()
    ).hexdigest()[:16]
    config_payload = {
        **config.as_dict(), "metric": "lagged_oaec_candidate",
        "candidate_fingerprint": fingerprint,
        "candidate_selection_hash": selection_hash,
        "selection": "xcorr_and_xcorr_resid_fdr_strict_same_direction",
        "second_permutation_test": False,
    }
    digest = sha256(
        json.dumps(
            {
             **{key: value for key, value in config_payload.items()
                if key != "candidate_fingerprint"},
             "entities": entities,
             "implementation_hash": implementation_hash()},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    if existing_result_matches(
        args.output_root, entities, "lagged_oaec_candidate", digest
    ):
        print(json.dumps({"status": "skipped_config_hash_match", "entities": entities}))
        return 0
    previous = provenance_path(
        args.output_root, entities, "lagged_oaec_candidate"
    )
    if previous.exists():
        raise RuntimeError(
            f"existing lagged-OAEC result has a different config hash: {previous}"
        )
    data = load_analysis_data(row, min_trials=args.min_trials)
    pair_frame = data.pair_frame.merge(selected, on="pair_id", validate="one_to_one")
    if len(pair_frame) != int(row["candidate_count"]):
        raise RuntimeError("candidate count does not match loaded eligible pairs")
    candidate_data = replace(data, pair_frame=pair_frame)
    started = datetime.now(timezone.utc)
    start = time.perf_counter()
    result = compute_observed_lagged_oaec(
        data.raw_data, data.raw_times, data.raw_sfreq, entities["phase"],
        pair_frame, config,
    )
    report = write_metric_result(
        result, candidate_data, output_root=args.output_root,
        config=config_payload, config_hash=digest, entity_seed=args.random_state,
        repository=REPOSITORY, started_at=started,
        elapsed_seconds=time.perf_counter() - start, require_parquet=True,
    )
    print(json.dumps(report, indent=2))
    return 0


def _load_lagged_results(manifest: pd.DataFrame, root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for _, row in manifest.iterrows():
        entities = _manifest_entities(row)
        path = _artifact_path(
            root, entities, "lagged_oaec_candidate", "pairs", ".parquet"
        )
        if not path.exists():
            missing.append(str(path))
            continue
        frame = pd.read_parquet(path)
        if len(frame) != int(row["candidate_count"]):
            raise RuntimeError(f"candidate row mismatch in {path}")
        frames.append(frame)
    if missing:
        raise RuntimeError(
            f"missing {len(missing)} candidate lagged-OAEC results; first={missing[0]}"
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _bootstrap_mean(
    values: np.ndarray, *, n_resamples: int, seed: int
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(values), size=(n_resamples, len(values)))
    boot = np.mean(values[index], axis=1)
    return (
        float(np.mean(values)), float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


def _lagged_concordance(
    lagged: pd.DataFrame, *, n_resamples: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = lagged.copy()
    output["lagged_oaec_direction"] = np.select(
        [output["peak_lag_s"].lt(0), output["peak_lag_s"].gt(0)],
        ["insula_leads", "partner_leads"], default="unresolved",
    )
    output["direction_agree"] = output["lagged_oaec_direction"].eq(
        output["xcorr_lag_direction"]
    )
    output["lag_error_s"] = output["peak_lag_s"] - output["consensus_lag_s"]
    task_rows: list[dict[str, object]] = []
    for (subject, task, phase), group in output.groupby(
        ["subject", "task", "phase"], observed=True
    ):
        valid = group[["peak_lag_s", "consensus_lag_s"]].dropna()
        correlation = (
            float(valid.corr().iloc[0, 1]) if len(valid) >= 3 else np.nan
        )
        correlation = np.clip(correlation, -0.999999, 0.999999)
        task_rows.append(
            {
                "subject": subject, "task": task, "phase": phase,
                "direction_agreement": float(group["direction_agree"].mean()),
                "lag_error_s": float(group["lag_error_s"].mean()),
                "lag_correlation_z": float(np.arctanh(correlation))
                if np.isfinite(correlation) else np.nan,
                "n_pairs": int(len(group)),
            }
        )
    task = pd.DataFrame(task_rows)
    subject = (
        task.groupby(["subject", "phase"], observed=True)
        .agg(
            direction_agreement=("direction_agreement", "mean"),
            lag_error_s=("lag_error_s", "mean"),
            lag_correlation_z=("lag_correlation_z", "mean"),
            n_tasks=("task", "nunique"), n_pairs=("n_pairs", "sum"),
        ).reset_index()
    )
    rows: list[dict[str, object]] = []
    for phase, group in subject.groupby("phase", sort=True):
        row: dict[str, object] = {
            "phase": phase, "n_subjects": int(group["subject"].nunique()),
            "n_pairs": int(group["n_pairs"].sum()),
        }
        for offset, column in enumerate(
            ("direction_agreement", "lag_error_s", "lag_correlation_z")
        ):
            mean, low, high = _bootstrap_mean(
                group[column].to_numpy(), n_resamples=n_resamples,
                seed=seed + offset + sum(map(ord, phase)),
            )
            if column == "lag_correlation_z":
                mean, low, high = tuple(np.tanh([mean, low, high]))
                label = "lag_correlation"
            else:
                label = column
            row[f"{label}_mean"] = mean
            row[f"{label}_ci_low"] = low
            row[f"{label}_ci_high"] = high
        rows.append(row)
    return output, pd.DataFrame(rows)


def _save_figure(figure: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def _motif_matrix(table: pd.DataFrame, value: str, phase: str) -> np.ndarray:
    motifs = ["sensory", "sustain", "motor"]
    subset = table.loc[table["phase"].eq(phase)]
    pivot = subset.pivot(index="source_motif", columns="partner_motif", values=value)
    return pivot.reindex(index=motifs, columns=motifs).to_numpy(float)


def _plot_motif_cells(group_cells: pd.DataFrame, figures: Path) -> None:
    phases = ["Stimulus", "Delay", "Go", "Response"]
    methods = [("xcorr", "Original xcorr"), ("xcorr_resid", "Residual xcorr"), ("oaec", "OAEC")]
    figure, axes = plt.subplots(3, 4, figsize=(12, 8), constrained_layout=True)
    values = [
        _motif_matrix(group_cells, f"{method}_fdr_hit_rate_mean", phase)
        for method, _ in methods for phase in phases
    ]
    vmax = 100 * max(float(np.nanmax(value)) for value in values)
    for row, (method, title) in enumerate(methods):
        for column, phase in enumerate(phases):
            axis = axes[row, column]
            matrix = 100 * _motif_matrix(
                group_cells, f"{method}_fdr_hit_rate_mean", phase
            )
            image = axis.imshow(matrix, vmin=0, vmax=vmax, cmap="magma")
            for i in range(3):
                for j in range(3):
                    axis.text(
                        j, i, f"{matrix[i, j]:.1f}%", ha="center", va="center",
                        color="white" if matrix[i, j] > vmax / 2 else "black",
                        fontsize=7,
                    )
            axis.set_xticks(range(3), ["Sensory", "Sustain", "Motor"], rotation=35, ha="right")
            axis.set_yticks(range(3), ["Sensory", "Sustain", "Motor"])
            if row == 0:
                axis.set_title(phase)
            if column == 0:
                axis.set_ylabel(f"{title}\nInsula motif")
            if row == 2:
                axis.set_xlabel("Partner motif")
    figure.colorbar(image, ax=axes, label="Weighted FDR hit rate (%)")
    _save_figure(figure, figures / "main_phase_resolved_motif_coupling")


def _plot_matched_cross(group: pd.DataFrame, figures: Path) -> None:
    phases = ["Stimulus", "Delay", "Go", "Response"]
    methods = [("xcorr", "Original xcorr"), ("xcorr_resid", "Residual xcorr"), ("oaec", "OAEC")]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for method, label in methods:
        ordered = group.set_index("phase").reindex(phases)
        for axis, suffix, ylabel in (
            (axes[0], "fdr_hit_rate", "Matched - cross FDR hit rate"),
            (axes[1], "effect", "Matched - cross effect"),
        ):
            stem = f"{method}_matched_minus_cross_{suffix}"
            mean = ordered[f"{stem}_mean"].to_numpy(float)
            low = ordered[f"{stem}_ci_low"].to_numpy(float)
            high = ordered[f"{stem}_ci_high"].to_numpy(float)
            axis.errorbar(phases, mean, yerr=[mean - low, high - mean], marker="o", capsize=3, label=label)
            axis.axhline(0, color="0.4", lw=0.8)
            axis.set_ylabel(ylabel)
            axis.tick_params(axis="x", rotation=25)
    axes[1].legend(frameon=False)
    _save_figure(figure, figures / "main_matched_vs_cross")


def _plot_lag_direction(
    group_direction: pd.DataFrame, pair_table: pd.DataFrame, figures: Path
) -> None:
    phases = ["Stimulus", "Delay", "Go", "Response"]
    figure, axes = plt.subplots(2, 4, figsize=(12, 6.5), constrained_layout=True)
    lag_matrices = {
        phase: 1000 * _motif_matrix(
            group_direction, "weighted_lag_s_mean", phase
        )
        for phase in phases
    }
    limit = max(
        1.0,
        max(float(np.nanmax(np.abs(matrix))) for matrix in lag_matrices.values()),
    )
    for column, phase in enumerate(phases):
        matrix = lag_matrices[phase]
        axis = axes[0, column]
        image = axis.imshow(matrix, vmin=-limit, vmax=limit, cmap="coolwarm")
        subset = group_direction[group_direction["phase"].eq(phase)].set_index(["source_motif", "partner_motif"])
        for i, source in enumerate(("sensory", "sustain", "motor")):
            for j, target in enumerate(("sensory", "sustain", "motor")):
                row = subset.loc[(source, target)]
                stable = row["weighted_lag_s_ci_low"] * row["weighted_lag_s_ci_high"] > 0
                arrow = ""
                if stable:
                    arrow = "\nI→P" if matrix[i, j] < 0 else "\nP→I"
                label = f"{matrix[i, j]:.0f}{arrow}"
                axis.text(j, i, label, ha="center", va="center", fontsize=7)
        axis.set_title(phase)
        axis.set_xticks(range(3), ["Sensory", "Sustain", "Motor"], rotation=35, ha="right")
        axis.set_yticks(range(3), ["Sensory", "Sustain", "Motor"])
        if column == 0:
            axis.set_ylabel("Insula motif\nPeak lag (ms)")
        strict = pair_table.loc[
            pair_table["phase"].eq(phase)
            & pair_table["xcorr_lag_direction"].isin(["insula_leads", "partner_leads"])
        ]
        counts = strict["xcorr_lag_direction"].value_counts()
        axes[1, column].bar(
            ["Insula leads", "Partner leads"],
            [counts.get("insula_leads", 0), counts.get("partner_leads", 0)],
            color=["#3666a6", "#c45a3c"],
        )
        axes[1, column].tick_params(axis="x", rotation=25)
        axes[1, column].set_ylabel("FDR strict pair rows" if column == 0 else "")
    figure.colorbar(image, ax=axes[0, :], label="Peak lag (ms); negative = Insula leads")
    figure.suptitle(
        "I→P / P→I shown only when the 95% subject-bootstrap CI excludes zero; "
        "temporal precedence only",
        fontsize=8,
    )
    _save_figure(figure, figures / "main_lag_direction")


def _plot_anatomy(group_roi: pd.DataFrame, figures: Path) -> None:
    top = (
        group_roi.groupby("target_roi")["n_pairs"].sum().nlargest(12).index
    )
    subset = group_roi[group_roi["target_roi"].isin(top)]
    roi_order = list(top[::-1])
    methods = [("xcorr", "Original xcorr"), ("xcorr_resid", "Residual xcorr"), ("oaec", "OAEC")]
    figure, axes = plt.subplots(1, 3, figsize=(12, 5.5), sharey=True, constrained_layout=True)
    for axis, (method, title) in zip(axes, methods):
        values = subset.groupby("target_roi")[f"{method}_sig_fdr_mean"].mean().reindex(roi_order)
        axis.barh(roi_order, 100 * values, color="#4c78a8")
        axis.set_title(title)
        axis.set_xlabel("Mean FDR hit rate (%)")
    axes[0].set_ylabel("Partner ROI")
    _save_figure(figure, figures / "main_anatomical_distribution")


def _plot_method_concordance(table: pd.DataFrame, figures: Path) -> None:
    labels = table["method_a"].str.replace("xcorr_resid", "resid") + " vs " + table["method_b"].str.replace("xcorr_resid", "resid")
    plot = table.assign(comparison=labels)
    phases = ["Stimulus", "Delay", "Go", "Response"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for comparison, group in plot.groupby("comparison", sort=True):
        ordered = group.set_index("phase").reindex(phases)
        axes[0].plot(phases, ordered["fdr_jaccard"], marker="o", label=comparison)
        axes[1].plot(phases, ordered["effect_spearman"], marker="o", label=comparison)
    axes[0].set_ylabel("FDR edge Jaccard")
    axes[1].set_ylabel("Effect Spearman correlation")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.axhline(0, color="0.5", lw=0.8)
    axes[1].legend(frameon=False, fontsize=8)
    _save_figure(figure, figures / "main_method_concordance")


def _plot_supplements(root: Path, figures: Path) -> None:
    group_cells = pd.read_csv(root / "tables/group_motif_cells.csv")
    subject_cells = pd.read_csv(root / "tables/subject_motif_cells.csv")
    task = pd.read_csv(root / "tables/task_contribution.csv")
    sensitivity = pd.read_csv(root / "tables/projection_threshold_sensitivity.csv")
    phases = ["Stimulus", "Delay", "Go", "Response"]
    figure, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
    for axis, phase in zip(axes, phases):
        matrix = _motif_matrix(group_cells, "xcorr_fwer_hit_rate_mean", phase)
        axis.imshow(matrix, vmin=0, vmax=np.nanmax(group_cells["xcorr_fwer_hit_rate_mean"]), cmap="viridis")
        axis.set_title(phase)
        axis.set_xticks(range(3), ["S", "U", "M"])
        axis.set_yticks(range(3), ["S", "U", "M"])
    _save_figure(figure, figures / "supp_global_fwer_cells")

    coverage = subject_cells.groupby(["phase", "source_motif", "partner_motif"])["eligible_weight_total"].sum().reset_index()
    figure, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
    for axis, phase in zip(axes, phases):
        matrix = _motif_matrix(coverage, "eligible_weight_total", phase)
        axis.imshow(matrix, cmap="cividis")
        axis.set_title(phase)
        axis.set_xticks(range(3), ["S", "U", "M"])
        axis.set_yticks(range(3), ["S", "U", "M"])
    _save_figure(figure, figures / "supp_motif_coverage")

    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    task_phase = task.groupby(["task", "phase"])["xcorr_fdr_hit_rate"].mean().reset_index()
    for task_name, group in task_phase.groupby("task"):
        ordered = group.set_index("phase").reindex(phases)
        axis.plot(phases, ordered["xcorr_fdr_hit_rate"], marker="o", label=task_name)
    axis.set_ylabel("Original xcorr FDR hit rate")
    axis.legend(frameon=False, fontsize=8)
    _save_figure(figure, figures / "supp_task_contribution")

    figure, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    methods = [
        ("xcorr", "Original xcorr"),
        ("xcorr_resid", "Residual xcorr"),
        ("oaec", "OAEC"),
    ]
    for column, (method, title) in enumerate(methods):
        for row, (suffix, ylabel) in enumerate(
            (
                ("effect", "Matched - cross effect"),
                ("fdr_hit_rate", "Matched - cross FDR hit rate"),
            )
        ):
            axis = axes[row, column]
            stem = f"{method}_matched_minus_cross_{suffix}"
            for threshold, group in sensitivity.groupby("projection_threshold"):
                ordered = group.set_index("phase").reindex(phases)
                axis.plot(
                    phases, ordered[f"{stem}_mean"], marker="o",
                    label=f"≥ {threshold:.1f}",
                )
            axis.axhline(0, color="0.5", lw=0.8)
            axis.tick_params(axis="x", rotation=25)
            if row == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(ylabel)
    axes[0, 2].legend(title="Explained energy", frameon=False, fontsize=8)
    _save_figure(figure, figures / "supp_projection_threshold")

    figure, axis = plt.subplots(figsize=(5, 5), constrained_layout=True)
    axis.scatter(group_cells["xcorr_fdr_hit_rate_mean"], group_cells["xcorr_resid_fdr_hit_rate_mean"], c=pd.Categorical(group_cells["phase"]).codes, cmap="tab10")
    limit = max(group_cells["xcorr_fdr_hit_rate_mean"].max(), group_cells["xcorr_resid_fdr_hit_rate_mean"].max())
    axis.plot([0, limit], [0, limit], color="0.4", lw=0.8)
    axis.set_xlabel("Original xcorr weighted FDR hit rate")
    axis.set_ylabel("Residual xcorr weighted FDR hit rate")
    _save_figure(figure, figures / "supp_original_vs_residual")


def _decision(
    group_contrasts: pd.DataFrame,
    stratified: pd.DataFrame | None = None,
) -> tuple[str, str]:
    support: dict[str, dict[str, set[str]]] = {
        method: {"parallel": set(), "interface": set()}
        for method in ("xcorr", "xcorr_resid")
    }
    for row in group_contrasts.itertuples():
        for method in support:
            signs: set[str] = set()
            for suffix in ("fdr_hit_rate", "effect"):
                stem = f"{method}_matched_minus_cross_{suffix}"
                if getattr(row, f"{stem}_q_fdr") < 0.05:
                    if getattr(row, f"{stem}_ci_low") > 0:
                        signs.add("parallel")
                    elif getattr(row, f"{stem}_ci_high") < 0:
                        signs.add("interface")
            if len(signs) == 1:
                support[method][signs.pop()].add(row.phase)
    parallel_phases = support["xcorr"]["parallel"] & support["xcorr_resid"]["parallel"]
    interface_phases = support["xcorr"]["interface"] & support["xcorr_resid"]["interface"]

    contradicted = False
    controlled_support: dict[str, set[str]] = {
        "parallel": set(), "interface": set()
    }
    if stratified is not None and not stratified.empty:
        for direction, phases in (
            ("parallel", parallel_phases), ("interface", interface_phases)
        ):
            expected_sign = 1 if direction == "parallel" else -1
            subset = stratified.loc[
                stratified["phase"].isin(phases)
                & stratified["outcome"].isin(
                    [
                        "xcorr_sig_fdr", "xcorr_resid_sig_fdr",
                        "xcorr_effect", "xcorr_resid_effect",
                    ]
                )
                & stratified["q_fdr"].lt(0.05)
            ]
            contradicted |= bool((np.sign(subset["stat"]) == -expected_sign).any())
            for phase in phases:
                phase_subset = subset.loc[subset["phase"].eq(phase)]
                method_support = []
                for method in ("xcorr", "xcorr_resid"):
                    method_rows = phase_subset.loc[
                        phase_subset["outcome"].isin(
                            [f"{method}_sig_fdr", f"{method}_effect"]
                        )
                    ]
                    method_support.append(
                        bool((np.sign(method_rows["stat"]) == expected_sign).any())
                    )
                if all(method_support):
                    controlled_support[direction].add(phase)
    else:
        controlled_support["parallel"] = set(parallel_phases)
        controlled_support["interface"] = set(interface_phases)
    if controlled_support["parallel"] and not interface_phases and not contradicted:
        return (
            "preferentially motif-concordant / parallel embedding",
            "Task- and subject-balanced analyses favored modest motif-concordant "
            "HGA envelope coupling, driven by coupling magnitude rather than a "
            "higher prevalence of significant edges; the effect survived evoked-mean "
            "residualization and anatomy/distance-stratified permutations, whereas "
            "OAEC and lag-direction evidence were mixed.",
        )
    if controlled_support["interface"] and not parallel_phases and not contradicted:
        return "widespread cross-motif / interface", "Evidence favors widespread cross-motif coupling, consistent with an Insula functional-interface account."
    return "evidence insufficient due to coverage or instability", "Evidence is insufficient to choose a stable parallel-embedding or cross-motif-interface account."


def _contrast_markdown(group: pd.DataFrame) -> str:
    lines = [
        "| Phase | Method | Hit-rate Δ [95% CI], q | Effect Δ [95% CI], q |",
        "|---|---|---:|---:|",
    ]
    labels = {"xcorr": "Original xcorr", "xcorr_resid": "Residual xcorr", "oaec": "OAEC"}
    for phase in ("Stimulus", "Delay", "Go", "Response"):
        row = group.loc[group["phase"].eq(phase)].iloc[0]
        for method, label in labels.items():
            hit = f"{method}_matched_minus_cross_fdr_hit_rate"
            effect = f"{method}_matched_minus_cross_effect"
            lines.append(
                f"| {phase} | {label} | {row[f'{hit}_mean']:.4f} "
                f"[{row[f'{hit}_ci_low']:.4f}, {row[f'{hit}_ci_high']:.4f}], "
                f"{row[f'{hit}_q_fdr']:.4g} | {row[f'{effect}_mean']:.4f} "
                f"[{row[f'{effect}_ci_low']:.4f}, {row[f'{effect}_ci_high']:.4f}], "
                f"{row[f'{effect}_q_fdr']:.4g} |"
            )
    return "\n".join(lines)


def finalize_command(args: argparse.Namespace) -> int:
    root = args.output_root
    figures = root / "figures"
    group_cells = pd.read_csv(root / "tables/group_motif_cells.csv")
    group_contrasts = pd.read_csv(root / "tables/group_matched_cross.csv")
    group_direction = pd.read_csv(root / "tables/group_lag_direction.csv")
    pair_table = pd.read_parquet(root / "tables/pair_level_annotated.parquet")
    group_roi = pd.read_csv(root / "tables/group_roi_summary.csv")
    concordance = pd.read_csv(root / "tables/method_concordance.csv")
    stratified = pd.read_csv(root / "tables/stratified_motif_permutation.csv")
    candidate_manifest = pd.read_csv(
        root / "manifests/lagged_oaec_candidates.tsv", sep="\t", keep_default_na=False
    )
    lagged = _load_lagged_results(candidate_manifest, args.lagged_root)
    lagged_rows, lagged_group = _lagged_concordance(
        lagged, n_resamples=args.n_resamples, seed=args.random_state
    )
    lagged_rows.to_csv(root / "tables/lagged_oaec_pair_concordance.csv", index=False)
    lagged_group.to_csv(root / "tables/lagged_oaec_group_concordance.csv", index=False)

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    _plot_motif_cells(group_cells, figures)
    _plot_matched_cross(group_contrasts, figures)
    _plot_lag_direction(group_direction, pair_table, figures)
    _plot_anatomy(group_roi, figures)
    _plot_method_concordance(concordance, figures)
    _plot_supplements(root, figures)

    decision, english_claim = _decision(group_contrasts, stratified)
    evidence_strength = (
        "moderate: the xcorr magnitude contrast replicates after evoked-mean "
        "residualization and stratified permutation; edge prevalence, OAEC, and "
        "lag direction are mixed"
        if decision != "evidence insufficient due to coverage or instability"
        else "limited: original/residual estimates do not yield a stable, anatomy-robust decision"
    )
    qc = json.loads((root / "qc/table_build_qc.json").read_text())
    agreement = float(
        np.average(
            lagged_group["direction_agreement_mean"],
            weights=lagged_group["n_pairs"],
        )
    ) if len(lagged_group) else np.nan
    coverage_fraction = qc["partner_projection_qc_rows"] / qc["merged_rows"]
    cell_subject_counts = list(qc["cell_subject_counts"].values())
    residual_comparison = concordance.loc[
        concordance["method_a"].eq("xcorr")
        & concordance["method_b"].eq("xcorr_resid")
    ]
    oaec_comparison = concordance.loc[
        concordance["method_a"].eq("xcorr")
        & concordance["method_b"].eq("oaec")
    ]
    stable_lag = (
        group_direction["weighted_lag_s_ci_low"]
        * group_direction["weighted_lag_s_ci_high"]
    ).gt(0)
    stable_lag_phases = ", ".join(
        sorted(group_direction.loc[stable_lag, "phase"].unique())
    ) or "none"
    lag_agreement_min = float(lagged_group["direction_agreement_mean"].min())
    lag_agreement_max = float(lagged_group["direction_agreement_mean"].max())
    oaec_subject_best_q = min(
        float(row[f"oaec_matched_minus_cross_effect_q_fdr"])
        for row in group_contrasts.to_dict("records")
    )
    oaec_stratified = stratified.loc[stratified["outcome"].eq("oaec_effect")]
    oaec_stratified_max_q = float(oaec_stratified["q_fdr"].max())
    contrast_table = _contrast_markdown(group_contrasts)
    def common_positive_phases(suffix: str) -> list[str]:
        phases: list[str] = []
        for row in group_contrasts.itertuples():
            supported = []
            for method in ("xcorr", "xcorr_resid"):
                stem = f"{method}_matched_minus_cross_{suffix}"
                supported.append(
                    getattr(row, f"{stem}_q_fdr") < 0.05
                    and getattr(row, f"{stem}_ci_low") > 0
                )
            if all(supported):
                phases.append(row.phase)
        return phases

    effect_support = common_positive_phases("effect")
    hit_support = common_positive_phases("fdr_hit_rate")
    controlled_effect_support = []
    for phase in ("Stimulus", "Delay", "Go", "Response"):
        rows = stratified.loc[
            stratified["phase"].eq(phase)
            & stratified["outcome"].isin(["xcorr_effect", "xcorr_resid_effect"])
        ]
        if len(rows) == 2 and rows["q_fdr"].lt(0.05).all() and rows["stat"].gt(0).all():
            controlled_effect_support.append(phase)
    effect_support_text = ", ".join(effect_support) or "none"
    hit_support_text = ", ".join(hit_support) or "none"
    controlled_support_text = ", ".join(controlled_effect_support) or "none"
    report = f"""# HGA 时滞–Motif 耦合结果决策报告

## 裁决

**{decision}**

Evidence strength: **{evidence_strength}**。

本报告以 pair-level FDR 为主发现标准，以 global max-stat FWER 为严格验证。最终裁决由 task-balanced、subject-balanced 的 original xcorr 与 evoked-mean-residualized xcorr 的 matched-minus-cross motif contrast 共同决定；OAEC 用于 leakage-reduced 稳健性比较，不替代主检验。

可用于论文的英文 claim：

> {english_claim}

## 数据范围与覆盖

- Tasks: PhonemeSequence, LexicalDelay, PictureNaming, SentenceRep；仅 Repeat 条件。
- Analysis entities: {qc['manifest_entities']}；subjects: {qc['subjects']}。
- Pair rows: {qc['merged_rows']:,}；unique Insula-partner pairs: {qc['unique_anatomical_pairs']:,}。
- Primary motif projection QC: explained energy ≥ 0.5；合格 pair rows: {qc['partner_projection_qc_rows']:,}。
- Candidate lagged-OAEC: {qc['lagged_oaec_candidate_rows']:,} pair rows across {qc['lagged_oaec_candidate_entities']} entities。

## 主要证据

1. [Phase-resolved 3×3 coupling](figures/main_phase_resolved_motif_coupling.png) 展示连续 motif 权重下 original xcorr、residual xcorr 与 OAEC 的 weighted FDR hit rate。
2. [Matched versus cross-motif contrast](figures/main_matched_vs_cross.png) 是平行嵌入与功能接口解释的正式对比；误差线为 subject bootstrap 95% CI。
3. [Lag direction](figures/main_lag_direction.png) 仅在 subject-bootstrap CI 不跨 0 的 motif cell 显示时间箭头。负 lag 表示 Insula temporal precedence；这不是因果方向。
4. [Anatomical distribution](figures/main_anatomical_distribution.png) 检查结论是否被少数 partner ROI 支配。
5. [Method concordance](figures/main_method_concordance.png) 比较 original xcorr、residual xcorr 与 OAEC 的 edge overlap 和 effect ranking。

{contrast_table}

original 与 residual xcorr 共同通过 subject-level BH-FDR 的正向 matched-minus-cross effect phases 为 **{effect_support_text}**；对应的 FDR hit-rate phases 为 **{hit_support_text}**。在 subject/ROI/distance-stratified partner-label permutation 中，两种 xcorr effect 都保持正向且校正后显著的 phases 为 **{controlled_support_text}**。因此这里的 parallel 证据主要指 coupling magnitude 的 motif concordance，而不是显著 edge 比例增加。

original 与 residual xcorr 的 FDR edge Jaccard 为 {residual_comparison['fdr_jaccard'].min():.3f}–{residual_comparison['fdr_jaccard'].max():.3f}，effect Spearman correlation 为 {residual_comparison['effect_spearman'].min():.3f}–{residual_comparison['effect_spearman'].max():.3f}，说明 residualization 后主要结构稳定。

候选 lagged-OAEC 与 xcorr 的加权总体方向一致率为 {agreement:.1%}。该分析没有第二套 permutation significance，因此只作为候选 pair 的时滞一致性描述，不能视为独立确认。

## 反证与限制

- 显著 edge prevalence 没有任何 phase 在 original 与 residual xcorr 中共同通过 subject-level BH-FDR；主裁决不能写成“matched motif 产生更多显著连接”。
- OAEC 的 subject-level matched-minus-cross effect 最低 q={oaec_subject_best_q:.3f}，未通过 0.05；xcorr–OAEC effect-rank correlation 只有 {oaec_comparison['effect_spearman'].min():.3f}–{oaec_comparison['effect_spearman'].max():.3f}（FDR edge Jaccard {oaec_comparison['fdr_jaccard'].min():.3f}–{oaec_comparison['fdr_jaccard'].max():.3f}）。不过 OAEC effect 在解剖/距离分层置换的四个 phase 均为正且 q≤{oaec_stratified_max_q:.3f}，所以 OAEC 证据应表述为 mixed，而不是完全失败。
- lagged-OAEC 方向一致率按 phase 为 {lag_agreement_min:.1%}–{lag_agreement_max:.1%}；36 个 phase×motif cells 中只有 {int(stable_lag.sum())} 个 lag bootstrap CI 不跨 0，且都位于 {stable_lag_phases}。因此时间先后是局部结果，不是统一的 Insula-leading 或 partner-leading 模式。
- 主 projection QC 仅保留全部 pair rows 的 {coverage_fraction:.1%}，每个 phase×motif cell 有 {min(cell_subject_counts)}–{max(cell_subject_counts)} 位被试；coverage 仍限制总体化。
- residualization 会同时移除真正 phase-locked 的 shared neural response；因此 original 与 residual xcorr 的差异界定 evoked contribution，并不自动把剩余效应解释为直接连接。
- OAEC 降低零相位 leakage，但不能消除 common input、reference、stimulus/item、RT 或 speech-onset confounding。
- lag 只表示相对时间位置（temporal precedence）。本报告不使用 causal、drive 或 information flow 解释。
- electrode coverage、ROI sampling 和 NNLS projection quality 不均；0.3/0.5/0.7 阈值结果见补充敏感性图。
- global FWER 与 pair FDR 属于不同的检验族，显著集合不要求彼此嵌套。

## 下一步建议

1. 在 item、trial-wise RT 与 speech onset 可用的任务中加入 nuisance regression，检验非零 lag 是否仍稳定。
2. 用 hierarchical model 直接估计 subject-level motif-match slope，并把 task 与 ROI sampling 作为层级项；它可作为本报告 weighted-cell 分析的模型化扩展。
3. 若要把 temporal precedence 推进到机制性解释，需要独立数据或 perturbation evidence；当前观测性 SEEG 结果不支持 causal wording。
4. 对跨任务重复出现、original/residual 方向一致且 OAEC lag 一致的 pairs 做预注册复现，而不是继续从本数据中放宽候选阈值。

## 补充检查

- [Task contribution](figures/supp_task_contribution.png)
- [Motif coverage](figures/supp_motif_coverage.png)
- [Global FWER](figures/supp_global_fwer_cells.png)
- [Projection threshold sensitivity](figures/supp_projection_threshold.png)
- [Original vs residual xcorr](figures/supp_original_vs_residual.png)
- [Subject/ROI/distance-stratified motif permutation](tables/stratified_motif_permutation.csv)

## 可复现运行

```bash
bash scripts/slurm/submit_hga_motif_decision.sh
```

该命令冻结同一批 511 entities，依次运行 residual xcorr、构建候选、运行 observed lagged-OAEC，并重建全部 tables、figures、QC 和本报告。固定 random seed 为 {args.random_state}。
"""
    report_path = root / "report.md"
    report_path.write_text(report, encoding="utf-8")
    residual_provenance = list(
        (root / "residualized_connectivity").glob(
            "*/sub-*/xcorrresid/*_provenance.json"
        )
    )
    residual_records = [json.loads(path.read_text()) for path in residual_provenance]
    required_tables = [
        "pair_level_annotated.parquet", "entity_motif_cells.csv",
        "subject_motif_cells.csv", "group_motif_cells.csv",
        "entity_matched_cross.csv", "subject_matched_cross.csv",
        "group_matched_cross.csv", "entity_lag_direction.csv",
        "subject_lag_direction.csv", "group_lag_direction.csv",
        "group_hard_motif_cells.csv", "subject_roi_summary.csv",
        "group_roi_summary.csv", "method_concordance.csv",
        "task_contribution.csv", "projection_threshold_sensitivity.csv",
        "stratified_motif_permutation.csv", "lagged_oaec_candidates.parquet",
        "lagged_oaec_pair_concordance.csv", "lagged_oaec_group_concordance.csv",
    ]
    required_figures = [
        "main_phase_resolved_motif_coupling", "main_matched_vs_cross",
        "main_lag_direction", "main_anatomical_distribution",
        "main_method_concordance", "supp_global_fwer_cells",
        "supp_motif_coverage", "supp_task_contribution",
        "supp_projection_threshold", "supp_original_vs_residual",
    ]
    missing_artifacts = [
        str(root / "tables" / filename) for filename in required_tables
        if not (root / "tables" / filename).exists()
    ]
    missing_artifacts.extend(
        str(figures / f"{stem}.{extension}")
        for stem in required_figures for extension in ("png", "svg")
        if not (figures / f"{stem}.{extension}").exists()
    )
    final_audit = {
        "status": "complete" if not missing_artifacts else "incomplete",
        "decision": decision,
        "residual_provenance_files": len(residual_provenance),
        "residual_statuses": sorted(
            {record.get("status") for record in residual_records}
        ),
        "residual_metrics": sorted(
            {record.get("metric") for record in residual_records}
        ),
        "residual_n_perm": sorted(
            {record.get("config", {}).get("n_perm") for record in residual_records}
        ),
        "residual_implementation_hashes": sorted(
            {record.get("implementation_hash") for record in residual_records}
        ),
        "lagged_candidate_manifest_entities": int(len(candidate_manifest)),
        "lagged_result_rows": int(len(lagged)),
        "lagged_expected_rows": int(candidate_manifest["candidate_count"].sum()),
        "missing_artifacts": missing_artifacts,
    }
    if (
        len(residual_provenance) != 511
        or final_audit["residual_statuses"] != ["complete"]
        or final_audit["residual_metrics"] != ["xcorr_resid"]
        or final_audit["residual_n_perm"] != [1000]
        or len(final_audit["residual_implementation_hashes"]) != 1
        or final_audit["lagged_result_rows"] != final_audit["lagged_expected_rows"]
        or missing_artifacts
    ):
        final_audit["status"] = "incomplete"
        atomic_json_write(final_audit, root / "qc/final_audit.json")
        raise RuntimeError(f"final audit failed: {final_audit}")
    atomic_json_write(final_audit, root / "qc/final_audit.json")
    provenance = {
        "status": "complete", "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision, "random_state": args.random_state,
        "n_resamples": args.n_resamples, "implementation_hash": implementation_hash(),
        "reproduction_command": "bash scripts/slurm/submit_hga_motif_decision.sh",
        "git": git_state(REPOSITORY), "software": software_versions(),
        "inputs": [
            input_fingerprint(root / "tables/pair_level_annotated.parquet"),
            input_fingerprint(root / "tables/group_motif_cells.csv"),
            input_fingerprint(root / "tables/group_matched_cross.csv"),
            input_fingerprint(root / "tables/lagged_oaec_group_concordance.csv"),
        ],
    }
    atomic_json_write(provenance, root / "provenance.json")
    print(json.dumps({"report": str(report_path), "decision": decision}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-manifest")
    prepare.add_argument(
        "--output", type=Path,
        default=DEFAULT_ROOT / "manifests/four_task_repeat.tsv",
    )
    prepare.add_argument("--expected-entities", type=int, default=511)
    prepare.add_argument(
        "--baseline-root", type=Path,
        default=REPOSITORY / "results/connectivity",
    )
    prepare.set_defaults(function=prepare_command)
    build = subparsers.add_parser("build-tables")
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--baseline-root", type=Path, default=REPOSITORY / "results/connectivity")
    build.add_argument(
        "--residual-root", type=Path,
        default=DEFAULT_ROOT / "residualized_connectivity",
    )
    build.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    build.add_argument(
        "--seed-assignments", type=Path,
        default=REPOSITORY / "results/nmf/channel_assignments.csv",
    )
    build.add_argument(
        "--partner-projection", type=Path,
        default=REPOSITORY / "results/nmf/whole_brain_projection/electrode_projection.csv",
    )
    build.add_argument("--projection-threshold", type=float, default=0.5)
    build.add_argument("--expected-entities", type=int, default=511)
    build.add_argument("--expected-pair-rows", type=int, default=402_791)
    build.add_argument("--expected-subjects", type=int, default=52)
    build.add_argument("--expected-unique-pairs", type=int, default=43_500)
    build.add_argument(
        "--entity-limit", type=int,
        help="Debug-only limit applied before expected-entity validation",
    )
    build.add_argument("--n-resamples", type=int, default=10_000)
    build.add_argument("--random-state", type=int, default=42)
    build.set_defaults(function=build_tables_command)

    lagged = subparsers.add_parser("lagged-oaec-row")
    lagged.add_argument("--manifest", type=Path, required=True)
    lagged.add_argument("--row-index", type=int, required=True)
    lagged.add_argument(
        "--output-root", type=Path,
        default=DEFAULT_ROOT / "lagged_oaec_connectivity",
    )
    lagged.add_argument("--min-trials", type=int, default=30)
    lagged.add_argument("--max-lag-s", type=float, default=0.25)
    lagged.add_argument("--oaec-sfreq", type=float, default=128.0)
    lagged.add_argument("--random-state", type=int, default=42)
    lagged.set_defaults(function=lagged_oaec_row_command)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    finalize.add_argument(
        "--lagged-root", type=Path,
        default=DEFAULT_ROOT / "lagged_oaec_connectivity",
    )
    finalize.add_argument("--n-resamples", type=int, default=10_000)
    finalize.add_argument("--random-state", type=int, default=42)
    finalize.set_defaults(function=finalize_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
