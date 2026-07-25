"""Command-line entry point for pairwise Insula-to-all connectivity."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback
from typing import Callable, Sequence

import pandas as pd

from .config import ConnectivityConfig, DEFAULT_DATASETS
from .diagnostics import create_prototype_diagnostic
from .io import (
    discover_manifest,
    load_analysis_data,
    parse_filename_entities,
    read_manifest_row,
    write_manifest,
)
from .oaec import compute_oaec
from .output import (
    DEFAULT_OUTPUT_ROOT,
    atomic_json_write,
    entity_stem,
    existing_result_matches,
    implementation_hash,
    metric_output_dir,
    provenance_path,
    write_failure_record,
    write_metric_result,
)
from .permutation import generate_derangements, stable_seed
from .tf_dwpli import compute_tf_dwpli
from .xcorr import compute_xcorr


REPOSITORY = Path(
    "/hpc/group/coganlab/nanlinshi/insula-functional"
)
METRICS = ("xcorr", "oaec", "wpli")


def _manifest_entities(row: pd.Series) -> dict[str, str]:
    parsed = parse_filename_entities(str(row["zscore_path"]))
    return {
        "dataset": str(row.get("dataset", "")),
        "subject": str(row.get("subject", parsed.get("sub", ""))),
        "task": str(row.get("task", parsed.get("task", ""))),
        "phase": str(row.get("phase", parsed.get("proc", ""))),
        "description": str(
            row.get("description", parsed.get("desc", ""))
        ),
        "recording": str(
            row.get(
                "recording",
                parsed.get("recording", parsed.get("rec", "")),
            )
        ),
        "run": str(row.get("run", parsed.get("run", ""))),
        "acquisition": str(
            row.get("acquisition", parsed.get("acq", ""))
        ),
    }


def _entity_seed(config: ConnectivityConfig, entities: dict[str, str]) -> int:
    return stable_seed(
        config.random_state,
        [entities.get(key, "") for key in sorted(entities)],
    )


def _metric_config_hash(
    config: ConnectivityConfig,
    *,
    metric: str,
    entities: dict[str, str],
    pair_limit: int | None,
) -> str:
    return config.stable_hash(
        {
            "metric": metric,
            "entities": entities,
            "pair_limit": pair_limit,
            "implementation_hash": implementation_hash(),
            "network": "strict_hammers_insula_to_all",
            "trial_null": "shared_target_trial_derangement",
        }
    )


def _assert_writable_result_state(
    output_root: Path,
    entities: dict[str, str],
    metric: str,
    config_hash: str,
    *,
    overwrite: bool,
) -> bool:
    """Return True for a matching completed result that can be skipped."""

    provenance_file = provenance_path(output_root, entities, metric)
    if existing_result_matches(
        output_root, entities, metric, config_hash
    ):
        return True
    if provenance_file.exists() and not overwrite:
        try:
            previous = json.loads(
                provenance_file.read_text(encoding="utf-8")
            )
            previous_hash = previous.get("config_hash", "unknown")
        except (OSError, json.JSONDecodeError):
            previous_hash = "unreadable"
        raise RuntimeError(
            f"Existing result at {provenance_file.parent} has config_hash="
            f"{previous_hash}, expected {config_hash}; pass --overwrite "
            "to replace it"
        )
    return False


def _compute_metric(
    metric: str,
    data,
    permutations,
    config: ConnectivityConfig,
    scratch_dir: Path,
):
    if metric == "xcorr":
        return compute_xcorr(
            data.hga_data,
            data.hga_sfreq,
            data.pair_frame,
            permutations,
            config,
            scratch_dir=scratch_dir,
        )
    if metric == "oaec":
        return compute_oaec(
            data.raw_data,
            data.raw_times,
            data.raw_sfreq,
            data.entities["phase"],
            data.pair_frame,
            permutations,
            config,
        )
    if metric == "wpli":
        return compute_tf_dwpli(
            data.raw_data,
            data.raw_times,
            data.raw_sfreq,
            data.entities["phase"],
            data.pair_frame,
            permutations,
            config,
        )
    raise ValueError(f"Unknown metric {metric!r}")


def build_manifest_command(args: argparse.Namespace) -> int:
    roots = DEFAULT_DATASETS
    if args.dataset:
        missing = sorted(set(args.dataset).difference(roots))
        if missing:
            raise ValueError(f"Unknown datasets: {missing}")
        roots = {name: roots[name] for name in args.dataset}
    frame = discover_manifest(roots)
    selected = frame
    if args.ready_only:
        selected = frame.loc[frame["status"] == "ready"].reset_index(
            drop=True
        )
        if args.excluded_output is not None:
            write_manifest(
                frame.loc[frame["status"] != "ready"].reset_index(drop=True),
                args.excluded_output,
            )
    write_manifest(selected, args.output)
    summary = (
        frame.groupby(["dataset", "status"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    print(summary.to_string(index=False))
    print(
        f"manifest={args.output} rows={len(selected)} "
        f"discovered={len(frame)}"
    )
    return 0


def run_row_command(args: argparse.Namespace) -> int:
    row = read_manifest_row(args.manifest, args.row_index)
    entities = _manifest_entities(row)
    output_root = Path(args.output_root)
    metrics = tuple(dict.fromkeys(args.metrics))
    config = ConnectivityConfig(
        n_perm=args.n_perm,
        random_state=args.random_state,
        alpha=args.alpha,
        min_trials=args.min_trials,
        max_lag_s=args.max_lag_s,
        permutation_chunk_size=args.permutation_chunk_size,
        pair_block_size=args.pair_block_size,
        n_jobs=args.n_jobs,
        oaec_sfreq=args.oaec_sfreq,
        wpli_sfreq=args.wpli_sfreq,
        wpli_freq_step=args.wpli_freq_step,
        save_full_null=args.save_full_null,
    )
    config.validate()
    entity_seed = _entity_seed(config, entities)
    hashes = {
        metric: _metric_config_hash(
            config,
            metric=metric,
            entities=entities,
            pair_limit=args.pair_limit,
        )
        for metric in metrics
    }
    try:
        if str(row.get("status", "ready")) != "ready":
            raise RuntimeError(
                f"Manifest row status={row.get('status')}: "
                f"{row.get('reason', '')}"
            )
        pending: list[str] = []
        report: list[dict[str, object]] = []
        for metric in metrics:
            destination = metric_output_dir(
                output_root, entities, metric
            )
            if _assert_writable_result_state(
                output_root,
                entities,
                metric,
                hashes[metric],
                overwrite=args.overwrite,
            ):
                report.append(
                    {
                        "metric": metric,
                        "status": "skipped_config_hash_match",
                        "output_dir": str(destination),
                    }
                )
            else:
                pending.append(metric)
        if not pending:
            print(json.dumps(report, indent=2))
            return 0

        data = load_analysis_data(
            row,
            min_trials=config.min_trials,
            pair_limit=args.pair_limit,
        )
        if data.entities != entities:
            raise RuntimeError(
                "Loaded input entities do not match manifest entities"
            )
        permutations = generate_derangements(
            config.n_perm, data.n_trials, entity_seed
        )
        scratch_dir = Path(
            args.scratch_dir
            or os.environ.get("SLURM_TMPDIR")
            or os.environ.get("TMPDIR")
            or "/tmp"
        ) / f"insula-connectivity-{entity_stem(entities)}-{os.getpid()}"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        for metric in pending:
            started_at = datetime.now(timezone.utc)
            start = time.perf_counter()
            result = _compute_metric(
                metric, data, permutations, config, scratch_dir
            )
            elapsed = time.perf_counter() - start
            report.append(
                write_metric_result(
                    result,
                    data,
                    output_root=output_root,
                    config={
                        **config.as_dict(),
                        "pair_limit": args.pair_limit,
                        "metric": metric,
                    },
                    config_hash=hashes[metric],
                    entity_seed=entity_seed,
                    repository=REPOSITORY,
                    started_at=started_at,
                    elapsed_seconds=elapsed,
                    require_parquet=args.require_parquet,
                )
            )
        summary_path = (
            output_root
            / "runs"
            / str(entities.get("dataset", "unknown"))
            / f"{entity_stem(entities)}.json"
        )
        atomic_json_write(
            {
                "status": "complete",
                "entities": entities,
                "metrics": report,
                "entity_seed": int(entity_seed),
                "n_perm": config.n_perm,
                "n_trials": data.n_trials,
                "n_channels": data.n_channels,
                "n_seeds": len(data.seed_frame),
                "n_eligible_pairs": data.n_eligible_pairs_before_limit,
                "n_pairs_computed": len(data.pair_frame),
            },
            summary_path,
        )
        print(json.dumps(report, indent=2))
        return 0
    except Exception as error:
        outer_hash = config.stable_hash(
            {
                "metrics": metrics,
                "entities": entities,
                "implementation_hash": implementation_hash(),
            }
        )
        failure = write_failure_record(
            output_root=output_root,
            entities=entities,
            config_hash=outer_hash,
            reason=f"{type(error).__name__}: {error}",
            traceback_text=traceback.format_exc(),
        )
        print(f"failure_record={failure}")
        raise


def audit_command(args: argparse.Namespace) -> int:
    manifest = pd.read_csv(
        args.manifest, sep="\t", keep_default_na=False
    )
    missing_rows: list[int] = []
    records: list[dict[str, object]] = []
    for row_index, row in manifest.iterrows():
        entities = _manifest_entities(row)
        incomplete: list[str] = []
        for metric in args.metrics:
            provenance_file = provenance_path(
                args.output_root, entities, metric
            )
            status = "missing"
            if provenance_file.exists():
                try:
                    status = json.loads(
                        provenance_file.read_text(encoding="utf-8")
                    ).get("status", "unknown")
                except (OSError, json.JSONDecodeError):
                    status = "unreadable"
            if status != "complete":
                incomplete.append(f"{metric}:{status}")
        records.append(
            {
                "row_index": row_index,
                **entities,
                "status": "complete" if not incomplete else "retry",
                "reason": ",".join(incomplete),
            }
        )
        if incomplete:
            missing_rows.append(row_index)
    audit = pd.DataFrame(records)
    output = Path(args.output)
    write_manifest(audit, output)
    retry = manifest.iloc[missing_rows].copy()
    retry.insert(0, "original_row_index", missing_rows)
    retry_path = output.with_name(output.stem + "_retry.tsv")
    write_manifest(retry, retry_path)
    print(
        f"audit={output} complete={len(audit)-len(retry)} "
        f"retry={len(retry)} retry_manifest={retry_path}"
    )
    return 0


def diagnostics_command(args: argparse.Namespace) -> int:
    output = create_prototype_diagnostic(args.entity_dir, args.output)
    print(f"diagnostic={output}")
    return 0


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-perm", type=int, default=1_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-trials", type=int, default=30)
    parser.add_argument("--max-lag-s", type=float, default=0.25)
    parser.add_argument("--permutation-chunk-size", type=int, default=100)
    parser.add_argument("--pair-block-size", type=int, default=32)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--oaec-sfreq", type=float, default=128.0)
    parser.add_argument("--wpli-sfreq", type=float, default=256.0)
    parser.add_argument("--wpli-freq-step", type=float, default=1.0)
    parser.add_argument("--save-full-null", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict Hammers Insula-to-all pairwise connectivity"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("build-manifest")
    manifest_parser.add_argument("--dataset", action="append")
    manifest_parser.add_argument("--output", type=Path, required=True)
    manifest_parser.add_argument("--ready-only", action="store_true")
    manifest_parser.add_argument("--excluded-output", type=Path)
    manifest_parser.set_defaults(function=build_manifest_command)

    run_parser = subparsers.add_parser("run-row")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--row-index", type=int, required=True)
    run_parser.add_argument(
        "--metrics", nargs="+", choices=METRICS, default=list(METRICS)
    )
    run_parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    run_parser.add_argument("--scratch-dir", type=Path)
    run_parser.add_argument("--pair-limit", type=int)
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.add_argument("--require-parquet", action="store_true")
    _add_config_arguments(run_parser)
    run_parser.set_defaults(function=run_row_command)

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--manifest", type=Path, required=True)
    audit_parser.add_argument(
        "--metrics", nargs="+", choices=METRICS, default=list(METRICS)
    )
    audit_parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    audit_parser.add_argument("--output", type=Path, required=True)
    audit_parser.set_defaults(function=audit_command)

    diagnostics_parser = subparsers.add_parser("diagnostics")
    diagnostics_parser.add_argument("--entity-dir", type=Path, required=True)
    diagnostics_parser.add_argument("--output", type=Path)
    diagnostics_parser.set_defaults(function=diagnostics_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    function: Callable[[argparse.Namespace], int] = args.function
    return function(args)


if __name__ == "__main__":
    raise SystemExit(main())
