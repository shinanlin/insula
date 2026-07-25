"""Atomic result serialization, provenance, and config-hash resume checks."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
from typing import Iterable, Mapping

import pandas as pd
import xarray as xr
from mne_bids import BIDSPath

from .config import SCHEMA_VERSION
from .io import AnalysisData, input_fingerprint
from .result import MetricResult


DEFAULT_OUTPUT_ROOT = Path(
    "/hpc/group/coganlab/nanlinshi/insula-functional/"
    "results/connectivity"
)


def implementation_hash() -> str:
    """Hash this package's Python source, including uncommitted changes."""

    digest = sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def git_state(repository: str | Path) -> dict[str, object]:
    """Return commit and dirty state without mutating the worktree."""

    root = Path(repository)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def software_versions() -> dict[str, str]:
    packages = (
        "numpy",
        "scipy",
        "pandas",
        "xarray",
        "mne",
        "h5netcdf",
        "netCDF4",
        "pyarrow",
    )
    versions: dict[str, str] = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _optional_entity(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _connectivity_dataset_root(
    output_root: str | Path, entities: Mapping[str, str]
) -> Path:
    return Path(output_root) / str(entities.get("dataset", "unknown"))


def connectivity_bids_path(
    output_root: str | Path,
    entities: Mapping[str, str],
    *,
    metric: str,
    suffix: str,
    extension: str,
) -> BIDSPath:
    """Build a BIDSPath for one connectivity artifact."""

    kwargs: dict[str, object] = {
        "root": str(_connectivity_dataset_root(output_root, entities)),
        "subject": str(entities.get("subject", "unknown")),
        "task": str(entities.get("task", "unknown")),
        "processing": str(entities.get("phase", "unknown")),
        "description": str(entities.get("description", "unknown")),
        "datatype": metric,
        "suffix": suffix,
        "extension": extension,
        "check": False,
    }
    recording = _optional_entity(entities.get("recording"))
    if recording is not None:
        kwargs["recording"] = recording
    run = _optional_entity(entities.get("run"))
    if run is not None:
        kwargs["run"] = run
    acquisition = _optional_entity(entities.get("acquisition"))
    if acquisition is not None:
        kwargs["acquisition"] = acquisition
    return BIDSPath(**kwargs)


def failure_bids_path(
    output_root: str | Path, entities: Mapping[str, str]
) -> BIDSPath:
    """Build a BIDSPath for a serialized failure record."""

    dataset = str(entities.get("dataset", "unknown"))
    kwargs: dict[str, object] = {
        "root": str(Path(output_root) / "failures" / dataset),
        "subject": str(entities.get("subject", "unknown")),
        "task": str(entities.get("task", "unknown")),
        "processing": str(entities.get("phase", "unknown")),
        "description": str(entities.get("description", "unknown")),
        "suffix": "failure",
        "extension": ".json",
        "check": False,
    }
    recording = _optional_entity(entities.get("recording"))
    if recording is not None:
        kwargs["recording"] = recording
    run = _optional_entity(entities.get("run"))
    if run is not None:
        kwargs["run"] = run
    acquisition = _optional_entity(entities.get("acquisition"))
    if acquisition is not None:
        kwargs["acquisition"] = acquisition
    return BIDSPath(**kwargs)


def entity_basename(entities: Mapping[str, str]) -> str:
    """Return the BIDS basename (entities only, no suffix/extension)."""

    pairs_path = connectivity_bids_path(
        output_root="/tmp",
        entities=entities,
        metric="xcorr",
        suffix="pairs",
        extension=".parquet",
    )
    filename = Path(pairs_path.fpath).name
    return filename[: -len("_pairs.parquet")]


def entity_stem(entities: Mapping[str, str]) -> str:
    """Backward-compatible alias for the entity BIDS basename."""

    return entity_basename(entities)


def metric_output_dir(
    output_root: str | Path,
    entities: Mapping[str, str],
    metric: str,
) -> Path:
    pairs_path = connectivity_bids_path(
        output_root,
        entities,
        metric=metric,
        suffix="pairs",
        extension=".parquet",
    )
    return Path(pairs_path.fpath).parent


def provenance_path(
    output_root: str | Path,
    entities: Mapping[str, str],
    metric: str,
) -> Path:
    return Path(
        connectivity_bids_path(
            output_root,
            entities,
            metric=metric,
            suffix="provenance",
            extension=".json",
        ).fpath
    )


def entity_output_dir(
    output_root: str | Path, entities: Mapping[str, str]
) -> Path:
    """Subject-level directory containing all metric datatype folders."""

    return (
        _connectivity_dataset_root(output_root, entities)
        / f"sub-{entities.get('subject', 'unknown')}"
    )


def atomic_json_write(payload: Mapping[str, object], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    return destination


def atomic_table_write(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    require_parquet: bool = False,
) -> tuple[Path, str, str | None]:
    """Write Parquet atomically, with an explicit compressed-CSV fallback."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(destination)
        return destination, "parquet", None
    except (ImportError, ModuleNotFoundError) as error:
        if temporary.exists():
            temporary.unlink()
        if require_parquet:
            raise RuntimeError(
                "Parquet output requires pyarrow or fastparquet"
            ) from error
        fallback = destination.with_suffix(".csv.gz")
        fallback_temporary = fallback.with_name(fallback.name + ".tmp")
        frame.to_csv(
            fallback_temporary, index=False, compression="gzip"
        )
        fallback_temporary.replace(fallback)
        return fallback, "csv.gz", str(error)


def atomic_netcdf_write(dataset: xr.Dataset, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    encoding = {
        name: {"zlib": True, "complevel": 4, "shuffle": True}
        for name, variable in dataset.data_vars.items()
        if variable.dtype.kind not in {"O", "U"}
    }
    dataset.to_netcdf(
        temporary, engine="h5netcdf", encoding=encoding
    )
    temporary.replace(destination)
    return destination


def existing_result_matches(
    output_root: str | Path,
    entities: Mapping[str, str],
    metric: str,
    config_hash: str,
) -> bool:
    provenance = provenance_path(output_root, entities, metric)
    if not provenance.exists():
        return False
    try:
        payload = json.loads(provenance.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "complete"
        and payload.get("config_hash") == config_hash
    )


def add_common_pair_columns(
    frame: pd.DataFrame,
    data: AnalysisData,
    *,
    entity_seed: int,
) -> pd.DataFrame:
    output = frame.copy()
    for key, value in data.entities.items():
        output[key] = value
    output["schema_version"] = SCHEMA_VERSION
    output["entity_seed"] = int(entity_seed)
    output["n_trials_original"] = data.n_original_trials
    output["n_trials_used"] = data.n_trials
    output["n_trials_dropped"] = len(data.dropped_trials)
    output["n_channels_used"] = data.n_channels
    output["n_strict_insula_seeds"] = len(data.seed_frame)
    output["n_eligible_pairs"] = data.n_eligible_pairs_before_limit
    output["n_pairs_computed"] = len(data.pair_frame)
    output["connectivity_interpretation"] = "pairwise_functional"
    return output


def base_provenance(
    data: AnalysisData,
    *,
    config: Mapping[str, object],
    config_hash: str,
    entity_seed: int,
    repository: str | Path,
) -> dict[str, object]:
    inputs = [
        data.zscore_path,
        data.raw_path,
        data.hammers_path,
    ]
    if data.effective_path is not None:
        inputs.append(data.effective_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "config": dict(config),
        "config_hash": config_hash,
        "implementation_hash": implementation_hash(),
        "entity_seed": int(entity_seed),
        "entities": data.entities,
        "input_fingerprints": [
            input_fingerprint(path) for path in inputs
        ],
        "software": software_versions(),
        "git": git_state(repository),
        "qc": {
            "n_trials_original": data.n_original_trials,
            "n_trials_used": data.n_trials,
            "trial_indices_used": data.trial_indices.astype(int).tolist(),
            "dropped_trials": data.dropped_trials,
            "dropped_channels": data.dropped_channels,
            "n_channels_used": data.n_channels,
            "n_strict_insula_seeds": len(data.seed_frame),
            "strict_insula_seed_channels": (
                data.seed_frame["channel"].astype(str).tolist()
            ),
            "n_eligible_pairs": data.n_eligible_pairs_before_limit,
            "n_pairs_computed": len(data.pair_frame),
        },
    }


def write_metric_result(
    result: MetricResult,
    data: AnalysisData,
    *,
    output_root: str | Path,
    config: Mapping[str, object],
    config_hash: str,
    entity_seed: int,
    repository: str | Path,
    started_at: datetime,
    elapsed_seconds: float,
    require_parquet: bool = False,
) -> dict[str, object]:
    """Atomically serialize a completed metric and its provenance."""

    destination = metric_output_dir(
        output_root, data.entities, result.metric
    )
    destination.mkdir(parents=True, exist_ok=True)
    pair_table = add_common_pair_columns(
        result.pair_table, data, entity_seed=entity_seed
    )
    table_path, table_format, table_warning = atomic_table_write(
        pair_table,
        connectivity_bids_path(
            output_root,
            data.entities,
            metric=result.metric,
            suffix="pairs",
            extension=".parquet",
        ).fpath,
        require_parquet=require_parquet,
    )
    detail_path = atomic_netcdf_write(
        result.detail,
        connectivity_bids_path(
            output_root,
            data.entities,
            metric=result.metric,
            suffix="detail",
            extension=".nc",
        ).fpath,
    )
    auxiliary_paths: dict[str, str] = {}
    auxiliary_formats: dict[str, str] = {}
    for name, table in result.auxiliary_tables.items():
        path, file_format, _ = atomic_table_write(
            table,
            connectivity_bids_path(
                output_root,
                data.entities,
                metric=result.metric,
                suffix=name,
                extension=".parquet",
            ).fpath,
            require_parquet=require_parquet,
        )
        auxiliary_paths[name] = str(path)
        auxiliary_formats[name] = file_format

    provenance = base_provenance(
        data,
        config=config,
        config_hash=config_hash,
        entity_seed=entity_seed,
        repository=repository,
    )
    provenance.update(
        {
            "status": "complete",
            "metric": result.metric,
            "started_at": started_at.astimezone(timezone.utc).isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": float(elapsed_seconds),
            "peak_rss_mb": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            ),
            "runtime_metadata": result.runtime_metadata,
            "outputs": {
                "pair_table": str(table_path),
                "pair_table_format": table_format,
                "detail_netcdf": str(detail_path),
                "auxiliary_tables": auxiliary_paths,
                "auxiliary_formats": auxiliary_formats,
            },
            "warnings": (
                []
                if table_warning is None
                else [
                    "Parquet engine unavailable; wrote csv.gz fallback: "
                    + table_warning
                ]
            ),
        }
    )
    provenance_path_written = atomic_json_write(
        provenance,
        connectivity_bids_path(
            output_root,
            data.entities,
            metric=result.metric,
            suffix="provenance",
            extension=".json",
        ).fpath,
    )
    return {
        "metric": result.metric,
        "status": "complete",
        "output_dir": str(destination),
        "pair_table": str(table_path),
        "detail": str(detail_path),
        "provenance": str(provenance_path_written),
        "elapsed_seconds": float(elapsed_seconds),
        "peak_rss_mb": provenance["peak_rss_mb"],
        "warnings": provenance["warnings"],
    }


def write_failure_record(
    *,
    output_root: str | Path,
    entities: Mapping[str, str],
    config_hash: str,
    reason: str,
    traceback_text: str,
) -> Path:
    return atomic_json_write(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "entities": dict(entities),
            "config_hash": config_hash,
            "reason": reason,
            "traceback": traceback_text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        failure_bids_path(output_root, entities).fpath,
    )


def collect_provenance_files(
    output_root: str | Path,
) -> Iterable[Path]:
    return Path(output_root).glob("**/*_provenance.json")
