"""Soft-membership summaries of Insula-seeded OAEC connectivity.

This module deliberately separates three operations that were previously
collapsed into a hard 3 x 3 table:

1. OAEC edges are expressed as excess Fisher-z above their trial-shuffle null.
2. All Insula seeds are collapsed before group inference.
3. Extra-insular targets contribute continuously to every temporal motif via
   their NNLS projection weights; no winner-take-all label is required.

The resulting primary estimand is an Insula-to-motif embedding strength per
subject.  Phase-resolved values are retained as correlated, descriptive
secondary estimates because the current event-centred windows can overlap in
physical trial time.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_COMPONENTS = ("sensory", "sustain", "motor")
DEFAULT_INSULA_ROIS = ("aic", "pic", "insula")
DEFAULT_PHASES = ("stimulus", "delay", "go", "response")


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.astype("string")
        .fillna("false")
        .str.strip()
        .str.lower()
        .isin({"1", "true", "t", "yes", "y"})
    )


def _normalise_subject(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"^sub-", "", regex=True)


def prepare_soft_projection(
    projection: pd.DataFrame,
    *,
    components: Sequence[str] = DEFAULT_COMPONENTS,
    exclude_discovery: bool = True,
    exclude_insula: bool = True,
    insula_rois: Sequence[str] = DEFAULT_INSULA_ROIS,
    min_explained_energy: float = 0.0,
) -> pd.DataFrame:
    """Prepare continuous target weights from the whole-brain NNLS table.

    ``soft_weight_<component>`` is ``explained_energy * proportion_component``.
    Multiplying by fit quality prevents a confident-looking mixture from
    receiving full weight when the frozen templates reconstruct it poorly.
    Electrodes are not filtered by their winner-take-all ``best_component`` or
    by dominance.
    """

    components = tuple(components)
    required = {
        "subject",
        "channel",
        "roi",
        "explained_energy",
        *[f"proportion_{name}" for name in components],
    }
    missing = required - set(projection.columns)
    if missing:
        raise ValueError(f"projection table missing columns: {sorted(missing)}")
    if min_explained_energy < 0:
        raise ValueError("min_explained_energy must be non-negative")

    out = projection.copy()
    out["subject"] = _normalise_subject(out["subject"])
    out["channel"] = out["channel"].astype("string")
    out["roi"] = out["roi"].astype("string").str.strip()
    energy = pd.to_numeric(out["explained_energy"], errors="coerce").clip(0.0, 1.0)
    keep = out["subject"].notna() & out["channel"].notna() & energy.notna()
    keep &= energy >= float(min_explained_energy)
    if exclude_discovery and "in_discovery" in out.columns:
        keep &= ~_as_bool(out["in_discovery"])
    if exclude_insula:
        names = {str(name).strip().lower() for name in insula_rois}
        keep &= ~out["roi"].str.lower().isin(names)
    out = out.loc[keep].copy()
    energy = energy.loc[keep]

    proportions = []
    for name in components:
        column = pd.to_numeric(
            out[f"proportion_{name}"], errors="coerce"
        ).clip(0.0, 1.0)
        proportions.append(column.to_numpy(dtype=float))
    proportion_matrix = np.column_stack(proportions)
    denominator = np.nansum(proportion_matrix, axis=1)
    valid = np.isfinite(denominator) & (denominator > 0)
    out = out.loc[valid].copy()
    energy_values = energy.to_numpy(dtype=float)[valid]
    proportion_matrix = proportion_matrix[valid] / denominator[valid, None]
    for index, name in enumerate(components):
        out[f"soft_proportion_{name}"] = proportion_matrix[:, index]
        out[f"soft_weight_{name}"] = energy_values * proportion_matrix[:, index]
    return out.drop_duplicates("channel", keep="first").reset_index(drop=True)


def load_oaec_pair_tables(
    connectivity_root: Path,
    *,
    tasks: Iterable[str] | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load cached OAEC pair tables from the project results tree."""

    connectivity_root = Path(connectivity_root)
    task_names = tuple(tasks) if tasks is not None else None
    paths: list[Path] = []
    if task_names is None:
        paths = sorted(connectivity_root.glob("*/sub-*/oaec/*_pairs.parquet"))
    else:
        for task in task_names:
            paths.extend(
                sorted(
                    (connectivity_root / task).glob(
                        "sub-*/oaec/*_pairs.parquet"
                    )
                )
            )
    if not paths:
        raise FileNotFoundError(f"No OAEC pair tables under {connectivity_root}")
    frames = [pd.read_parquet(path, columns=columns) for path in paths]
    return pd.concat(frames, ignore_index=True)


def prepare_oaec_edges(
    pairs: pd.DataFrame,
    *,
    description: str | None = "Repeat",
    phases: Sequence[str] = DEFAULT_PHASES,
    require_target_effective: bool = False,
) -> pd.DataFrame:
    """Filter cached OAEC rows and calculate excess over trial-shuffle null.

    No edge-level p-value threshold is applied.  Pairwise significance is not
    used as a weight or inclusion criterion for the primary group estimate.
    """

    required = {"source", "target", "stat", "null_mean", "subject", "phase"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"OAEC pair table missing columns: {sorted(missing)}")

    out = pairs.copy()
    if "metric" in out.columns:
        out = out.loc[out["metric"].astype(str).str.lower().eq("oaec")].copy()
    if description is not None and "description" in out.columns:
        out = out.loc[out["description"].astype(str).eq(description)].copy()
    if "qc_pass" in out.columns:
        out = out.loc[_as_bool(out["qc_pass"])].copy()
    if "source_is_seed" in out.columns:
        out = out.loc[_as_bool(out["source_is_seed"])].copy()
    if "target_is_seed" in out.columns:
        out = out.loc[~_as_bool(out["target_is_seed"])].copy()
    if require_target_effective:
        if "target_effective" not in out.columns:
            raise ValueError("target_effective is required but absent")
        out = out.loc[_as_bool(out["target_effective"])].copy()

    out["phase"] = out["phase"].astype("string").str.lower().str.strip()
    phase_set = {str(phase).lower() for phase in phases}
    out = out.loc[out["phase"].isin(phase_set)].copy()
    out["subject"] = _normalise_subject(out["subject"])
    out = out.rename(columns={"source": "source_channel", "target": "target_channel"})
    out["source_channel"] = out["source_channel"].astype("string")
    out["target_channel"] = out["target_channel"].astype("string")
    observed = pd.to_numeric(out["stat"], errors="coerce")
    null_mean = pd.to_numeric(out["null_mean"], errors="coerce")
    out["oaec_excess"] = observed - null_mean
    out = out.dropna(
        subset=[
            "subject",
            "source_channel",
            "target_channel",
            "phase",
            "oaec_excess",
        ]
    )
    low = np.minimum(out["source_channel"], out["target_channel"])
    high = np.maximum(out["source_channel"], out["target_channel"])
    out["pair_key"] = low + "||" + high
    return out.reset_index(drop=True)


def attach_soft_target_weights(
    edges: pd.DataFrame,
    projection: pd.DataFrame,
    *,
    components: Sequence[str] = DEFAULT_COMPONENTS,
) -> pd.DataFrame:
    """Attach continuous target motif weights to Insula-seeded OAEC edges."""

    weight_columns = [f"soft_weight_{name}" for name in components]
    required_edges = {"subject", "target_channel", "oaec_excess"}
    required_projection = {"subject", "channel", *weight_columns}
    missing_edges = required_edges - set(edges.columns)
    missing_projection = required_projection - set(projection.columns)
    if missing_edges:
        raise ValueError(f"edge table missing columns: {sorted(missing_edges)}")
    if missing_projection:
        raise ValueError(
            f"soft projection table missing columns: {sorted(missing_projection)}"
        )

    target = projection.copy().rename(columns={"channel": "target_channel"})
    keep = [
        "subject",
        "target_channel",
        "roi",
        "explained_energy",
        *weight_columns,
        *[
            f"soft_proportion_{name}"
            for name in components
            if f"soft_proportion_{name}" in target.columns
        ],
    ]
    keep = [column for column in keep if column in target.columns]
    out = edges.merge(
        target[keep],
        on=["subject", "target_channel"],
        how="inner",
        validate="many_to_one",
    )
    return out.reset_index(drop=True)


def collapse_insula_seeds(
    weighted_edges: pd.DataFrame,
    *,
    condition_columns: Sequence[str] = (
        "subject",
        "dataset",
        "task",
        "phase",
        "description",
    ),
    components: Sequence[str] = DEFAULT_COMPONENTS,
) -> pd.DataFrame:
    """Collapse all Insula seeds to one OAEC value per target and condition."""

    condition_columns = tuple(
        column for column in condition_columns if column in weighted_edges.columns
    )
    if "subject" not in condition_columns:
        condition_columns = ("subject", *condition_columns)
    group_columns = [*condition_columns, "target_channel"]
    weight_columns = [f"soft_weight_{name}" for name in components]
    agg: dict[str, tuple[str, str]] = {
        "oaec_excess": ("oaec_excess", "mean"),
        "n_insula_seeds": ("source_channel", "nunique"),
        "n_edge_rows": ("pair_key", "size"),
    }
    for column in ("roi", "explained_energy", *weight_columns):
        if column in weighted_edges.columns:
            agg[column] = (column, "first")
    return (
        weighted_edges.groupby(group_columns, observed=True, dropna=False)
        .agg(**agg)
        .reset_index()
    )


def condition_network_embedding(
    target_values: pd.DataFrame,
    *,
    components: Sequence[str] = DEFAULT_COMPONENTS,
    condition_columns: Sequence[str] = (
        "subject",
        "dataset",
        "task",
        "phase",
        "description",
    ),
    min_targets: int = 1,
) -> pd.DataFrame:
    """Calculate a soft-weighted Insula embedding score per condition."""

    if min_targets < 1:
        raise ValueError("min_targets must be at least 1")
    condition_columns = tuple(
        column for column in condition_columns if column in target_values.columns
    )
    rows: list[dict[str, object]] = []
    for keys, frame in target_values.groupby(
        list(condition_columns), observed=True, dropna=False, sort=True
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(condition_columns, keys))
        baseline = float(frame["oaec_excess"].mean())
        for component in components:
            weight_column = f"soft_weight_{component}"
            weights = pd.to_numeric(frame[weight_column], errors="coerce").to_numpy()
            values = pd.to_numeric(frame["oaec_excess"], errors="coerce").to_numpy()
            valid = np.isfinite(weights) & np.isfinite(values) & (weights > 0)
            n_targets = int(valid.sum())
            weight_sum = float(weights[valid].sum()) if n_targets else 0.0
            embedding = (
                float(np.average(values[valid], weights=weights[valid]))
                if n_targets >= min_targets and weight_sum > 0
                else np.nan
            )
            rows.append(
                {
                    **base,
                    "component": component,
                    "embedding": embedding,
                    "network_selectivity": embedding - baseline,
                    "all_target_baseline": baseline,
                    "n_targets": n_targets,
                    "weight_sum": weight_sum,
                }
            )
    return pd.DataFrame(rows)


def subject_embedding(
    condition_scores: pd.DataFrame,
    *,
    keep_phase: bool,
) -> pd.DataFrame:
    """Average correlated condition estimates within subject before inference."""

    required = {"subject", "component", "embedding", "network_selectivity"}
    missing = required - set(condition_scores.columns)
    if missing:
        raise ValueError(f"condition table missing columns: {sorted(missing)}")
    group_columns = ["subject", "component"]
    if keep_phase:
        if "phase" not in condition_scores.columns:
            raise ValueError("phase is required when keep_phase=True")
        group_columns.append("phase")
    grouped = condition_scores.groupby(group_columns, observed=True, sort=True)
    return grouped.agg(
        embedding=("embedding", "mean"),
        network_selectivity=("network_selectivity", "mean"),
        n_conditions=("embedding", "count"),
        median_n_targets=("n_targets", "median"),
    ).reset_index()


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    out = np.full_like(p_values, np.nan)
    valid = np.isfinite(p_values)
    if not valid.any():
        return out
    p = p_values[valid]
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    out[valid] = restored
    return out


def group_sign_flip_inference(
    subject_scores: pd.DataFrame,
    *,
    value_column: str = "embedding",
    group_columns: Sequence[str] = ("component",),
    n_permutations: int = 20_000,
    n_bootstrap: int = 20_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Subject-level mean, bootstrap CI, and one-sided sign-flip test.

    The test asks whether the group mean excess OAEC is greater than zero.  It
    treats subject, not pair or phase, as the independent sampling unit.
    """

    if n_permutations < 1 or n_bootstrap < 1:
        raise ValueError("resampling counts must be positive")
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []
    for keys, frame in subject_scores.groupby(
        list(group_columns), observed=True, dropna=False, sort=True
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = pd.to_numeric(frame[value_column], errors="coerce").dropna().to_numpy()
        if len(values) == 0:
            continue
        observed = float(values.mean())
        bootstrap_index = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
        bootstrap_means = values[bootstrap_index].mean(axis=1)
        signs = rng.choice((-1.0, 1.0), size=(n_permutations, len(values)))
        permuted = (signs * values).mean(axis=1)
        p_greater = float((1 + np.sum(permuted >= observed)) / (n_permutations + 1))
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "value": value_column,
                "n_subjects": int(len(values)),
                "mean": observed,
                "median": float(np.median(values)),
                "ci_low": float(np.quantile(bootstrap_means, 0.025)),
                "ci_high": float(np.quantile(bootstrap_means, 0.975)),
                "p_greater_zero": p_greater,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_fdr"] = _bh_fdr(out["p_greater_zero"].to_numpy())
    return out

