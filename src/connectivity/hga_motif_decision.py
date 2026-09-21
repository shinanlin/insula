"""Decision-report statistics for lagged HGA amplitude motif coupling.

The functions in this module deliberately operate on tidy data frames.  Raw
signal estimation remains in :mod:`src.connectivity.pairwise`; this module
annotates those results with continuous NMF/NNLS motif weights and performs
subject-balanced summaries.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .pairwise.permutation import benjamini_hochberg


MOTIFS: tuple[str, ...] = ("sensory", "sustain", "motor")
PHASES: tuple[str, ...] = ("Stimulus", "Delay", "Go", "Response")
PRIMARY_TASKS: tuple[str, ...] = (
    "PhonemeSequence",
    "LexicalDelay",
    "PictureNaming",
    "SentenceRep",
)
ENTITY_COLUMNS: tuple[str, ...] = (
    "dataset",
    "subject",
    "task",
    "phase",
    "description",
    "recording",
    "run",
    "acquisition",
)


@dataclass(frozen=True)
class MetricColumns:
    """Columns used to summarize one connectivity estimator."""

    name: str
    sig_fdr: str
    sig_fwer: str
    effect: str


def normalize_motif_weights(
    frame: pd.DataFrame,
    *,
    input_prefix: str = "loading_",
    output_prefix: str = "proportion_",
    motifs: Sequence[str] = MOTIFS,
) -> pd.DataFrame:
    """Normalize non-negative motif loadings row-wise.

    Rows with missing, negative, non-finite, or zero-sum loadings receive NaN
    proportions and can therefore be retained for coverage accounting without
    entering motif-weighted inference.
    """

    output = frame.copy()
    columns = [f"{input_prefix}{motif}" for motif in motifs]
    missing = set(columns).difference(output.columns)
    if missing:
        raise ValueError(f"missing motif loading columns: {sorted(missing)}")
    values = output[columns].to_numpy(dtype=float)
    valid = np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1)
    totals = np.sum(values, axis=1)
    valid &= totals > np.finfo(float).eps
    normalized = np.full(values.shape, np.nan, dtype=float)
    normalized[valid] = values[valid] / totals[valid, None]
    for index, motif in enumerate(motifs):
        output[f"{output_prefix}{motif}"] = normalized[:, index]
    return output


def classify_lag_clusters(
    pair_table: pd.DataFrame,
    cluster_table: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Classify FDR-significant xcorr pairs by their strongest lag cluster.

    Only clusters passing the within-pair cluster threshold are considered.
    A pair is ambiguous when significant clusters occur strictly on both sides
    of zero.  Otherwise the largest-mass cluster determines whether temporal
    precedence is strictly negative/positive or unresolved because it spans
    zero.  Negative lag follows the production convention: source/Insula leads.
    """

    required_pairs = {"pair_id", "sig_fdr"}
    required_clusters = {
        "pair_id",
        "lag_start_s",
        "lag_stop_s",
        "peak_lag_s",
        "cluster_mass",
        "p_pair_cluster",
    }
    missing_pairs = required_pairs.difference(pair_table.columns)
    missing_clusters = required_clusters.difference(cluster_table.columns)
    if missing_pairs or missing_clusters:
        raise ValueError(
            "lag classification missing columns: "
            f"pairs={sorted(missing_pairs)}, clusters={sorted(missing_clusters)}"
        )
    base = pair_table[["pair_id", "sig_fdr"]].drop_duplicates("pair_id").copy()
    base["lag_direction"] = "not_fdr"
    base["strict_peak_lag_s"] = np.nan
    base["strict_cluster_mass"] = np.nan
    base["strict_cluster_start_s"] = np.nan
    base["strict_cluster_stop_s"] = np.nan
    base["n_significant_lag_clusters"] = 0
    significant_ids = set(base.loc[base["sig_fdr"].astype(bool), "pair_id"])
    clusters = cluster_table.loc[
        cluster_table["pair_id"].isin(significant_ids)
        & cluster_table["p_pair_cluster"].lt(alpha)
    ].copy()
    if clusters.empty:
        return base.drop(columns="sig_fdr")

    index_by_id = {pair_id: index for index, pair_id in enumerate(base["pair_id"])}
    for pair_id, group in clusters.groupby("pair_id", sort=False):
        row_index = index_by_id[pair_id]
        negative = (group["lag_start_s"] < 0) & (group["lag_stop_s"] < 0)
        positive = (group["lag_start_s"] > 0) & (group["lag_stop_s"] > 0)
        base.loc[row_index, "n_significant_lag_clusters"] = len(group)
        strongest = group.loc[group["cluster_mass"].idxmax()]
        base.loc[row_index, "strict_peak_lag_s"] = float(strongest["peak_lag_s"])
        base.loc[row_index, "strict_cluster_mass"] = float(strongest["cluster_mass"])
        base.loc[row_index, "strict_cluster_start_s"] = float(
            strongest["lag_start_s"]
        )
        base.loc[row_index, "strict_cluster_stop_s"] = float(strongest["lag_stop_s"])
        if negative.any() and positive.any():
            direction = "ambiguous"
        elif strongest["lag_start_s"] < 0 and strongest["lag_stop_s"] < 0:
            direction = "insula_leads"
        elif strongest["lag_start_s"] > 0 and strongest["lag_stop_s"] > 0:
            direction = "partner_leads"
        else:
            direction = "unresolved"
        base.loc[row_index, "lag_direction"] = direction
    return base.drop(columns="sig_fdr")


def annotate_motif_weights(
    pairs: pd.DataFrame,
    seed_assignments: pd.DataFrame,
    partner_projection: pd.DataFrame,
    *,
    min_explained_energy: float = 0.5,
    motifs: Sequence[str] = MOTIFS,
) -> pd.DataFrame:
    """Attach continuous seed and partner motif proportions to pair rows."""

    if not 0 <= min_explained_energy <= 1:
        raise ValueError("min_explained_energy must lie in [0, 1]")
    for required in ("source", "target"):
        if required not in pairs:
            raise ValueError(f"pair table missing {required!r}")

    seeds = normalize_motif_weights(seed_assignments)
    seed_columns = ["channel", "x", "y", "z"] + [
        f"proportion_{motif}" for motif in motifs
    ]
    seed_columns = [column for column in seed_columns if column in seeds]
    seeds = seeds[seed_columns].drop_duplicates("channel", keep="first")
    seeds = seeds.rename(
        columns={
            "channel": "source",
            "x": "source_x",
            "y": "source_y",
            "z": "source_z",
            **{
                f"proportion_{motif}": f"source_proportion_{motif}"
                for motif in motifs
            },
        }
    )

    projection_columns = [
        "channel",
        "roi",
        "hemi",
        "x",
        "y",
        "z",
        "explained_energy",
        *[f"proportion_{motif}" for motif in motifs],
    ]
    missing = set(projection_columns).difference(partner_projection.columns)
    if missing:
        raise ValueError(f"partner projection missing columns: {sorted(missing)}")
    partners = partner_projection[projection_columns].drop_duplicates(
        "channel", keep="first"
    )
    partners = partners.rename(
        columns={
            "channel": "target",
            "roi": "target_roi",
            "hemi": "target_projection_hemi",
            "x": "target_x",
            "y": "target_y",
            "z": "target_z",
            **{
                f"proportion_{motif}": f"target_proportion_{motif}"
                for motif in motifs
            },
        }
    )

    output = pairs.merge(seeds, on="source", how="left", validate="many_to_one")
    output = output.merge(
        partners, on="target", how="left", validate="many_to_one"
    )
    source_columns = [f"source_proportion_{motif}" for motif in motifs]
    target_columns = [f"target_proportion_{motif}" for motif in motifs]
    output["seed_motif_available"] = output[source_columns].notna().all(axis=1)
    output["partner_projection_available"] = output[target_columns].notna().all(axis=1)
    output["partner_projection_qc"] = (
        output["partner_projection_available"]
        & output["explained_energy"].ge(min_explained_energy)
    )
    source_xyz = output[["source_x", "source_y", "source_z"]].to_numpy(float)
    target_xyz = output[["target_x", "target_y", "target_z"]].to_numpy(float)
    output["distance_mm"] = np.linalg.norm(source_xyz - target_xyz, axis=1)
    source_values = output[source_columns].to_numpy(float)
    target_values = output[target_columns].to_numpy(float)
    output["motif_match"] = np.sum(source_values * target_values, axis=1)
    output.loc[
        ~(
            output["seed_motif_available"]
            & output["partner_projection_available"]
        ),
        "motif_match",
    ] = np.nan
    return output


def add_distance_bins(
    frame: pd.DataFrame,
    *,
    groups: Sequence[str] = ("subject", "target_roi"),
    n_bins: int = 5,
) -> pd.DataFrame:
    """Add deterministic within-group distance quantile bins."""

    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    output = frame.copy()
    output["distance_bin"] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    for _, index in output.groupby(list(groups), dropna=False, sort=False).groups.items():
        values = output.loc[index, "distance_mm"]
        finite = values[np.isfinite(values)]
        if finite.empty:
            continue
        # Repeated task/phase rows for one anatomical pair have identical
        # distances and must remain in the same stratum.
        ranks = finite.rank(method="average", pct=True)
        bins = np.minimum((ranks * n_bins).apply(np.ceil).astype(int), n_bins) - 1
        output.loc[bins.index, "distance_bin"] = bins.astype("Int64")
    return output


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def weighted_cell_summary(
    pairs: pd.DataFrame,
    metrics: Sequence[MetricColumns],
    *,
    group_columns: Sequence[str] = ENTITY_COLUMNS,
    motifs: Sequence[str] = MOTIFS,
) -> pd.DataFrame:
    """Summarize all nine continuous motif cells within each entity."""

    required = set(group_columns)
    required.update(f"source_proportion_{motif}" for motif in motifs)
    required.update(f"target_proportion_{motif}" for motif in motifs)
    for metric in metrics:
        required.update((metric.sig_fdr, metric.sig_fwer, metric.effect))
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"weighted summary missing columns: {sorted(missing)}")

    eligible = pairs.loc[
        pairs["seed_motif_available"] & pairs["partner_projection_qc"]
    ].copy()
    rows: list[dict[str, object]] = []
    for key, group in eligible.groupby(
        list(group_columns), dropna=False, observed=True, sort=False
    ):
        key_values = key if isinstance(key, tuple) else (key,)
        common = dict(zip(group_columns, key_values))
        source = group[
            [f"source_proportion_{motif}" for motif in motifs]
        ].to_numpy(float)
        target = group[
            [f"target_proportion_{motif}" for motif in motifs]
        ].to_numpy(float)
        for source_index, source_motif in enumerate(motifs):
            for target_index, target_motif in enumerate(motifs):
                weights = source[:, source_index] * target[:, target_index]
                row: dict[str, object] = {
                    **common,
                    "source_motif": source_motif,
                    "partner_motif": target_motif,
                    "within_motif": source_motif == target_motif,
                    "eligible_weight": float(np.nansum(weights)),
                    "n_pairs": int(np.sum(np.isfinite(weights) & (weights > 0))),
                }
                for metric in metrics:
                    for label, column in (
                        ("fdr_hit_rate", metric.sig_fdr),
                        ("fwer_hit_rate", metric.sig_fwer),
                        ("effect", metric.effect),
                    ):
                        row[f"{metric.name}_{label}"] = _weighted_mean(
                            group[column].to_numpy(float), weights
                        )
                rows.append(row)
    return pd.DataFrame(rows)


def balance_tasks_within_subject(
    entity_cells: pd.DataFrame,
    *,
    value_columns: Sequence[str],
) -> pd.DataFrame:
    """Average task-level cells equally inside subject×phase×motif."""

    group_columns = ["subject", "phase", "source_motif", "partner_motif"]
    missing = set(group_columns + list(value_columns)).difference(entity_cells.columns)
    if missing:
        raise ValueError(f"task balancing missing columns: {sorted(missing)}")
    task_group_columns = [
        "subject", "task", "phase", "source_motif", "partner_motif"
    ]
    task_named: dict[str, tuple[str, str]] = {
        column: (column, "mean") for column in value_columns
    }
    task_named["eligible_weight"] = ("eligible_weight", "sum")
    task_named["n_pairs"] = ("n_pairs", "sum")
    task_cells = (
        entity_cells.groupby(task_group_columns, observed=True, sort=True)
        .agg(**task_named)
        .reset_index()
    )
    named: dict[str, tuple[str, str]] = {}
    for column in value_columns:
        named[column] = (column, "mean")
    named["n_tasks"] = ("task", "nunique")
    named["eligible_weight_total"] = ("eligible_weight", "sum")
    named["n_pairs_total"] = ("n_pairs", "sum")
    output = (
        task_cells.groupby(group_columns, observed=True, sort=True)
        .agg(**named)
        .reset_index()
    )
    output["within_motif"] = output["source_motif"].eq(output["partner_motif"])
    return output


def bootstrap_group_cells(
    subject_cells: pd.DataFrame,
    *,
    value_columns: Sequence[str],
    n_bootstrap: int = 10_000,
    random_state: int = 42,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Subject bootstrap CIs and sign-flip p-values for cell summaries."""

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []
    grouping = ["phase", "source_motif", "partner_motif"]
    for key, group in subject_cells.groupby(grouping, observed=True, sort=True):
        row: dict[str, object] = dict(zip(grouping, key))
        row["n_subjects"] = int(group["subject"].nunique())
        for column in value_columns:
            values = group[column].to_numpy(float)
            values = values[np.isfinite(values)]
            row[f"{column}_mean"] = float(np.mean(values)) if len(values) else np.nan
            if not len(values):
                row[f"{column}_ci_low"] = np.nan
                row[f"{column}_ci_high"] = np.nan
                row[f"{column}_p_signflip"] = np.nan
                continue
            samples = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
            boot = np.mean(values[samples], axis=1)
            row[f"{column}_ci_low"] = float(np.quantile(boot, alpha / 2))
            row[f"{column}_ci_high"] = float(np.quantile(boot, 1 - alpha / 2))
            signs = rng.choice((-1.0, 1.0), size=(n_bootstrap, len(values)))
            null = np.mean(values[None, :] * signs, axis=1)
            observed = abs(float(np.mean(values)))
            row[f"{column}_p_signflip"] = float(
                (1 + np.sum(np.abs(null) >= observed)) / (n_bootstrap + 1)
            )
        rows.append(row)
    output = pd.DataFrame(rows)
    for column in value_columns:
        p_column = f"{column}_p_signflip"
        q_column = f"{column}_q_fdr"
        output[q_column] = benjamini_hochberg(output[p_column].to_numpy(float))
    return output


def matched_cross_entity_summary(
    pairs: pd.DataFrame,
    metrics: Sequence[MetricColumns],
    *,
    group_columns: Sequence[str] = ENTITY_COLUMNS,
) -> pd.DataFrame:
    """Compute continuous diagonal-vs-off-diagonal contrasts per entity."""

    eligible = pairs.loc[
        pairs["seed_motif_available"] & pairs["partner_projection_qc"]
    ].copy()
    rows: list[dict[str, object]] = []
    for key, group in eligible.groupby(
        list(group_columns), dropna=False, observed=True, sort=False
    ):
        key_values = key if isinstance(key, tuple) else (key,)
        row: dict[str, object] = dict(zip(group_columns, key_values))
        match = group["motif_match"].to_numpy(float)
        cross = 1.0 - match
        row["matched_weight"] = float(np.nansum(match))
        row["cross_weight"] = float(np.nansum(cross))
        row["n_pairs"] = len(group)
        for metric in metrics:
            for label, column in (
                ("fdr_hit_rate", metric.sig_fdr),
                ("fwer_hit_rate", metric.sig_fwer),
                ("effect", metric.effect),
            ):
                values = group[column].to_numpy(float)
                matched = _weighted_mean(values, match)
                crossed = _weighted_mean(values, cross)
                row[f"{metric.name}_matched_{label}"] = matched
                row[f"{metric.name}_cross_{label}"] = crossed
                row[f"{metric.name}_matched_minus_cross_{label}"] = matched - crossed
        rows.append(row)
    return pd.DataFrame(rows)


def subject_balance_contrasts(
    entity_contrasts: pd.DataFrame,
    *,
    value_columns: Sequence[str],
) -> pd.DataFrame:
    """Equal-weight tasks within each subject and phase for contrasts."""

    task_values = (
        entity_contrasts.groupby(
            ["subject", "task", "phase"], observed=True, sort=True
        )[list(value_columns)]
        .mean()
        .reset_index()
    )
    named = {column: (column, "mean") for column in value_columns}
    named["n_tasks"] = ("task", "nunique")
    return (
        task_values.groupby(["subject", "phase"], observed=True, sort=True)
        .agg(**named)
        .reset_index()
    )


def infer_subject_contrasts(
    subject_contrasts: pd.DataFrame,
    *,
    value_columns: Sequence[str],
    n_permutations: int = 10_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Bootstrap and sign-flip inference for task-balanced subject contrasts."""

    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []
    for phase, group in subject_contrasts.groupby("phase", sort=True):
        row: dict[str, object] = {"phase": phase, "n_subjects": group.subject.nunique()}
        for column in value_columns:
            values = group[column].to_numpy(float)
            values = values[np.isfinite(values)]
            row[f"{column}_mean"] = float(np.mean(values)) if len(values) else np.nan
            if not len(values):
                row[f"{column}_ci_low"] = np.nan
                row[f"{column}_ci_high"] = np.nan
                row[f"{column}_p"] = np.nan
                continue
            indices = rng.integers(0, len(values), size=(n_permutations, len(values)))
            boot = np.mean(values[indices], axis=1)
            row[f"{column}_ci_low"] = float(np.quantile(boot, 0.025))
            row[f"{column}_ci_high"] = float(np.quantile(boot, 0.975))
            signs = rng.choice((-1.0, 1.0), size=(n_permutations, len(values)))
            null = np.mean(values[None, :] * signs, axis=1)
            observed = abs(float(np.mean(values)))
            row[f"{column}_p"] = float(
                (1 + np.sum(np.abs(null) >= observed)) / (n_permutations + 1)
            )
        rows.append(row)
    output = pd.DataFrame(rows)
    for column in value_columns:
        output[f"{column}_q_fdr"] = benjamini_hochberg(
            output[f"{column}_p"].to_numpy(float)
        )
    return output


def stratified_partner_permutation(
    pairs: pd.DataFrame,
    *,
    outcome_column: str,
    strata: Sequence[str] = ("subject", "target_roi", "distance_bin"),
    n_permutations: int = 10_000,
    random_state: int = 42,
) -> Mapping[str, float | int]:
    """Test motif-match weighting after shuffling partner motifs in strata.

    The statistic is the matched-minus-cross weighted outcome.  Target motif
    vectors are permuted jointly so their simplex geometry is preserved.
    This is intended as an anatomy/distance sensitivity analysis, not as the
    primary subject-level inferential unit.
    """

    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    required = {
        outcome_column,
        *strata,
        *[f"source_proportion_{motif}" for motif in MOTIFS],
        *[f"target_proportion_{motif}" for motif in MOTIFS],
    }
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"stratified permutation missing columns: {sorted(missing)}")
    work = pairs.loc[
        pairs["seed_motif_available"]
        & pairs["partner_projection_qc"]
        & np.isfinite(pairs[outcome_column])
    ].copy()
    if work.empty:
        return {"n_pairs": 0, "stat": np.nan, "p": np.nan, "null_mean": np.nan}
    source = work[
        [f"source_proportion_{motif}" for motif in MOTIFS]
    ].to_numpy(float)
    target = work[
        [f"target_proportion_{motif}" for motif in MOTIFS]
    ].to_numpy(float)
    outcome = work[outcome_column].to_numpy(float)

    def statistic(candidate: np.ndarray) -> float:
        match = np.einsum("ij,ij->i", source, candidate)
        cross = 1.0 - match
        return _weighted_mean(outcome, match) - _weighted_mean(outcome, cross)

    observed = statistic(target)
    group_indices = [
        np.asarray(index, dtype=int)
        for index in work.reset_index(drop=True)
        .groupby(list(strata), dropna=False, sort=False)
        .indices.values()
        if len(index) > 1
    ]
    rng = np.random.default_rng(random_state)
    null = np.empty(n_permutations, dtype=float)
    for permutation in range(n_permutations):
        shuffled = target.copy()
        for index in group_indices:
            shuffled[index] = target[rng.permutation(index)]
        null[permutation] = statistic(shuffled)
    p_value = float(
        (1 + np.sum(np.abs(null) >= abs(observed))) / (n_permutations + 1)
    )
    return {
        "n_pairs": int(len(work)),
        "n_strata": int(len(group_indices)),
        "stat": float(observed),
        "p": p_value,
        "null_mean": float(np.mean(null)),
        "null_std": float(np.std(null, ddof=1)),
    }


def stratified_partner_permutation_multi(
    pairs: pd.DataFrame,
    *,
    outcome_columns: Sequence[str],
    strata: Sequence[str] = ("subject", "target_roi", "distance_bin"),
    n_permutations: int = 10_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run one shared sequence of stratified label shuffles for many outcomes."""

    if not outcome_columns:
        raise ValueError("outcome_columns must not be empty")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    required = {
        *outcome_columns,
        *strata,
        *[f"source_proportion_{motif}" for motif in MOTIFS],
        *[f"target_proportion_{motif}" for motif in MOTIFS],
    }
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"stratified permutation missing columns: {sorted(missing)}")
    finite_outcomes = np.isfinite(pairs[list(outcome_columns)].to_numpy(float)).all(axis=1)
    work = pairs.loc[
        pairs["seed_motif_available"]
        & pairs["partner_projection_qc"]
        & finite_outcomes
    ].reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(
            {
                "outcome": list(outcome_columns), "n_pairs": 0,
                "n_strata": 0, "stat": np.nan, "p": np.nan,
                "null_mean": np.nan, "null_std": np.nan,
            }
        )
    source = work[
        [f"source_proportion_{motif}" for motif in MOTIFS]
    ].to_numpy(float)
    target = work[
        [f"target_proportion_{motif}" for motif in MOTIFS]
    ].to_numpy(float)
    outcomes = work[list(outcome_columns)].to_numpy(float)

    def statistics(candidate: np.ndarray) -> np.ndarray:
        match = np.einsum("ij,ij->i", source, candidate)
        cross = 1.0 - match
        matched = outcomes.T @ match / np.sum(match)
        crossed = outcomes.T @ cross / np.sum(cross)
        return matched - crossed

    observed = statistics(target)
    group_indices = [
        np.asarray(index, dtype=int)
        for index in work.groupby(list(strata), dropna=False, sort=False).indices.values()
        if len(index) > 1
    ]
    rng = np.random.default_rng(random_state)
    null = np.empty((n_permutations, len(outcome_columns)), dtype=float)
    for permutation in range(n_permutations):
        shuffled = target.copy()
        for index in group_indices:
            shuffled[index] = target[rng.permutation(index)]
        null[permutation] = statistics(shuffled)
    p_values = (
        1 + np.sum(np.abs(null) >= np.abs(observed)[None, :], axis=0)
    ) / (n_permutations + 1)
    return pd.DataFrame(
        {
            "outcome": list(outcome_columns),
            "n_pairs": len(work),
            "n_strata": len(group_indices),
            "stat": observed,
            "p": p_values,
            "null_mean": np.mean(null, axis=0),
            "null_std": np.std(null, axis=0, ddof=1),
        }
    )


def weighted_direction_summary(
    pairs: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ENTITY_COLUMNS,
    direction_column: str = "lag_direction",
    lag_column: str = "strict_peak_lag_s",
) -> pd.DataFrame:
    """Continuous motif-cell summaries of strict lag direction."""

    eligible = pairs.loc[
        pairs["seed_motif_available"] & pairs["partner_projection_qc"]
    ].copy()
    rows: list[dict[str, object]] = []
    directions = ("insula_leads", "partner_leads", "unresolved", "ambiguous")
    for key, group in eligible.groupby(
        list(group_columns), dropna=False, observed=True, sort=False
    ):
        key_values = key if isinstance(key, tuple) else (key,)
        common = dict(zip(group_columns, key_values))
        source = group[[f"source_proportion_{m}" for m in MOTIFS]].to_numpy(float)
        target = group[[f"target_proportion_{m}" for m in MOTIFS]].to_numpy(float)
        for i, source_motif in enumerate(MOTIFS):
            for j, partner_motif in enumerate(MOTIFS):
                weights = source[:, i] * target[:, j]
                strict = group[direction_column].isin(
                    ["insula_leads", "partner_leads"]
                ).to_numpy()
                strict_weight = float(np.sum(weights[strict]))
                row: dict[str, object] = {
                    **common,
                    "source_motif": source_motif,
                    "partner_motif": partner_motif,
                    "strict_weight": strict_weight,
                    "weighted_lag_s": _weighted_mean(
                        group[lag_column].to_numpy(float),
                        weights * strict,
                    ),
                }
                for direction in directions:
                    mask = group[direction_column].eq(direction).to_numpy()
                    row[f"{direction}_weight"] = float(np.sum(weights[mask]))
                    row[f"{direction}_fraction"] = (
                        float(np.sum(weights[mask]) / np.sum(weights))
                        if np.sum(weights) > 0
                        else np.nan
                    )
                rows.append(row)
    return pd.DataFrame(rows)
