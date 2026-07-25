"""Signed HGA cross-correlation with trial-shuffle cluster inference."""

from __future__ import annotations

from pathlib import Path
import json
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from .config import ConnectivityConfig
from .permutation import benjamini_hochberg
from .result import MetricResult


def lagged_cross_trial_pearson_z(
    source_trials: np.ndarray,
    target_trials: np.ndarray,
    lags: Iterable[int],
) -> np.ndarray:
    """Cross-trial lagged Pearson correlations in Fisher-z units.

    The output has shape ``(lag, source_trial, target_trial)``.  With the
    ``correlate(source, target)`` convention implemented here, a negative lag
    means that the source pattern occurs earlier.
    """

    source = np.asarray(source_trials, dtype=float)
    target = np.asarray(target_trials, dtype=float)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source_trials and target_trials must be 2D")
    if source.shape != target.shape:
        raise ValueError("source and target must have the same shape")
    n_trials, n_time = source.shape
    lag_values = np.asarray(list(lags), dtype=int)
    output = np.full(
        (lag_values.size, n_trials, n_trials), np.nan, dtype=np.float32
    )
    eps = np.finfo(float).eps
    for lag_index, lag in enumerate(lag_values):
        if abs(lag) >= n_time - 2:
            continue
        if lag < 0:
            x = source[:, : n_time + lag]
            y = target[:, -lag:]
        elif lag > 0:
            x = source[:, lag:]
            y = target[:, : n_time - lag]
        else:
            x = source
            y = target
        x_centered = x - x.mean(axis=1, keepdims=True)
        y_centered = y - y.mean(axis=1, keepdims=True)
        x_norm = np.linalg.norm(x_centered, axis=1)
        y_norm = np.linalg.norm(y_centered, axis=1)
        denominator = x_norm[:, None] * y_norm[None, :]
        correlation = np.divide(
            x_centered @ y_centered.T,
            denominator,
            out=np.full((n_trials, n_trials), np.nan, dtype=float),
            where=denominator > eps,
        )
        correlation = np.clip(correlation, -1.0 + 1e-7, 1.0 - 1e-7)
        output[lag_index] = np.arctanh(correlation).astype(np.float32)
    return output


def _max_cluster_mass(
    curves: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Maximum two-sided contiguous exceedance mass for many curves."""

    values = np.atleast_2d(np.asarray(curves, dtype=float))
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    running_positive = np.zeros(values.shape[0], dtype=float)
    running_negative = np.zeros(values.shape[0], dtype=float)
    maximum = np.zeros(values.shape[0], dtype=float)
    for lag_index in range(values.shape[1]):
        positive = values[:, lag_index] - upper[lag_index]
        negative = lower[lag_index] - values[:, lag_index]
        running_positive = np.where(
            positive > 0, running_positive + positive, 0.0
        )
        running_negative = np.where(
            negative > 0, running_negative + negative, 0.0
        )
        maximum = np.maximum(
            maximum, np.maximum(running_positive, running_negative)
        )
    return maximum


def _observed_clusters(
    score_curve: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    lag_times: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sign, mask, excess in (
        ("positive", score_curve > upper, score_curve - upper),
        ("negative", score_curve < lower, lower - score_curve),
    ):
        start: int | None = None
        for index in range(mask.size + 1):
            active = index < mask.size and bool(mask[index])
            if active and start is None:
                start = index
            elif not active and start is not None:
                stop = index
                segment = excess[start:stop]
                peak_local = int(np.argmax(segment))
                peak_index = start + peak_local
                rows.append(
                    {
                        "sign": sign,
                        "start_index": start,
                        "stop_index": stop,
                        "lag_start_s": float(lag_times[start]),
                        "lag_stop_s": float(lag_times[stop - 1]),
                        "peak_lag_s": float(lag_times[peak_index]),
                        "cluster_mass": float(np.sum(segment)),
                    }
                )
                start = None
    return rows


def compute_xcorr(
    hga_data: np.ndarray,
    sfreq: float,
    pair_frame: pd.DataFrame,
    permutations: np.ndarray,
    config: ConnectivityConfig,
    *,
    scratch_dir: str | Path | None = None,
) -> MetricResult:
    """Compute signed HGA xcorr for one aligned analysis entity."""

    data = np.asarray(hga_data, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError("hga_data must have shape (trial, channel, time)")
    n_trials, _, n_time = data.shape
    if permutations.shape[1] != n_trials:
        raise ValueError("permutations do not match trial count")
    max_lag = int(round(config.max_lag_s * sfreq))
    if max_lag * 2 > n_time:
        raise ValueError(
            "max_lag_s must not exceed half of the phase window"
        )
    lags = np.arange(-max_lag, max_lag + 1, dtype=int)
    lag_times = lags.astype(float) / float(sfreq)
    n_pairs = len(pair_frame)
    n_perm = permutations.shape[0]

    observed_z = np.empty((n_pairs, lags.size), dtype=np.float32)
    scratch_root = (
        Path(scratch_dir)
        if scratch_dir is not None
        else Path(tempfile.mkdtemp(prefix="insula-xcorr-"))
    )
    scratch_root.mkdir(parents=True, exist_ok=True)
    null_path = scratch_root / "xcorr_null.dat"
    null_z = np.memmap(
        null_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_perm, n_pairs, lags.size),
    )
    identity = np.arange(n_trials)

    for pair_index, pair in pair_frame.iterrows():
        source = data[:, int(pair["source_index"]), :]
        target = data[:, int(pair["target_index"]), :]
        cross_trial_z = lagged_cross_trial_pearson_z(source, target, lags)
        observed_z[pair_index] = np.nanmean(
            cross_trial_z[:, identity, identity], axis=1
        )
        for lag_index in range(lags.size):
            matrix = cross_trial_z[lag_index]
            selected = matrix[identity[None, :], permutations]
            null_z[:, pair_index, lag_index] = np.nanmean(selected, axis=1)
    null_z.flush()

    null_mean = np.empty_like(observed_z)
    null_lower = np.empty_like(observed_z)
    null_upper = np.empty_like(observed_z)
    observed_score = np.empty_like(observed_z)
    null_pair_mass = np.empty((n_perm, n_pairs), dtype=np.float32)
    observed_pair_mass = np.empty(n_pairs, dtype=np.float32)
    cluster_rows: list[dict[str, object]] = []

    for pair_index in range(n_pairs):
        pair_null = np.asarray(null_z[:, pair_index, :], dtype=float)
        center = np.nanmean(pair_null, axis=0)
        scale = np.nanstd(pair_null, axis=0, ddof=1)
        valid_scale = np.where(scale > np.finfo(float).eps, scale, np.nan)
        null_mean[pair_index] = center
        null_lower[pair_index] = np.nanquantile(
            pair_null, config.alpha / 2.0, axis=0
        )
        null_upper[pair_index] = np.nanquantile(
            pair_null, 1.0 - config.alpha / 2.0, axis=0
        )
        score_null = (pair_null - center) / valid_scale
        score_observed = (observed_z[pair_index] - center) / valid_scale
        lower_score = np.nanquantile(
            score_null, config.alpha / 2.0, axis=0
        )
        upper_score = np.nanquantile(
            score_null, 1.0 - config.alpha / 2.0, axis=0
        )
        observed_score[pair_index] = score_observed
        null_pair_mass[:, pair_index] = _max_cluster_mass(
            score_null, lower_score, upper_score
        )
        observed_pair_mass[pair_index] = _max_cluster_mass(
            score_observed, lower_score, upper_score
        )[0]
        rows = _observed_clusters(
            score_observed, lower_score, upper_score, lag_times
        )
        pair_null_mass = null_pair_mass[:, pair_index]
        for row in rows:
            row["pair_index"] = pair_index
            row["pair_id"] = pair_frame.iloc[pair_index]["pair_id"]
            row["source"] = pair_frame.iloc[pair_index]["source"]
            row["target"] = pair_frame.iloc[pair_index]["target"]
            row["p_pair_cluster"] = (
                1.0 + np.sum(pair_null_mass >= row["cluster_mass"])
            ) / (n_perm + 1.0)
            cluster_rows.append(row)

    global_null_max = np.max(null_pair_mass, axis=1)
    for row in cluster_rows:
        row["p_fwer_cluster"] = (
            1.0 + np.sum(global_null_max >= row["cluster_mass"])
        ) / (n_perm + 1.0)
    p_uncorrected = (
        1.0
        + np.sum(
            null_pair_mass >= observed_pair_mass[None, :],
            axis=0,
        )
    ) / (n_perm + 1.0)
    p_fwer = (
        1.0
        + np.sum(global_null_max[:, None] >= observed_pair_mass[None, :], axis=0)
    ) / (n_perm + 1.0)
    q_fdr = benjamini_hochberg(p_uncorrected)

    finite_score = np.where(
        np.isfinite(observed_score), np.abs(observed_score), -np.inf
    )
    peak_index = np.argmax(finite_score, axis=1)
    no_peak = np.all(~np.isfinite(observed_score), axis=1)
    output = pair_frame.copy()
    output["metric"] = "xcorr"
    output["stat"] = observed_pair_mass
    output["peak_lag_s"] = lag_times[peak_index]
    output["peak_r"] = np.tanh(
        observed_z[np.arange(n_pairs), peak_index]
    )
    output.loc[no_peak, ["peak_lag_s", "peak_r"]] = np.nan
    output["null_mean_stat"] = np.mean(null_pair_mass, axis=0)
    output["p_uncorrected"] = p_uncorrected
    output["q_fdr"] = q_fdr
    output["p_fwer_maxstat"] = p_fwer
    output["sig_fdr"] = q_fdr < config.alpha
    output["sig_fwer"] = p_fwer < config.alpha
    output["qc_pass"] = np.isfinite(observed_pair_mass)
    output["lag_convention"] = "negative_lag_source_leads"
    cluster_table = pd.DataFrame(
        cluster_rows,
        columns=[
            "pair_index",
            "pair_id",
            "source",
            "target",
            "sign",
            "start_index",
            "stop_index",
            "lag_start_s",
            "lag_stop_s",
            "peak_lag_s",
            "cluster_mass",
            "p_pair_cluster",
            "p_fwer_cluster",
        ],
    )
    significant_intervals: dict[int, list[dict[str, object]]] = {}
    if not cluster_table.empty:
        significant = cluster_table.loc[
            cluster_table["p_pair_cluster"] < config.alpha
        ]
        for pair_index, group in significant.groupby("pair_index"):
            significant_intervals[int(pair_index)] = group[
                ["sign", "lag_start_s", "lag_stop_s", "p_pair_cluster"]
            ].to_dict(orient="records")
    output["n_observed_clusters"] = (
        cluster_table.groupby("pair_index").size()
        .reindex(np.arange(n_pairs), fill_value=0)
        .to_numpy()
    )
    output["has_significant_cluster"] = [
        bool(significant_intervals.get(index)) for index in range(n_pairs)
    ]
    output["significant_lag_intervals_s"] = [
        json.dumps(significant_intervals.get(index, []), separators=(",", ":"))
        for index in range(n_pairs)
    ]

    detail_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "observed_fisher_z": (("pair", "lag"), observed_z),
        "observed_r": (("pair", "lag"), np.tanh(observed_z)),
        "observed_score": (("pair", "lag"), observed_score),
        "null_mean_fisher_z": (("pair", "lag"), null_mean),
        "null_lower_fisher_z": (("pair", "lag"), null_lower),
        "null_upper_fisher_z": (("pair", "lag"), null_upper),
    }
    if config.save_full_null:
        detail_vars["null_fisher_z"] = (
            ("permutation", "pair", "lag"),
            np.asarray(null_z).copy(),
        )
    detail = xr.Dataset(
        data_vars=detail_vars,
        coords={
            "pair": np.arange(n_pairs, dtype=np.int32),
            "lag": lag_times.astype(np.float32),
            "pair_id": ("pair", pair_frame["pair_id"].astype(str).to_numpy()),
            "source": ("pair", pair_frame["source"].astype(str).to_numpy()),
            "target": ("pair", pair_frame["target"].astype(str).to_numpy()),
        },
        attrs={
            "metric": "xcorr",
            "lag_convention": "negative_lag_source_leads",
            "tail": "two-sided",
            "n_perm": int(n_perm),
        },
    )
    del null_z
    if null_path.exists():
        null_path.unlink()
    return MetricResult(
        metric="xcorr",
        pair_table=output,
        detail=detail,
        auxiliary_tables={"clusters": cluster_table},
        runtime_metadata={
            "n_pairs": n_pairs,
            "n_lags": int(lags.size),
            "n_trials": n_trials,
        },
    )
