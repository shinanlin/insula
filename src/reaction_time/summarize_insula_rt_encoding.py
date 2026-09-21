"""Describe Go-aligned HGA–RT encoding for assigned pre-Go RT hits.

Post-selection descriptive analysis. Electrodes retain the FWER-significant,
positive-OOF-r Go window that starts before Go onset and has the strongest
prediction score. Trial-wise mean HGA in that window is correlated with
log RT. Negative r means higher HGA with shorter RT.

Subject-level Wilcoxon on encoding r is the primary test. A trial-level
mixed model (rt_log ~ hga_z + (1|subject)) is reported as a supplement.
Both use the same selected windows and are not independent confirmation.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, wilcoxon

from src.reaction_time.insula_rt_data import NoStrictInsulaError, load_phase_data
from src.reaction_time.summarize_insula_rt_direction import (
    DEFAULT_ASSIGNMENTS,
    DEFAULT_BIDS_ROOTS,
    DEFAULT_OUTPUT_ROOT,
    _decode,
    mean_hga_rt_direction,
)


MIN_TRIALS = 16
CLUSTER_ORDER = ("sustain", "motor", "sensory")
INFERENCE_NOTE = "post-selection descriptive; not independent confirmation"


def window_mean_hga(
    X: np.ndarray,
    times: np.ndarray,
    *,
    channel_index: int,
    window_start: float,
    window_end: float,
) -> np.ndarray:
    """Trial-wise mean HGA inside ``[window_start, window_end]``."""

    times = np.asarray(times, dtype=float)
    sample_mask = (times >= float(window_start) - 1e-10) & (
        times <= float(window_end) + 1e-10
    )
    if not sample_mask.any():
        raise ValueError("Encoding window contains no HGA samples")
    return np.mean(
        np.asarray(X, dtype=float)[:, int(channel_index), :][:, sample_mask],
        axis=1,
    )


def median_split_traces(
    trial_hga: np.ndarray,
    rt_log: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return fast/slow mean time courses from a within-subject median split.

    ``trial_hga`` has shape (n_trials, n_times). Trials with non-finite RT or
    non-finite HGA at any timepoint are dropped. Ties at the median RT are
    assigned to the slow group (``rt_log >= median``).
    """

    trial_hga = np.asarray(trial_hga, dtype=float)
    rt_log = np.asarray(rt_log, dtype=float)
    if trial_hga.ndim != 2:
        raise ValueError(f"trial_hga must be 2-D, got {trial_hga.shape}")
    keep = np.isfinite(rt_log) & np.isfinite(trial_hga).all(axis=1)
    if keep.sum() < 2:
        n_times = trial_hga.shape[1]
        nan = np.full(n_times, np.nan, dtype=float)
        return nan, nan, int(keep.sum()), 0
    hga = trial_hga[keep]
    rt = rt_log[keep]
    median = float(np.median(rt))
    fast = rt < median
    slow = ~fast
    if not fast.any() or not slow.any():
        n_times = hga.shape[1]
        nan = np.full(n_times, np.nan, dtype=float)
        return nan, nan, int(fast.sum()), int(slow.sum())
    return (
        np.mean(hga[fast], axis=0),
        np.mean(hga[slow], axis=0),
        int(fast.sum()),
        int(slow.sum()),
    )


def aligned_fast_slow_traces(
    data,
    *,
    task: str,
    subject: str,
    retained_clusters: list[str],
    by_cluster: dict[str, list[str]],
    min_trials: int,
) -> list[dict[str, object]]:
    """Median-split HGA traces for one aligned phase and the assigned electrodes."""

    rows: list[dict[str, object]] = []
    target = data.trial_meta["rt_log"].to_numpy(dtype=float)
    channel_index = {
        channel: index
        for index, channel in enumerate(data.channel_meta["channel"].astype(str))
    }
    for cluster in retained_clusters:
        channels = [
            channel for channel in by_cluster[cluster] if channel in channel_index
        ]
        if not channels:
            continue
        channel_indices = [channel_index[channel] for channel in channels]
        trial_hga = np.mean(data.X[:, channel_indices, :], axis=1)
        keep = np.isfinite(target) & np.isfinite(trial_hga).all(axis=1)
        if int(keep.sum()) < int(min_trials):
            continue
        fast, slow, n_fast, n_slow = median_split_traces(trial_hga, target)
        for time_index, time in enumerate(data.times):
            rows.append(
                {
                    "task": task,
                    "subject": subject,
                    "functional_cluster": cluster,
                    "time": float(time),
                    "hga_fast": float(fast[time_index]),
                    "hga_slow": float(slow[time_index]),
                    "n_fast": n_fast,
                    "n_slow": n_slow,
                    "n_electrodes": len(channels),
                }
            )
    return rows


def collapse_electrode_r_to_subject(
    electrode: pd.DataFrame,
    *,
    value_col: str = "mean_hga_log_rt_r",
) -> pd.DataFrame:
    """Median electrode encoding r within task × subject × functional cluster."""

    required = {"task", "subject", "functional_cluster", value_col}
    missing = required.difference(electrode.columns)
    if missing:
        raise ValueError(f"Missing columns for subject collapse: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    grouped = electrode.groupby(
        ["task", "subject", "functional_cluster"], sort=False
    )
    for (task, subject, cluster), frame in grouped:
        values = frame[value_col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        rows.append(
            {
                "task": task,
                "subject": subject,
                "functional_cluster": cluster,
                "n_electrodes": int(len(frame)),
                "mean_hga_log_rt_r": float(np.median(values)),
                "n_trials": int(frame["n_trials"].median())
                if "n_trials" in frame.columns
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[keep]
    y = np.asarray(y, dtype=float)[keep]
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(pearsonr(x, y)[0])


def one_sample_t(values: np.ndarray) -> float:
    """t for mean(values) against 0. Negative when encoding r < 0."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n < 2:
        return np.nan
    se = float(np.std(values, ddof=1) / np.sqrt(n))
    if se == 0:
        return np.nan
    return float(np.mean(values) / se)


def subject_encoding_r(trials: pd.DataFrame) -> np.ndarray:
    """One encoding r per subject: median of task-level Pearson r(hga_z, rt_log)."""
    required = {"task", "subject", "hga_z", "rt_log"}
    missing = required.difference(trials.columns)
    if missing:
        raise ValueError(f"Missing columns for subject encoding r: {sorted(missing)}")
    frame = trials.loc[
        np.isfinite(trials["hga_z"]) & np.isfinite(trials["rt_log"]),
        ["task", "subject", "hga_z", "rt_log"],
    ]
    if frame.empty:
        return np.asarray([], dtype=float)
    unit_r: list[tuple[str, float]] = []
    for (task, subject), group in frame.groupby(["task", "subject"], sort=False):
        del task
        unit_r.append(
            (
                str(subject),
                _pearson_r(
                    group["hga_z"].to_numpy(),
                    group["rt_log"].to_numpy(),
                ),
            )
        )
    by_subject: dict[str, list[float]] = {}
    for subject, value in unit_r:
        if not np.isfinite(value):
            continue
        by_subject.setdefault(subject, []).append(value)
    return np.asarray(
        [float(np.median(values)) for values in by_subject.values()],
        dtype=float,
    )


def permute_subject_encoding_t(
    trials: pd.DataFrame,
    *,
    n_perm: int = 5000,
    random_state: int = 42,
) -> dict[str, object]:
    """Permutation null for the one-sample t of subject-median encoding r.

    Shuffle ``rt_log`` within each task × subject. This is still post-selection:
    windows were chosen from the observed RT prediction, then frozen.
    """
    required = {"task", "subject", "hga_z", "rt_log"}
    missing = required.difference(trials.columns)
    if missing:
        raise ValueError(f"Missing columns for encoding permutation: {sorted(missing)}")
    frame = trials.loc[
        np.isfinite(trials["hga_z"]) & np.isfinite(trials["rt_log"]),
        ["task", "subject", "hga_z", "rt_log"],
    ].copy()
    observed_r = subject_encoding_r(frame)
    observed_t = one_sample_t(observed_r)
    if frame.empty or not np.isfinite(observed_t):
        return {
            "observed_t": observed_t,
            "observed_r": observed_r,
            "null_t": np.asarray([], dtype=float),
            "p_left": np.nan,
            "n_perm": int(n_perm),
            "n_subjects": int(observed_r.size),
        }

    keys = frame["task"].astype(str) + "\t" + frame["subject"].astype(str)
    codes, _uniques = pd.factorize(keys, sort=False)
    hga = frame["hga_z"].to_numpy(dtype=float)
    rt = frame["rt_log"].to_numpy(dtype=float)
    subjects = frame["subject"].astype(str).to_numpy()
    n_groups = int(codes.max()) + 1
    group_index = [np.flatnonzero(codes == g) for g in range(n_groups)]
    group_subject = [str(subjects[idx[0]]) for idx in group_index]

    def t_from_rt(rt_arr: np.ndarray) -> float:
        by_subject: dict[str, list[float]] = {}
        for idx, subject in zip(group_index, group_subject):
            value = _pearson_r(hga[idx], rt_arr[idx])
            if not np.isfinite(value):
                continue
            by_subject.setdefault(subject, []).append(value)
        values = np.asarray(
            [float(np.median(parts)) for parts in by_subject.values()],
            dtype=float,
        )
        return one_sample_t(values)

    rng = np.random.default_rng(int(random_state))
    null_t = np.empty(int(n_perm), dtype=float)
    for perm_i in range(int(n_perm)):
        rt_perm = rt.copy()
        for idx in group_index:
            rt_perm[idx] = rng.permutation(rt[idx])
        null_t[perm_i] = t_from_rt(rt_perm)

    finite_null = null_t[np.isfinite(null_t)]
    if finite_null.size == 0 or not np.isfinite(observed_t):
        p_left = np.nan
    else:
        p_left = (1.0 + float(np.sum(finite_null <= observed_t))) / (
            1.0 + float(finite_null.size)
        )
    return {
        "observed_t": observed_t,
        "observed_r": observed_r,
        "null_t": null_t,
        "p_left": p_left,
        "n_perm": int(n_perm),
        "n_subjects": int(observed_r.size),
    }


def _cluster_label(value: object) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return "unassigned"
    return str(value)


def _peak_prego_windows(path: Path) -> list[dict[str, object]]:
    """Peak FWER+r>0 Go window that starts before Go for each electrode."""

    with h5py.File(path, "r") as h5:
        phase = str(h5.attrs["phase"])
        if phase != "Go":
            raise ValueError(f"Expected Go HDF5, got phase={phase!r} in {path}")
        start = h5["windows/start"][:]
        end = h5["windows/end"][:]
        score = h5["scores/r"][:]
        valid = (
            h5["inference/sig_mask_fwer"][:].astype(bool)
            & np.isfinite(score)
            & (score > 0)
            & (start < 0)[None, :]
        )
        output: list[dict[str, object]] = []
        for channel_index, channel in enumerate(_decode(h5["channels/channel"][:])):
            candidates = np.flatnonzero(valid[channel_index])
            if candidates.size == 0:
                continue
            peak_index = int(candidates[np.argmax(score[channel_index, candidates])])
            output.append(
                {
                    "phase": phase,
                    "channel": channel,
                    "window_index": peak_index,
                    "window_start": float(start[peak_index]),
                    "window_end": float(end[peak_index]),
                    "prediction_r": float(score[channel_index, peak_index]),
                }
            )
        return output


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    keep = np.isfinite(values)
    if keep.sum() < 2 or np.std(values[keep]) == 0:
        return out
    out[keep] = (values[keep] - np.mean(values[keep])) / np.std(values[keep])
    return out


def _wilcoxon_row(
    subject_r: pd.DataFrame,
    *,
    scope_type: str,
    scope: str,
) -> dict[str, object]:
    # One value per subject: median across tasks (and electrodes already
    # collapsed into subject_r rows).
    if subject_r.empty:
        values = np.asarray([], dtype=float)
    else:
        values = (
            subject_r.groupby("subject", sort=False)["mean_hga_log_rt_r"]
            .median()
            .to_numpy(dtype=float)
        )
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return {
            "test": "wilcoxon",
            "scope_type": scope_type,
            "scope": scope,
            "n_subjects": 0,
            "n_negative_subjects": 0,
            "median_subject_r": np.nan,
            "statistic": np.nan,
            "p_descriptive": np.nan,
            "coef": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "inference_note": INFERENCE_NOTE,
        }
    negative = int((values < 0).sum())
    if n < 2 or np.allclose(values, 0.0):
        wilcoxon_p = np.nan
        statistic = np.nan
    else:
        result = wilcoxon(values, alternative="less")
        wilcoxon_p = float(result.pvalue)
        statistic = float(result.statistic)
    return {
        "test": "wilcoxon",
        "scope_type": scope_type,
        "scope": scope,
        "n_subjects": n,
        "n_negative_subjects": negative,
        "median_subject_r": float(np.median(values)),
        "statistic": statistic,
        "p_descriptive": wilcoxon_p,
        "coef": float(np.median(values)),
        "ci_low": np.nan,
        "ci_high": np.nan,
        "inference_note": INFERENCE_NOTE,
    }


def _fit_lmm(
    trials: pd.DataFrame,
    *,
    formula: str,
    scope_type: str,
    scope: str,
) -> dict[str, object]:
    empty = {
        "test": "lmm",
        "scope_type": scope_type,
        "scope": scope,
        "n_subjects": 0,
        "n_negative_subjects": np.nan,
        "median_subject_r": np.nan,
        "statistic": np.nan,
        "p_descriptive": np.nan,
        "coef": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "formula": formula,
        "n_trials": 0,
        "inference_note": INFERENCE_NOTE,
    }
    if trials.empty or trials["subject"].nunique() < 2:
        empty["n_subjects"] = int(trials["subject"].nunique()) if not trials.empty else 0
        empty["n_trials"] = int(len(trials))
        return empty
    frame = trials.loc[
        np.isfinite(trials["hga_z"]) & np.isfinite(trials["rt_log"])
    ].copy()
    empty["n_subjects"] = int(frame["subject"].nunique())
    empty["n_trials"] = int(len(frame))
    if len(frame) < MIN_TRIALS or frame["subject"].nunique() < 2:
        return empty
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.mixedlm(formula, data=frame, groups=frame["subject"]).fit(
                reml=True, method="lbfgs"
            )
        coef = float(fit.params["hga_z"])
        se = float(fit.bse["hga_z"])
        p_value = float(fit.pvalues["hga_z"])
        return {
            "test": "lmm",
            "scope_type": scope_type,
            "scope": scope,
            "n_subjects": int(frame["subject"].nunique()),
            "n_negative_subjects": np.nan,
            "median_subject_r": np.nan,
            "statistic": coef / se if se > 0 else np.nan,
            "p_descriptive": p_value,
            "coef": coef,
            "ci_low": coef - 1.96 * se,
            "ci_high": coef + 1.96 * se,
            "formula": formula,
            "n_trials": int(len(frame)),
            "inference_note": INFERENCE_NOTE,
        }
    except Exception as exc:  # noqa: BLE001 - surface fit failures in CSV
        empty["inference_note"] = f"{INFERENCE_NOTE}; lmm_failed: {exc}"
        return empty


def summarize_hga_rt_encoding(
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    *,
    assignments_path: Path | str = DEFAULT_ASSIGNMENTS,
    bids_roots: dict[str, Path] = DEFAULT_BIDS_ROOTS,
    min_trials: int = MIN_TRIALS,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Write Go-aligned HGA–RT encoding summaries under ``summaries/``.

    Also writes Delay- and Response-aligned fast/slow traces for the same
    assigned pre-Go electrodes (``hga_rt_delay_traces.csv``,
    ``hga_rt_response_traces.csv``).
    """

    output_root = Path(output_root)
    assignments = pd.read_csv(assignments_path)
    cluster_by_channel = (
        assignments.drop_duplicates("channel")
        .set_index("channel")["functional_cluster"]
        .map(_cluster_label)
        .to_dict()
    )

    electrode_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    delay_trace_rows: list[dict[str, object]] = []
    response_trace_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []

    for subject_dir in sorted(output_root.glob("task-*/sub-*")):
        go_paths = list(subject_dir.glob("*proc-Go*_rt-ridge.h5"))
        if len(go_paths) != 1:
            continue
        go_path = go_paths[0]
        selected = _peak_prego_windows(go_path)
        if not selected:
            continue
        with h5py.File(go_path, "r") as h5:
            task = str(h5.attrs["task"])
            subject = str(h5.attrs["subject"])
            expected_target = h5["trials/rt_log"][:]
            expected_uids = _decode(h5["trials/trial_uid"][:])

        assigned = []
        for row in selected:
            cluster = cluster_by_channel.get(str(row["channel"]), "unassigned")
            if cluster not in CLUSTER_ORDER:
                continue
            assigned.append({**row, "functional_cluster": cluster})
        if not assigned:
            continue

        try:
            data = load_phase_data(
                bids_roots[task],
                task=task,
                subject=subject,
                phase="Go",
                description="Repeat",
                band="highgamma",
                ref="bipolar",
                atlas="hammers",
            )
        except (FileNotFoundError, NoStrictInsulaError) as exc:
            print(f"skip {task}/{subject}: {exc}")
            continue

        target = data.trial_meta["rt_log"].to_numpy(dtype=float)
        trial_uids = data.trial_meta["trial_uid"].astype(str).tolist()
        if target.shape != expected_target.shape or not np.allclose(
            target, expected_target, atol=1e-10, rtol=0
        ):
            raise RuntimeError(f"RT trial alignment changed for {go_path}")
        if trial_uids != expected_uids:
            raise RuntimeError(f"trial_uid alignment changed for {go_path}")

        channel_index = {
            channel: index
            for index, channel in enumerate(data.channel_meta["channel"].astype(str))
        }

        # Electrode-level encoding and per-electrode z-scored window HGA.
        electrode_amplitude: dict[str, np.ndarray] = {}
        for row in assigned:
            channel = str(row["channel"])
            if channel not in channel_index:
                continue
            correlation, p_value, slope, n_trials = mean_hga_rt_direction(
                data.X,
                data.times,
                target,
                channel_index=channel_index[channel],
                window_start=float(row["window_start"]),
                window_end=float(row["window_end"]),
            )
            amplitude = window_mean_hga(
                data.X,
                data.times,
                channel_index=channel_index[channel],
                window_start=float(row["window_start"]),
                window_end=float(row["window_end"]),
            )
            electrode_amplitude[channel] = amplitude
            electrode_rows.append(
                {
                    "task": task,
                    "subject": subject,
                    "phase": "Go",
                    "channel": channel,
                    "functional_cluster": row["functional_cluster"],
                    "window_index": row["window_index"],
                    "window_start": row["window_start"],
                    "window_end": row["window_end"],
                    "prediction_r": row["prediction_r"],
                    "n_trials": n_trials,
                    "mean_hga_log_rt_r": correlation,
                    "mean_hga_log_rt_p_descriptive": p_value,
                    "raw_slope_log_rt_per_hga": slope,
                    "direction": (
                        "higher_HGA_shorter_RT"
                        if np.isfinite(correlation) and correlation < 0
                        else "higher_HGA_longer_RT"
                    ),
                    "source_h5": str(go_path),
                }
            )

        # Subject × cluster traces and trial table.
        by_cluster: dict[str, list[str]] = {name: [] for name in CLUSTER_ORDER}
        for row in assigned:
            channel = str(row["channel"])
            if channel in electrode_amplitude:
                by_cluster[str(row["functional_cluster"])].append(channel)

        retained_clusters: list[str] = []
        for cluster, channels in by_cluster.items():
            if not channels:
                continue
            # Average z-scored window HGA across electrodes for LMM / subject r
            # companion; traces use full-epoch electrode-averaged HGA.
            z_stack = np.column_stack(
                [_zscore(electrode_amplitude[channel]) for channel in channels]
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                hga_z = np.nanmean(z_stack, axis=1)
            finite_trials = np.isfinite(target) & np.isfinite(hga_z)
            n_finite = int(finite_trials.sum())
            if n_finite < int(min_trials):
                continue
            retained_clusters.append(cluster)

            channel_indices = [channel_index[channel] for channel in channels]
            trial_hga = np.mean(data.X[:, channel_indices, :], axis=1)
            fast, slow, n_fast, n_slow = median_split_traces(trial_hga, target)
            for time_index, time in enumerate(data.times):
                trace_rows.append(
                    {
                        "task": task,
                        "subject": subject,
                        "functional_cluster": cluster,
                        "time": float(time),
                        "hga_fast": float(fast[time_index]),
                        "hga_slow": float(slow[time_index]),
                        "n_fast": n_fast,
                        "n_slow": n_slow,
                        "n_electrodes": len(channels),
                    }
                )

            for trial_index in np.flatnonzero(finite_trials):
                trial_rows.append(
                    {
                        "task": task,
                        "subject": subject,
                        "functional_cluster": cluster,
                        "trial_uid": trial_uids[int(trial_index)],
                        "rt_log": float(target[int(trial_index)]),
                        "hga_z": float(hga_z[int(trial_index)]),
                        "n_electrodes": len(channels),
                    }
                )

        # Delay- and Response-aligned traces for the same assigned pre-Go
        # electrodes. Median-split uses that phase's rt_log table.
        if retained_clusters:
            for phase, row_list in (
                ("Delay", delay_trace_rows),
                ("Response", response_trace_rows),
            ):
                try:
                    phase_data = load_phase_data(
                        bids_roots[task],
                        task=task,
                        subject=subject,
                        phase=phase,
                        description="Repeat",
                        band="highgamma",
                        ref="bipolar",
                        atlas="hammers",
                    )
                except (FileNotFoundError, NoStrictInsulaError) as exc:
                    print(f"skip {phase} traces {task}/{subject}: {exc}")
                    continue
                row_list.extend(
                    aligned_fast_slow_traces(
                        phase_data,
                        task=task,
                        subject=subject,
                        retained_clusters=retained_clusters,
                        by_cluster=by_cluster,
                        min_trials=min_trials,
                    )
                )

    electrodes = pd.DataFrame(electrode_rows)
    traces = pd.DataFrame(trace_rows)
    delay_traces = pd.DataFrame(delay_trace_rows)
    response_traces = pd.DataFrame(response_trace_rows)
    trials = pd.DataFrame(trial_rows)
    if electrodes.empty:
        subjects = pd.DataFrame(
            columns=[
                "task",
                "subject",
                "functional_cluster",
                "n_electrodes",
                "mean_hga_log_rt_r",
                "n_trials",
            ]
        )
    else:
        # Subject collapse only for electrodes that belong to retained
        # subject × cluster units (min_trials already applied via trial_rows).
        if trials.empty:
            subjects = collapse_electrode_r_to_subject(electrodes.iloc[0:0])
        else:
            keep_keys = trials[
                ["task", "subject", "functional_cluster"]
            ].drop_duplicates()
            electrodes_kept = electrodes.merge(
                keep_keys, on=["task", "subject", "functional_cluster"], how="inner"
            )
            subjects = collapse_electrode_r_to_subject(electrodes_kept)

    inference_rows: list[dict[str, object]] = []
    if not subjects.empty:
        inference_rows.append(
            _wilcoxon_row(subjects, scope_type="all_clusters", scope="all")
        )
        for cluster in CLUSTER_ORDER:
            frame = subjects.loc[subjects["functional_cluster"] == cluster]
            inference_rows.append(
                _wilcoxon_row(
                    frame, scope_type="functional_cluster", scope=cluster
                )
            )
    if not trials.empty:
        inference_rows.append(
            _fit_lmm(
                trials,
                formula="rt_log ~ hga_z",
                scope_type="all_clusters",
                scope="all",
            )
        )
        inference_rows.append(
            _fit_lmm(
                trials,
                formula="rt_log ~ hga_z + C(task)",
                scope_type="all_clusters",
                scope="all_task_covariate",
            )
        )
        for cluster in CLUSTER_ORDER:
            frame = trials.loc[trials["functional_cluster"] == cluster]
            inference_rows.append(
                _fit_lmm(
                    frame,
                    formula="rt_log ~ hga_z",
                    scope_type="functional_cluster",
                    scope=cluster,
                )
            )
    inference = pd.DataFrame(inference_rows)

    summary_dir = output_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    traces.to_csv(summary_dir / "hga_rt_go_traces.csv", index=False)
    delay_traces.to_csv(summary_dir / "hga_rt_delay_traces.csv", index=False)
    response_traces.to_csv(summary_dir / "hga_rt_response_traces.csv", index=False)
    electrodes.to_csv(summary_dir / "hga_rt_encoding_electrodes.csv", index=False)
    subjects.to_csv(summary_dir / "hga_rt_encoding_subjects.csv", index=False)
    trials.to_csv(summary_dir / "hga_rt_trials.csv", index=False)
    inference.to_csv(summary_dir / "hga_rt_inference.csv", index=False)
    return (
        traces,
        delay_traces,
        response_traces,
        electrodes,
        subjects,
        trials,
        inference,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--min-trials", type=int, default=MIN_TRIALS)
    args = parser.parse_args()
    traces, delay_traces, response_traces, electrodes, subjects, trials, inference = (
        summarize_hga_rt_encoding(
            args.output_root,
            assignments_path=args.assignments,
            min_trials=args.min_trials,
        )
    )
    print(
        f"go_traces={len(traces)} | delay_traces={len(delay_traces)} | "
        f"response_traces={len(response_traces)} | "
        f"electrodes={len(electrodes)} | subjects={len(subjects)} | "
        f"trials={len(trials)} | inference={len(inference)}"
    )
    if not inference.empty:
        print(inference.to_string(index=False))


if __name__ == "__main__":
    main()
