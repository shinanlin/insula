#!/usr/bin/env python3
"""Unsupervised insula waveform analysis without anatomical label leakage.

The original exploratory notebook concatenates four event-aligned epochs, demeans
each epoch, fills missing epochs with zero, and shifts the entire matrix by its
global minimum.  Those operations erase a sustained delay response and make a
large constant offset dominate NMF.  This analysis instead:

1. discovers waveform classes from the stimulus epoch only;
2. rectifies already baseline-normalized HGA at zero and L2-normalizes each
   electrode so that clustering reflects temporal shape rather than amplitude;
3. normalizes the NMF factors before comparing loadings (NMF scale is otherwise
   arbitrary);
4. uses delay/go/response only after clustering, as held-out descriptions; and
5. uses Hammers AIC/PIC labels only after clustering, as spatial validation.

Defaults intentionally match the task, condition, and subject exclusion used by
``vizpub/fig2.ipynb``.  Outputs are written to ``tmp/nmf_corrected``.
"""

from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, wilcoxon
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score, silhouette_score


TASKS = (
    "PhonemeSequence",
    "LexicalDelay",
    "PictureNaming",
    "LexicalNoDelay",
)
PHASE_WINDOWS = {
    "stimulus": (-0.5, 1.0),
    "delay": (0.0, 1.0),
    "go": (-0.5, 1.0),
    "response": (-0.5, 0.5),
}
PHASE_ALIASES = {"audio": "stimulus", "resp": "response"}
USECOLS = (
    "time",
    "channel",
    "value",
    "mask",
    "subject",
    "phase",
    "modality",
    "label",
    "roi",
    "hemi",
    "x",
    "y",
    "z",
    "mix",
)
FUNCTION_COLORS = {
    "sustained_ramping": "#A9373B",
    "sensory_transient": "#2369BD",
}


def normalize_nmf_factors(
    W: np.ndarray, H: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Unit-normalize H rows and transfer their scale into W.

    This preserves ``W @ H`` while making columns of W comparable for hard
    assignment.  Comparing raw W columns is invalid because NMF admits arbitrary
    reciprocal scaling of each W column and H row.
    """

    scales = np.linalg.norm(H, axis=1)
    if np.any(scales <= 0):
        raise ValueError("Every NMF component must have non-zero norm")
    return W * scales[None, :], H / scales[:, None]


def orient_two_components(
    H: np.ndarray, times: np.ndarray
) -> dict[int, str]:
    """Name k=2 components from stimulus shape, without anatomy.

    The component with the larger early-minus-late response is called sensory
    transient.  The other component is called sustained/ramping.
    """

    if H.shape[0] != 2:
        raise ValueError("Functional orientation is defined only for k=2")
    early = (times >= 0.0) & (times <= 0.30)
    late = (times >= 0.50) & (times <= 0.90)
    if not early.any() or not late.any():
        raise ValueError("Stimulus time vector does not cover naming windows")
    transient_score = H[:, early].mean(axis=1) - H[:, late].mean(axis=1)
    sensory = int(np.argmax(transient_score))
    return {
        sensory: "sensory_transient",
        1 - sensory: "sustained_ramping",
    }


def discover_paths(results_root: Path, tasks: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    for task in tasks:
        task_root = results_root / f"{task}(bipolar)(hammers)"
        paths.extend(sorted(task_root.glob("sub-*/HGA/*desc-Repeat_time.csv")))
    if not paths:
        raise FileNotFoundError(
            f"No Repeat HGA time CSVs found below {results_root.resolve()}"
        )
    return paths


def _filename_has_phase(path: Path, wanted: set[str]) -> bool:
    name = path.name.lower()
    aliases = {
        "stimulus": ("_proc-stimulus_", "_proc-audio_"),
        "delay": ("_proc-delay_",),
        "go": ("_proc-go_",),
        "response": ("_proc-response_", "_proc-resp_"),
    }
    return any(token in name for phase in wanted for token in aliases[phase])


def load_hga_rows(
    paths: list[Path],
    *,
    phases: set[str],
    exclude_subjects: set[str],
    channels: set[str] | None = None,
) -> pd.DataFrame:
    """Stream only requested pure-insula rows from the large HGA CSV set."""

    frames: list[pd.DataFrame] = []
    selected_paths = [path for path in paths if _filename_has_phase(path, phases)]
    for index, path in enumerate(selected_paths, start=1):
        frame = pd.read_csv(path, usecols=USECOLS)
        phase = frame["phase"].astype(str).str.lower().replace(PHASE_ALIASES)
        frame = frame.assign(phase=phase)
        keep = (
            frame["phase"].isin(phases)
            & frame["modality"].eq("sound")
            & frame["roi"].isin(("AIC", "PIC"))
            & ~frame["subject"].isin(exclude_subjects)
            & ~frame["mix"].fillna(False).astype(bool)
        )
        if channels is not None:
            keep &= frame["channel"].isin(channels)
        label_number = pd.to_numeric(frame["label"], errors="coerce")
        keep &= ~label_number.eq(0)
        frame = frame.loc[keep]
        if not frame.empty:
            frames.append(frame)
        if index % 100 == 0:
            print(f"  read {index}/{len(selected_paths)} files", flush=True)
    if not frames:
        raise ValueError(f"No rows remain for phases={sorted(phases)}")
    return pd.concat(frames, ignore_index=True)


def restrict_windows(frame: pd.DataFrame) -> pd.DataFrame:
    keep = np.zeros(len(frame), dtype=bool)
    for phase, (start, stop) in PHASE_WINDOWS.items():
        keep |= (
            frame["phase"].eq(phase)
            & frame["time"].gt(start)
            & frame["time"].lt(stop)
        ).to_numpy()
    return frame.loc[keep].copy()


def channel_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("channel", sort=True)
        .agg(
            subject=("subject", "first"),
            roi=("roi", "first"),
            hemi=("hemi", "first"),
            x=("x", "first"),
            y=("y", "first"),
            z=("z", "first"),
        )
    )


def stimulus_matrix(
    frame: pd.DataFrame, *, min_coverage: float = 0.95
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stimulus = frame.loc[frame["phase"].eq("stimulus")]
    matrix = (
        stimulus.groupby(["channel", "time"], sort=True)["value"]
        .mean()
        .unstack("time")
        .sort_index(axis=1)
    )
    matrix = matrix.loc[matrix.notna().mean(axis=1) >= min_coverage]
    # A rare missing sample is interpolated within an otherwise complete epoch.
    matrix = matrix.interpolate(axis=1, limit_direction="both")
    metadata = channel_metadata(stimulus).loc[matrix.index]
    return matrix, metadata


def prepare_shape_matrix(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rectify positive HGA and normalize each electrode's temporal shape."""

    rectified = np.clip(np.asarray(raw, dtype=float), 0.0, None)
    norms = np.linalg.norm(rectified, axis=1)
    keep = norms > np.finfo(float).eps
    return rectified[keep] / norms[keep, None], keep


def fit_one_nmf(
    X: np.ndarray, k: int, seed: int, *, max_iter: int
) -> dict[str, object]:
    model = NMF(
        n_components=k,
        init="nndsvdar",
        solver="cd",
        random_state=seed,
        max_iter=max_iter,
        tol=1e-4,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        W = model.fit_transform(X)
    H = model.components_
    W_normalized, H_normalized = normalize_nmf_factors(W, H)
    labels = W_normalized.argmax(axis=1)
    return {
        "model": model,
        "W": W_normalized,
        "H": H_normalized,
        "labels": labels,
        "error": float(model.reconstruction_err_),
        "converged": bool(model.n_iter_ < max_iter),
    }


def fit_nmf_grid(
    X: np.ndarray,
    *,
    k_max: int,
    n_init: int,
    random_state: int,
    max_iter: int,
) -> tuple[pd.DataFrame, dict[int, dict[str, object]]]:
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []
    best_by_k: dict[int, dict[str, object]] = {}

    for k in range(1, min(k_max, len(X)) + 1):
        fits = [
            fit_one_nmf(X, k, int(seed), max_iter=max_iter)
            for seed in rng.integers(0, np.iinfo(np.int32).max, size=n_init)
        ]
        best = min(fits, key=lambda fit: float(fit["error"]))
        best_by_k[k] = best
        labels = np.asarray(best["labels"])
        counts = np.bincount(labels, minlength=k)
        if k > 1 and np.all(counts > 0):
            silhouette = float(silhouette_score(X, labels, metric="cosine"))
            aris = [
                adjusted_rand_score(
                    np.asarray(left["labels"]), np.asarray(right["labels"])
                )
                for left, right in itertools.combinations(fits, 2)
            ]
            stability = float(np.mean(aris))
        else:
            silhouette = np.nan
            stability = np.nan
        reconstruction = np.asarray(best["W"]) @ np.asarray(best["H"])
        explained_energy = 1.0 - float(
            np.square(X - reconstruction).sum() / np.square(X).sum()
        )
        rows.append(
            {
                "k": k,
                "reconstruction_error": float(best["error"]),
                "explained_energy": explained_energy,
                "silhouette_cosine": silhouette,
                "stability_ari": stability,
                "min_cluster_n": int(counts.min()),
                "cluster_counts": ";".join(map(str, counts)),
                "converged_runs": int(sum(bool(fit["converged"]) for fit in fits)),
                "n_init": n_init,
            }
        )
        print(
            f"  k={k}: counts={counts.tolist()} "
            f"sil={silhouette:.3f} stability={stability:.3f}",
            flush=True,
        )
    return pd.DataFrame(rows), best_by_k


def within_subject_permutation_p(
    functional_labels: np.ndarray,
    rois: np.ndarray,
    subjects: np.ndarray,
    *,
    n_permutations: int,
    random_state: int,
) -> tuple[float, float]:
    """Test anatomy/function agreement while preserving each subject's counts."""

    functional_labels = np.asarray(functional_labels)
    rois = np.asarray(rois)
    subjects = np.asarray(subjects)

    def agreement(labels: np.ndarray) -> float:
        predicted_roi = np.where(labels == "sustained_ramping", "AIC", "PIC")
        return float(np.mean(predicted_roi == rois))

    observed = agreement(functional_labels)
    rng = np.random.default_rng(random_state)
    indices = [np.flatnonzero(subjects == subject) for subject in np.unique(subjects)]
    exceedances = 0
    for _ in range(n_permutations):
        permuted = functional_labels.copy()
        for subject_indices in indices:
            permuted[subject_indices] = rng.permutation(permuted[subject_indices])
        exceedances += agreement(permuted) >= observed
    p_value = (exceedances + 1) / (n_permutations + 1)
    return observed, float(p_value)


def summarize_held_out(
    frame: pd.DataFrame, assignments: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return subject-balanced waveforms and channel/subject sample sizes."""

    values = (
        restrict_windows(frame)
        .groupby(["channel", "phase", "time"], as_index=False)["value"]
        .mean()
        .merge(
            assignments[["channel", "subject", "functional_cluster"]],
            on="channel",
            how="inner",
            validate="many_to_one",
        )
    )
    subject_means = (
        values.groupby(
            ["subject", "functional_cluster", "phase", "time"], as_index=False
        )["value"]
        .mean()
    )
    summary = (
        subject_means.groupby(
            ["functional_cluster", "phase", "time"], as_index=False
        )["value"]
        .agg(mean="mean", sem="sem", n_subjects="count")
    )
    coverage = (
        values.groupby(["functional_cluster", "phase"])
        .agg(n_channels=("channel", "nunique"), n_subjects=("subject", "nunique"))
        .reset_index()
    )
    return summary, coverage


def held_out_contrasts(
    frame: pd.DataFrame, assignments: pd.DataFrame
) -> pd.DataFrame:
    """Paired, subject-level tests of the two held-out predictions.

    Stimulus contrasts are intentionally omitted because stimulus defined the
    clusters.  Delay and response were not used by NMF and are valid held-out
    characterizations.  Only subjects containing both functional classes enter
    each paired Wilcoxon test.
    """

    tests = (
        ("delay_plateau", "delay", 0.20, 0.80, "sustained_ramping"),
        ("response_peak", "response", 0.00, 0.30, "sensory_transient"),
    )
    windowed = restrict_windows(frame)
    assignment_columns = ["channel", "functional_cluster"]
    if "subject" not in windowed.columns:
        assignment_columns.append("subject")
    values = windowed.merge(
        assignments[assignment_columns],
        on="channel",
        how="inner",
        validate="many_to_one",
    )
    rows = []
    for name, phase, start, stop, predicted_higher in tests:
        window = values.loc[
            values["phase"].eq(phase)
            & values["time"].ge(start)
            & values["time"].le(stop)
        ]
        subject_values = (
            window.groupby(["subject", "functional_cluster"])["value"]
            .mean()
            .unstack("functional_cluster")
            .dropna(subset=["sustained_ramping", "sensory_transient"])
        )
        if predicted_higher == "sustained_ramping":
            difference = (
                subject_values["sustained_ramping"]
                - subject_values["sensory_transient"]
            )
        else:
            difference = (
                subject_values["sensory_transient"]
                - subject_values["sustained_ramping"]
            )
        if len(difference) and np.any(difference.to_numpy() != 0):
            statistic, p_value = wilcoxon(
                difference.to_numpy(), alternative="greater", method="auto"
            )
        else:
            statistic, p_value = np.nan, np.nan
        rows.append(
            {
                "contrast": name,
                "test_type": "paired_between_cluster",
                "phase": phase,
                "window_start": start,
                "window_stop": stop,
                "predicted_higher": predicted_higher,
                "n_subjects": len(subject_values),
                "mean_sustained_ramping": subject_values[
                    "sustained_ramping"
                ].mean(),
                "mean_sensory_transient": subject_values[
                    "sensory_transient"
                ].mean(),
                "mean_predicted_difference": difference.mean(),
                "wilcoxon_statistic": statistic,
                "wilcoxon_one_sided_p": p_value,
            }
        )

    # Direct tests of the held-out physiological predictions use every subject
    # with the relevant functional class, not only the small paired subset.
    delay_sustained = values.loc[
        values["phase"].eq("delay")
        & values["functional_cluster"].eq("sustained_ramping")
        & values["time"].between(0.20, 0.80, inclusive="both")
    ].groupby("subject")["value"].mean()
    if len(delay_sustained) and np.any(delay_sustained.to_numpy() != 0):
        statistic, p_value = wilcoxon(
            delay_sustained.to_numpy(), alternative="greater", method="auto"
        )
    else:
        statistic, p_value = np.nan, np.nan
    rows.append(
        {
            "contrast": "sustained_delay_above_zero",
            "test_type": "one_sample_within_cluster",
            "phase": "delay",
            "window_start": 0.20,
            "window_stop": 0.80,
            "predicted_higher": "sustained_ramping > 0",
            "n_subjects": len(delay_sustained),
            "mean_sustained_ramping": delay_sustained.mean(),
            "mean_sensory_transient": np.nan,
            "mean_predicted_difference": delay_sustained.mean(),
            "wilcoxon_statistic": statistic,
            "wilcoxon_one_sided_p": p_value,
        }
    )

    response_sensory = values.loc[
        values["phase"].eq("response")
        & values["functional_cluster"].eq("sensory_transient")
        & values["time"].between(-0.40, 0.30, inclusive="both")
    ].copy()
    response_sensory["response_window"] = np.select(
        (
            response_sensory["time"].between(-0.40, -0.10, inclusive="both"),
            response_sensory["time"].between(0.00, 0.30, inclusive="both"),
        ),
        ("pre", "post"),
        default="exclude",
    )
    response_subject = (
        response_sensory.loc[response_sensory["response_window"].ne("exclude")]
        .groupby(["subject", "response_window"])["value"]
        .mean()
        .unstack("response_window")
        .reindex(columns=["pre", "post"])
        .dropna(subset=["pre", "post"])
    )
    response_difference = response_subject["post"] - response_subject["pre"]
    if len(response_difference) and np.any(response_difference.to_numpy() != 0):
        statistic, p_value = wilcoxon(
            response_difference.to_numpy(), alternative="greater", method="auto"
        )
    else:
        statistic, p_value = np.nan, np.nan
    rows.append(
        {
            "contrast": "sensory_response_post_vs_pre",
            "test_type": "paired_within_cluster",
            "phase": "response",
            "window_start": 0.00,
            "window_stop": 0.30,
            "predicted_higher": "sensory post > pre",
            "n_subjects": len(response_subject),
            "mean_sustained_ramping": np.nan,
            "mean_sensory_transient": response_subject["post"].mean(),
            "mean_predicted_difference": response_difference.mean(),
            "wilcoxon_statistic": statistic,
            "wilcoxon_one_sided_p": p_value,
        }
    )
    return pd.DataFrame(rows)


def plot_model_selection(
    metrics: pd.DataFrame, path: Path | None = None
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    axes[0].plot(metrics["k"], metrics["reconstruction_error"], "o-")
    axes[0].set_ylabel("Reconstruction error")
    axes[1].plot(metrics["k"], metrics["silhouette_cosine"], "o-")
    axes[1].set_ylabel("Cosine silhouette")
    axes[2].plot(metrics["k"], metrics["stability_ari"], "o-")
    axes[2].set_ylabel("Mean stability (ARI)")
    for axis in axes:
        axis.axvline(2, color="0.35", linestyle="--", linewidth=0.8)
        axis.set_xlabel("k")
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_waveforms(
    summary: pd.DataFrame,
    coverage: pd.DataFrame,
    path: Path | None = None,
) -> plt.Figure:
    phases = list(PHASE_WINDOWS)
    fig, axes = plt.subplots(1, len(phases), figsize=(12.0, 3.0), sharey=True)
    for axis, phase in zip(axes, phases):
        phase_summary = summary.loc[summary["phase"].eq(phase)]
        for functional_cluster in ("sustained_ramping", "sensory_transient"):
            line = phase_summary.loc[
                phase_summary["functional_cluster"].eq(functional_cluster)
            ].sort_values("time")
            if line.empty:
                continue
            x = line["time"].to_numpy(float)
            mean = line["mean"].to_numpy(float)
            sem = line["sem"].fillna(0).to_numpy(float)
            label_coverage = coverage.loc[
                coverage["phase"].eq(phase)
                & coverage["functional_cluster"].eq(functional_cluster)
            ]
            if label_coverage.empty:
                label = functional_cluster
            else:
                row = label_coverage.iloc[0]
                label = (
                    f"{functional_cluster.replace('_', ' ')} "
                    f"(E={int(row.n_channels)}, S={int(row.n_subjects)})"
                )
            color = FUNCTION_COLORS[functional_cluster]
            axis.plot(x, mean, color=color, linewidth=1.5, label=label)
            axis.fill_between(
                x, mean - 1.96 * sem, mean + 1.96 * sem,
                color=color, alpha=0.18, linewidth=0,
            )
        axis.axvline(0, color="0.25", linestyle="--", linewidth=0.7)
        axis.axhline(0, color="0.6", linewidth=0.5)
        axis.set_title(phase.capitalize())
        axis.set_xlabel("Time (s)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("HGA (z), subject-balanced mean")
    axes[-1].legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_spatial(
    assignments: pd.DataFrame, path: Path | None = None
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharex=True, sharey=True)
    for axis, (hemi, title) in zip(axes, (("L", "Left"), ("R", "Right"))):
        side = assignments.loc[assignments["hemi"].eq(hemi)]
        for functional_cluster in ("sustained_ramping", "sensory_transient"):
            points = side.loc[
                side["functional_cluster"].eq(functional_cluster)
            ]
            axis.scatter(
                points["y"], points["z"], s=25,
                color=FUNCTION_COLORS[functional_cluster],
                edgecolor="0.2", linewidth=0.35, alpha=0.85,
                label=functional_cluster.replace("_", " "),
            )
        axis.set_title(f"{title} insula")
        axis.set_xlabel("y (mm; anterior +)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("z (mm)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(fig)
    return fig


def write_component_table(H: np.ndarray, times: np.ndarray, path: Path) -> None:
    mapping = orient_two_components(H, times)
    rows = []
    for component in range(2):
        rows.extend(
            {
                "component": component,
                "functional_cluster": mapping[component],
                "phase": "stimulus",
                "time": time,
                "normalized_H": value,
            }
            for time, value in zip(times, H[component])
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    all_paths = discover_paths(args.results_root, tuple(args.tasks))

    print("Loading stimulus discovery data...", flush=True)
    stimulus_rows = load_hga_rows(
        all_paths,
        phases={"stimulus"},
        exclude_subjects=set(args.exclude_subject),
    )
    stimulus_rows = restrict_windows(stimulus_rows)
    raw_matrix, metadata = stimulus_matrix(stimulus_rows)
    X, keep = prepare_shape_matrix(raw_matrix.to_numpy())
    raw_matrix = raw_matrix.iloc[np.flatnonzero(keep)]
    metadata = metadata.iloc[np.flatnonzero(keep)]
    print(
        f"Discovery matrix: {X.shape[0]} electrodes x {X.shape[1]} time points; "
        f"AIC={metadata['roi'].eq('AIC').sum()}, PIC={metadata['roi'].eq('PIC').sum()}",
        flush=True,
    )

    print("Fitting NMF grid...", flush=True)
    metrics, fits = fit_nmf_grid(
        X,
        k_max=args.k_max,
        n_init=args.n_init,
        random_state=args.random_state,
        max_iter=args.max_iter,
    )
    if 2 not in fits:
        raise ValueError("At least two electrodes are required for k=2")
    fit = fits[2]
    W = np.asarray(fit["W"])
    H = np.asarray(fit["H"])
    component = np.asarray(fit["labels"], dtype=int)
    times = raw_matrix.columns.to_numpy(float)
    component_names = orient_two_components(H, times)
    functional_cluster = np.array([component_names[value] for value in component])

    assignments = metadata.reset_index().copy()
    assignments["component"] = component
    assignments["functional_cluster"] = functional_cluster
    for component_index, functional_name in component_names.items():
        assignments[f"loading_{functional_name}"] = W[:, component_index]
    assignments["dominance"] = W.max(axis=1) / np.maximum(W.sum(axis=1), 1e-12)
    assignments.to_csv(output_dir / "channel_assignments.csv", index=False)
    metrics.to_csv(output_dir / "model_selection_metrics.csv", index=False)
    write_component_table(H, times, output_dir / "stimulus_components.csv")

    crosstab = pd.crosstab(
        assignments["functional_cluster"], assignments["roi"]
    ).reindex(
        index=["sustained_ramping", "sensory_transient"],
        columns=["AIC", "PIC"],
        fill_value=0,
    )
    crosstab.to_csv(output_dir / "functional_by_hammers.csv")
    odds_ratio, fisher_p = fisher_exact(crosstab.to_numpy())
    agreement, permutation_p = within_subject_permutation_p(
        assignments["functional_cluster"].to_numpy(),
        assignments["roi"].to_numpy(),
        assignments["subject"].to_numpy(),
        n_permutations=args.n_permutations,
        random_state=args.random_state,
    )
    spatial_stats = pd.DataFrame(
        [
            {
                "n_electrodes": len(assignments),
                "n_subjects": assignments["subject"].nunique(),
                "anatomy_function_agreement": agreement,
                "within_subject_permutation_p": permutation_p,
                "fisher_odds_ratio": odds_ratio,
                "fisher_p_electrode_level": fisher_p,
            }
        ]
    )
    spatial_stats.to_csv(output_dir / "spatial_validation.csv", index=False)

    print("Loading held-out delay/go/response data...", flush=True)
    held_out_rows = load_hga_rows(
        all_paths,
        phases=set(PHASE_WINDOWS),
        exclude_subjects=set(args.exclude_subject),
        channels=set(assignments["channel"]),
    )
    waveform_summary, coverage = summarize_held_out(held_out_rows, assignments)
    waveform_summary.to_csv(output_dir / "held_out_waveform_summary.csv", index=False)
    coverage.to_csv(output_dir / "held_out_coverage.csv", index=False)
    contrasts = held_out_contrasts(held_out_rows, assignments)
    contrasts.to_csv(output_dir / "held_out_contrasts.csv", index=False)

    plot_model_selection(metrics, output_dir / "model_selection.png")
    plot_waveforms(waveform_summary, coverage, output_dir / "cluster_waveforms.png")
    plot_spatial(assignments, output_dir / "spatial_yz.png")

    print("\nFunctional cluster x Hammers atlas:")
    print(crosstab)
    print(
        f"Anatomy/function agreement={agreement:.1%}; "
        f"within-subject permutation p={permutation_p:.4g}; "
        f"electrode-level OR={odds_ratio:.2f} (Fisher p={fisher_p:.4g})"
    )
    print("\nHeld-out subject-level contrasts:")
    print(
        contrasts[
            [
                "contrast",
                "test_type",
                "n_subjects",
                "mean_predicted_difference",
                "wilcoxon_one_sided_p",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote outputs to {output_dir.resolve()}")


