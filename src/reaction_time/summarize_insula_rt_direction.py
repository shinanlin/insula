"""Describe mean-HGA/RT direction in valid strict pre-Go prediction windows.

This is a post-selection descriptive analysis: each task x electrode retains
the strictly Delay-to-Go, FWER-significant, positive-OOF-r window with the
strongest prediction score.  It then correlates trial-wise mean HGA in that
window with log RT.  A negative amplitude correlation/slope means higher HGA
is associated with shorter RT.  Because window selection and direction use the
same trials, the reported direction is not an independent confirmatory test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import binomtest, pearsonr, wilcoxon

from src.reaction_time.insula_rt_data import load_phase_data


DEFAULT_OUTPUT_ROOT = Path(
    "/hpc/group/coganlab/nanlinshi/insula-functional/results/rt"
)
DEFAULT_ASSIGNMENTS = Path(
    "/hpc/group/coganlab/nanlinshi/insula-functional/"
    "results/nmf/channel_assignments.csv"
)
DEFAULT_BIDS_ROOTS = {
    "LexicalDelay": Path("/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS"),
    "PhonemeSequence": Path("/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS"),
    "PictureNaming": Path("/cwork/ns458/BIDS-1.3_PictureNaming/BIDS"),
}


def _decode(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


def mean_hga_rt_direction(
    X: np.ndarray,
    times: np.ndarray,
    target_log_rt: np.ndarray,
    *,
    channel_index: int,
    window_start: float,
    window_end: float,
) -> tuple[float, float, float, int]:
    """Return Pearson r, p, raw slope, and n for mean HGA versus log RT."""

    times = np.asarray(times, dtype=float)
    sample_mask = (times >= float(window_start) - 1e-10) & (
        times <= float(window_end) + 1e-10
    )
    if not sample_mask.any():
        raise ValueError("Direction window contains no HGA samples")
    amplitude = np.mean(
        np.asarray(X, dtype=float)[:, int(channel_index), :][:, sample_mask], axis=1
    )
    target = np.asarray(target_log_rt, dtype=float)
    keep = np.isfinite(amplitude) & np.isfinite(target)
    if keep.sum() < 3 or np.std(amplitude[keep]) == 0 or np.std(target[keep]) == 0:
        return np.nan, np.nan, np.nan, int(keep.sum())
    correlation, p_value = pearsonr(amplitude[keep], target[keep])
    raw_slope = np.polyfit(amplitude[keep], target[keep], 1)[0]
    return float(correlation), float(p_value), float(raw_slope), int(keep.sum())


def _selected_peak_windows(path: Path, min_gap: float) -> list[dict[str, object]]:
    with h5py.File(path, "r") as h5:
        phase = str(h5.attrs["phase"])
        start = h5["windows/start"][:]
        end = h5["windows/end"][:]
        interval = (
            ((start >= 0) & (end <= min_gap))
            if phase == "Delay"
            else ((start >= -min_gap) & (end <= 0))
        )
        score = h5["scores/r"][:]
        valid = (
            h5["inference/sig_mask_fwer"][:].astype(bool)
            & np.isfinite(score)
            & (score > 0)
            & interval[None, :]
        )
        output: list[dict[str, object]] = []
        for channel_index, channel in enumerate(_decode(h5["channels/channel"][:])):
            candidates = np.flatnonzero(valid[channel_index])
            if candidates.size == 0:
                continue
            peak_index = int(
                candidates[np.argmax(score[channel_index, candidates])]
            )
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


def _subject_summary(
    electrode: pd.DataFrame,
    *,
    scope_type: str,
    scope: str,
    random_state: int,
) -> dict[str, object]:
    subject_values = electrode.groupby("subject")["mean_hga_log_rt_r"].median()
    values = subject_values.to_numpy(dtype=float)
    rng = np.random.default_rng(int(random_state))
    if len(values):
        bootstrap = np.median(
            rng.choice(values, size=(20_000, len(values)), replace=True), axis=1
        )
        ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
        negative = int((values < 0).sum())
        sign_p = binomtest(
            negative, len(values), 0.5, alternative="greater"
        ).pvalue
        wilcoxon_p = wilcoxon(values, alternative="less").pvalue
    else:
        ci_low = ci_high = sign_p = wilcoxon_p = np.nan
        negative = 0
    return {
        "scope_type": scope_type,
        "scope": scope,
        "n_electrodes": int(len(electrode)),
        "n_subjects": int(len(values)),
        "n_negative_electrodes": int((electrode["mean_hga_log_rt_r"] < 0).sum()),
        "n_negative_subjects": negative,
        "median_electrode_r": float(electrode["mean_hga_log_rt_r"].median()),
        "median_subject_r": float(np.median(values)) if len(values) else np.nan,
        "subject_median_ci_low": float(ci_low),
        "subject_median_ci_high": float(ci_high),
        "subject_sign_p_descriptive": float(sign_p),
        "subject_wilcoxon_p_descriptive": float(wilcoxon_p),
        "inference_note": "post-selection descriptive; not independent confirmation",
    }


def summarize_amplitude_direction(
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    *,
    assignments_path: Path | str = DEFAULT_ASSIGNMENTS,
    bids_roots: dict[str, Path] = DEFAULT_BIDS_ROOTS,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create phase, electrode, and subject-collapsed direction summaries."""

    output_root = Path(output_root)
    assignments = pd.read_csv(assignments_path)
    cluster_by_channel = (
        assignments.drop_duplicates("channel")
        .set_index("channel")["functional_cluster"]
        .to_dict()
    )
    phase_rows: list[dict[str, object]] = []
    for subject_dir in sorted(output_root.glob("task-*/sub-*")):
        delay_paths = list(subject_dir.glob("*proc-Delay*_rt-ridge.h5"))
        go_paths = list(subject_dir.glob("*proc-Go*_rt-ridge.h5"))
        if len(delay_paths) != 1 or len(go_paths) != 1:
            continue
        with h5py.File(delay_paths[0], "r") as h5:
            task = str(h5.attrs["task"])
            subject = str(h5.attrs["subject"])
            min_gap = float(
                np.min(h5["trials/go_onset"][:] - h5["trials/target_onset"][:])
            )
        for phase, path in (("Delay", delay_paths[0]), ("Go", go_paths[0])):
            selected = _selected_peak_windows(path, min_gap)
            if not selected:
                continue
            data = load_phase_data(
                bids_roots[task],
                task=task,
                subject=subject,
                phase=phase,
                description="Repeat",
                band="highgamma",
                ref="bipolar",
                atlas="hammers",
            )
            target = data.trial_meta["rt_log"].to_numpy(dtype=float)
            with h5py.File(path, "r") as h5:
                expected_target = h5["trials/rt_log"][:]
            if target.shape != expected_target.shape or not np.allclose(
                target, expected_target, atol=1e-10, rtol=0
            ):
                raise RuntimeError(f"RT trial alignment changed for {path}")
            channel_index = {
                channel: index
                for index, channel in enumerate(
                    data.channel_meta["channel"].astype(str)
                )
            }
            for row in selected:
                correlation, p_value, slope, n_trials = mean_hga_rt_direction(
                    data.X,
                    data.times,
                    target,
                    channel_index=channel_index[str(row["channel"])],
                    window_start=float(row["window_start"]),
                    window_end=float(row["window_end"]),
                )
                phase_rows.append(
                    {
                        "task": task,
                        "subject": subject,
                        **row,
                        "functional_cluster": cluster_by_channel.get(
                            str(row["channel"]), "unassigned"
                        ),
                        "min_delay_go_s": min_gap,
                        "n_trials": n_trials,
                        "mean_hga_log_rt_r": correlation,
                        "mean_hga_log_rt_p_descriptive": p_value,
                        "standardized_slope": correlation,
                        "raw_slope_log_rt_per_z_hga": slope,
                        "direction": (
                            "higher_HGA_shorter_RT"
                            if correlation < 0
                            else "higher_HGA_longer_RT"
                        ),
                        "source_h5": str(path),
                    }
                )
    by_phase = pd.DataFrame(phase_rows)
    electrode = (
        by_phase.sort_values("prediction_r", ascending=False)
        .drop_duplicates(["task", "subject", "channel"])
        .reset_index(drop=True)
    )
    summary_rows = [
        _subject_summary(
            electrode,
            scope_type="all_tasks",
            scope="all",
            random_state=random_state,
        )
    ]
    for task, frame in electrode.groupby("task", sort=False):
        summary_rows.append(
            _subject_summary(
                frame,
                scope_type="task",
                scope=str(task),
                random_state=random_state,
            )
        )
    for cluster, frame in electrode.groupby("functional_cluster", sort=False):
        summary_rows.append(
            _subject_summary(
                frame,
                scope_type="functional_cluster",
                scope=str(cluster),
                random_state=random_state,
            )
        )
    summary = pd.DataFrame(summary_rows)
    summary_dir = output_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    by_phase.to_csv(
        summary_dir / "positive_prego_amplitude_direction_by_phase.csv", index=False
    )
    electrode.to_csv(
        summary_dir / "positive_prego_amplitude_direction.csv", index=False
    )
    summary.to_csv(
        summary_dir / "positive_prego_amplitude_direction_summary.csv", index=False
    )
    return by_phase, electrode, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    by_phase, electrode, summary = summarize_amplitude_direction(
        args.output_root,
        assignments_path=args.assignments,
        random_state=args.random_state,
    )
    print(
        f"phase_rows={len(by_phase)} | electrodes={len(electrode)} | "
        f"summary_rows={len(summary)}"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
