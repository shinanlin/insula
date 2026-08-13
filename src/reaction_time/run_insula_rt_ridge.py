"""Run item-grouped, single-electrode time-resolved RT ridge prediction."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.reaction_time.insula_ridge import (
    fit_window_scores,
    joint_cluster_correction,
    make_group_splits,
    make_permutation_seeds,
)
from src.reaction_time.insula_rt_data import (
    NoStrictInsulaError,
    PhaseData,
    load_phase_data,
)
from src.reaction_time.insula_rt_io import (
    DEFAULT_RT_OUTPUT_ROOT,
    PhaseModelResult,
    phase_output_path,
    write_phase_result,
)


LOGGER = logging.getLogger(__name__)
SUPPORTED_TASKS = ("LexicalDelay", "PhonemeSequence", "PictureNaming")
SUPPORTED_PHASES = ("Delay", "Go")


@dataclass(frozen=True)
class UncorrectedPhaseResult:
    data: PhaseData
    score_r: np.ndarray
    score_r2: np.ndarray
    score_mae: np.ndarray
    perm_score_r: np.ndarray
    oof_prediction: np.ndarray
    fold_id: np.ndarray
    window_start: np.ndarray
    window_end: np.ndarray
    window_center: np.ndarray


def sliding_windows(
    times: np.ndarray,
    sfreq: float,
    *,
    window_s: float,
    step_s: float,
    max_windows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return sample slices and exact start/end/center coordinates."""

    times = np.asarray(times, dtype=float)
    window_samples = max(1, int(round(float(window_s) * float(sfreq))))
    step_samples = max(1, int(round(float(step_s) * float(sfreq))))
    if window_samples > len(times):
        raise ValueError("Sliding window is longer than the epoch")
    starts = np.arange(0, len(times) - window_samples + 1, step_samples, dtype=int)
    if max_windows is not None:
        starts = starts[: int(max_windows)]
    stops = starts + window_samples
    start_time = times[starts]
    end_time = times[stops - 1]
    center_time = (start_time + end_time) / 2.0
    return starts, stops, start_time, end_time, center_time


def fit_phase(
    data: PhaseData,
    *,
    window_s: float,
    step_s: float,
    n_folds: int,
    inner_folds: int,
    n_perm: int,
    random_state: int,
    n_jobs: int,
    max_windows: int | None,
) -> UncorrectedPhaseResult:
    groups = data.trial_meta["item_id"].astype(str).to_numpy()
    splits, fold_id = make_group_splits(
        groups, n_splits=n_folds, random_state=random_state
    )
    permutation_seeds = make_permutation_seeds(
        len(splits), n_perm, random_state=random_state
    )
    starts, stops, window_start, window_end, window_center = sliding_windows(
        data.times,
        data.sfreq,
        window_s=window_s,
        step_s=step_s,
        max_windows=max_windows,
    )
    shape = (data.X.shape[1], len(starts))
    score_r = np.full(shape, np.nan, dtype=float)
    score_r2 = np.full(shape, np.nan, dtype=float)
    score_mae = np.full(shape, np.nan, dtype=float)
    perm_score_r = np.full((*shape, n_perm), np.nan, dtype=float)
    oof_prediction = np.full((*shape, data.X.shape[0]), np.nan, dtype=float)
    target = data.trial_meta["rt_log"].to_numpy(dtype=float)

    iterator = zip(starts, stops)
    for time_index, (start, stop) in enumerate(
        tqdm(
            iterator,
            total=len(starts),
            desc=f"sub-{data.subject} {data.phase}",
            leave=False,
        )
    ):
        fitted = fit_window_scores(
            data.X[..., start:stop],
            target,
            groups,
            splits,
            permutation_seeds,
            inner_splits=inner_folds,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        score_r[:, time_index] = fitted.score_r
        score_r2[:, time_index] = fitted.score_r2
        score_mae[:, time_index] = fitted.score_mae
        perm_score_r[:, time_index, :] = fitted.perm_score_r
        oof_prediction[:, time_index, :] = fitted.oof_prediction

    return UncorrectedPhaseResult(
        data=data,
        score_r=score_r,
        score_r2=score_r2,
        score_mae=score_mae,
        perm_score_r=perm_score_r,
        oof_prediction=oof_prediction,
        fold_id=fold_id,
        window_start=window_start,
        window_end=window_end,
        window_center=window_center,
    )


def _write_status(output_root: Path, task: str, subject: str, payload: dict) -> Path:
    path = output_root / f"task-{task}" / f"sub-{subject}" / "run_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return path


def run_subject(
    *,
    bids_root: Path | str,
    output_root: Path | str,
    task: str,
    subject: str,
    phases: tuple[str, ...] = SUPPORTED_PHASES,
    description: str = "Repeat",
    band: str = "highgamma",
    ref: str = "bipolar",
    atlas: str = "hammers",
    window_s: float = 0.2,
    step_s: float = 0.02,
    n_folds: int = 10,
    inner_folds: int = 5,
    n_perm: int = 1000,
    random_state: int = 42,
    n_jobs: int = 1,
    raw_sfreq: float = 2048.0,
    rt_min_s: float = 0.05,
    max_windows: int | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Fit both phases, correct them jointly, then atomically write results."""

    if task not in SUPPORTED_TASKS:
        raise ValueError(f"task must be one of {SUPPORTED_TASKS}")
    invalid_phases = set(phases).difference(SUPPORTED_PHASES)
    if invalid_phases:
        raise ValueError(f"Unsupported phases: {sorted(invalid_phases)}")
    subject = subject[4:] if str(subject).startswith("sub-") else str(subject)
    output_root = Path(output_root)
    fitted: dict[str, UncorrectedPhaseResult] = {}
    try:
        for phase in phases:
            data = load_phase_data(
                bids_root,
                task=task,
                subject=subject,
                phase=phase,
                description=description,
                band=band,
                ref=ref,
                atlas=atlas,
                raw_sfreq=raw_sfreq,
                rt_min_s=rt_min_s,
            )
            LOGGER.info(
                "sub-%s %s: %d trials, %d items, %d strict-insula channels",
                subject,
                phase,
                len(data.trial_meta),
                data.trial_meta["item_id"].nunique(),
                len(data.channel_meta),
            )
            fitted[phase] = fit_phase(
                data,
                window_s=window_s,
                step_s=step_s,
                n_folds=n_folds,
                inner_folds=inner_folds,
                n_perm=n_perm,
                random_state=random_state,
                n_jobs=n_jobs,
                max_windows=max_windows,
            )
    except NoStrictInsulaError as error:
        _write_status(
            output_root,
            task,
            subject,
            {"status": "skipped_no_insula", "reason": str(error)},
        )
        LOGGER.warning("%s", error)
        return []

    corrected = joint_cluster_correction(
        {
            phase: (result.score_r, result.perm_score_r)
            for phase, result in fitted.items()
        }
    )
    config = {
        "window_requested_s": float(window_s),
        "step_requested_s": float(step_s),
        "n_folds_requested": int(n_folds),
        "inner_folds_requested": int(inner_folds),
        "n_perm": int(n_perm),
        "random_state": int(random_state),
        "rt_min_s": float(rt_min_s),
        "band": band,
        "ref": ref,
        "atlas": atlas,
        "strict_rois": "AIC,PIC",
        "max_windows": max_windows,
    }
    paths: list[Path] = []
    for phase, result in fitted.items():
        model_result = PhaseModelResult(
            score_r=result.score_r,
            score_r2=result.score_r2,
            score_mae=result.score_mae,
            perm_score_r=result.perm_score_r,
            oof_prediction=result.oof_prediction,
            fold_id=result.fold_id,
            window_start=result.window_start,
            window_end=result.window_end,
            window_center=result.window_center,
            cluster=corrected[phase],
        )
        path = phase_output_path(
            output_root,
            task=task,
            subject=subject,
            phase=phase,
            description=description,
        )
        paths.append(
            write_phase_result(
                path,
                data=result.data,
                result=model_result,
                config=config,
                overwrite=overwrite,
            )
        )
    _write_status(
        output_root,
        task,
        subject,
        {
            "status": "complete",
            "outputs": [str(path) for path in paths],
            "phases": list(phases),
            "n_perm": int(n_perm),
        },
    )
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RT_OUTPUT_ROOT)
    parser.add_argument("--task", required=True, choices=SUPPORTED_TASKS)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--phases", nargs="+", choices=SUPPORTED_PHASES, default=list(SUPPORTED_PHASES))
    parser.add_argument("--description", default="Repeat")
    parser.add_argument("--band", default="highgamma")
    parser.add_argument("--ref", default="bipolar")
    parser.add_argument("--atlas", default="hammers")
    parser.add_argument("--window-s", type=float, default=0.2)
    parser.add_argument("--step-s", type=float, default=0.02)
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--raw-sfreq", type=float, default=2048.0)
    parser.add_argument("--rt-min-s", type=float, default=0.05)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = vars(build_arg_parser().parse_args())
    outputs = run_subject(**args)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
