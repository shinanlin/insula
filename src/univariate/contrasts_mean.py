"""Phase-window HGA univariate contrasts.

Reduces each trial inside a fixed phase window (mean or 75th percentile),
then runs per-channel two-sample label permutation tests with BH-FDR.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from collections.abc import Sequence

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from src.paths import hga_results_dir

ATLAS = "hammers"
DEFAULT_BAND = "highgamma"
MIN_TRIALS_PER_CLASS = 3

PHASE_WINDOWS: dict[str, tuple[float, float]] = {
    "Stimulus": (0.0, 0.5),
    "Delay": (0.0, 0.5),
    "Go": (0.0, 0.5),
    "Response": (-0.5, 0.5),
}

CONTRAST_DESCRIPTIONS = {
    "DecisionVsRepeat": "DecisionVsRepeatMean",
    "WordVsNonwordDecision": "WordVsNonwordDecisionMean",
    "WordVsNonwordRepeat": "WordVsNonwordRepeatMean",
    "RepeatVsPassive": "RepeatVsPassiveMean",
}

DEFAULT_CONTRASTS = (
    "DecisionVsRepeat",
    "WordVsNonwordDecision",
    "WordVsNonwordRepeat",
)

DESCRIPTION_ARMS: dict[str, tuple[str, ...]] = {
    "DecisionVsRepeat": ("Decision", "Repeat"),
    "WordVsNonwordDecision": ("Decision",),
    "WordVsNonwordRepeat": ("Repeat",),
    "RepeatVsPassive": ("Repeat", "Passive"),
}

WINDOW_STATS = ("mean", "p75")
WINDOW_P75 = 75.0


def normalize_phase_name(phase: str) -> str:
    key = str(phase).strip().lower()
    for name in PHASE_WINDOWS:
        if name.lower() == key:
            return name
    raise ValueError(f"Unknown phase {phase!r}; expected one of {list(PHASE_WINDOWS)}")


def resolve_contrast_description(contrast_key: str, window_stat: str = "mean") -> str:
    """Filename description: ``*Mean`` or ``*P75``."""
    if contrast_key not in CONTRAST_DESCRIPTIONS:
        raise ValueError(f"Unknown contrast {contrast_key!r}")
    if window_stat not in WINDOW_STATS:
        raise ValueError(f"Unknown window_stat {window_stat!r}; expected {WINDOW_STATS}")
    desc = CONTRAST_DESCRIPTIONS[contrast_key]
    if window_stat == "mean":
        return desc
    if not desc.endswith("Mean"):
        raise ValueError(f"Cannot map {desc!r} to a P75 description")
    return f"{desc[:-4]}P75"


def window_reduce(trial_ch_time: np.ndarray, window_stat: str = "mean") -> np.ndarray:
    """Reduce one trial ``(n_channel, n_time)`` to ``(n_channel,)``."""
    if window_stat == "mean":
        return np.nanmean(trial_ch_time, axis=1)
    if window_stat == "p75":
        return np.nanpercentile(trial_ch_time, WINDOW_P75, axis=1)
    raise ValueError(f"Unknown window_stat {window_stat!r}; expected {WINDOW_STATS}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def parse_contrast_keys(values: Sequence[str] | None) -> list[str]:
    """Flatten ``--contrasts`` tokens (repeatable and/or comma-separated)."""
    if not values:
        return list(DEFAULT_CONTRASTS)
    keys: list[str] = []
    for value in values:
        keys.extend(part.strip() for part in str(value).split(",") if part.strip())
    unknown = [key for key in keys if key not in CONTRAST_DESCRIPTIONS]
    if unknown:
        raise ValueError(
            f"Unknown contrast(s) {unknown}; expected one of "
            f"{sorted(CONTRAST_DESCRIPTIONS)}"
        )
    return keys


def descriptions_for_contrasts(contrast_keys: Sequence[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for key in contrast_keys:
        if key not in DESCRIPTION_ARMS:
            raise ValueError(f"Unknown contrast {key!r}")
        for description in DESCRIPTION_ARMS[key]:
            if description not in seen:
                seen.append(description)
    return tuple(seen)


def select_epoch_path(paths: Sequence, recording: str | None = None):
    """Return the unique matching epoch path, or None if empty."""
    selected = list(paths)
    if recording is not None:
        selected = [path for path in selected if getattr(path, "recording", None) == recording]
    if not selected:
        return None
    if len(selected) > 1:
        raise ValueError(
            "Multiple epoch files match; set --recording to disambiguate: "
            + ", ".join(str(path) for path in selected)
        )
    return selected[0]


def keep_correct_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Keep CORRECT trials when accuracy is labeled; otherwise keep all.

    Lexical events use a slash-delimited CORRECT/INCORRECT token. PictureNaming
    events (ListenSpeak / JustListen) have no accuracy tag, so they are kept.
    """
    if df.empty:
        return df
    if "event_name" in df.columns:
        names = df["event_name"].astype(str)
        has_correct = names.str.contains(r"(?:^|/)CORRECT(?:/|$)", na=False)
        has_incorrect = names.str.contains(r"(?:^|/)INCORRECT(?:/|$)", na=False)
        if not (has_correct | has_incorrect).any():
            return df.copy()
        return df[has_correct].copy()
    if "remark" not in df.columns:
        return df.copy()
    remarks = df["remark"].astype(str)
    labeled = remarks.isin(["CORRECT", "INCORRECT"])
    if not labeled.any():
        return df.copy()
    return df[remarks.eq("CORRECT")].copy()


def channel_rng(subject: str, channel: str, phase: str, contrast: str) -> np.random.Generator:
    key = f"{subject}|{channel}|{phase}|{contrast}".encode()
    seed = int.from_bytes(hashlib.sha256(key).digest()[:4], "little")
    return np.random.default_rng(seed)


def sem(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def perm_test_two_sample(
    vals_a: np.ndarray,
    vals_b: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Two-tailed label permutation (mean_a - mean_b)."""
    vals_a = vals_a[np.isfinite(vals_a)]
    vals_b = vals_b[np.isfinite(vals_b)]
    if len(vals_a) == 0 or len(vals_b) == 0:
        return float("nan"), float("nan")

    obs = float(np.mean(vals_a) - np.mean(vals_b))
    pooled = np.concatenate([vals_a, vals_b])
    labels = np.array([0] * len(vals_a) + [1] * len(vals_b))
    exceed = 0
    for _ in range(n_perm):
        perm_labels = rng.permutation(labels)
        pa = pooled[perm_labels == 0]
        pb = pooled[perm_labels == 1]
        if abs(float(np.mean(pa) - np.mean(pb))) >= abs(obs):
            exceed += 1
    p_value = (exceed + 1) / (n_perm + 1)
    return p_value, obs


def apply_channel_fdr(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    out = df.copy()
    pvals = out["p_value"].to_numpy(dtype=float)
    valid = np.isfinite(pvals)
    out["p_fdr"] = np.nan
    if valid.any():
        _, fdr, _, _ = multipletests(pvals[valid], method="fdr_bh")
        out.loc[valid, "p_fdr"] = fdr
    out["significant"] = out["p_fdr"] < alpha
    return out


def parse_epoch_metadata(
    epochs: mne.Epochs,
    description: str,
    subject: str,
    phase: str,
    window_stat: str = "mean",
) -> pd.DataFrame:
    """Trial-level window summaries with parsed condition fields."""
    conditions = epochs.events[:, 2]
    event_id_inv = {v: k for k, v in epochs.event_id.items()}
    rows: list[dict] = []
    data = epochs.get_data()

    for trial_idx in range(len(epochs)):
        condition = event_id_inv[int(conditions[trial_idx])]
        parts = condition.split("/")
        lexicality = parts[2] if len(parts) > 2 else ""
        remark = "/".join(parts[4:]) if len(parts) > 4 else ""
        condition_short = "/".join(parts[2:4]) if len(parts) > 3 else condition
        trial_means = window_reduce(data[trial_idx], window_stat)

        for ch_idx, channel in enumerate(epochs.ch_names):
            rows.append(
                {
                    "channel": channel,
                    "lexicality": lexicality,
                    "remark": remark,
                    "event_name": condition,
                    "condition_short": condition_short,
                    "description": description,
                    "phase": phase.lower(),
                    "subject": subject,
                    "trial": f"{condition_short}_{trial_idx}",
                    "mean_hga": float(trial_means[ch_idx]),
                }
            )

    return pd.DataFrame(rows)


def load_phase_description_means(
    bids_root: str,
    reference: str,
    band: str,
    subject: str,
    phase: str,
    description: str,
    tmin: float,
    tmax: float,
    recording: str | None = None,
    window_stat: str = "mean",
) -> pd.DataFrame:
    query_kwargs: dict = {
        "root": os.path.join(bids_root, "derivatives", f"epoch({reference})"),
        "datatype": "epoch(band)(zscore)",
        "suffix": band,
        "subject": subject,
        "processing": phase,
        "description": description,
        "extension": ".h5",
        "check": False,
    }
    if recording is not None:
        query_kwargs["recording"] = recording
    path = select_epoch_path(BIDSPath(**query_kwargs).match(), recording=recording)
    if path is None:
        return pd.DataFrame()

    epochs = mne.read_epochs(path, preload=True, verbose="error")
    epochs = epochs.copy().crop(tmin=tmin, tmax=tmax)
    if epochs.get_data().shape[-1] == 0:
        return pd.DataFrame()

    df = parse_epoch_metadata(
        epochs, description, subject, phase, window_stat=window_stat
    )
    return keep_correct_trials(df)


def analyze_channel_contrast(
    phase_df: pd.DataFrame,
    channel: str,
    mask_a,
    mask_b,
    label_a: str,
    label_b: str,
    subject: str,
    phase: str,
    contrast_key: str,
    n_perm: int,
) -> dict | None:
    arm_a = phase_df.loc[mask_a & (phase_df["channel"] == channel), "mean_hga"].to_numpy()
    arm_b = phase_df.loc[mask_b & (phase_df["channel"] == channel), "mean_hga"].to_numpy()
    arm_a = arm_a[np.isfinite(arm_a)]
    arm_b = arm_b[np.isfinite(arm_b)]

    if len(arm_a) < MIN_TRIALS_PER_CLASS or len(arm_b) < MIN_TRIALS_PER_CLASS:
        return None

    rng = channel_rng(subject, channel, phase, contrast_key)
    p_value, mean_diff = perm_test_two_sample(arm_a, arm_b, n_perm, rng)
    if not np.isfinite(p_value):
        return None

    mean_a = float(np.mean(arm_a))
    mean_b = float(np.mean(arm_b))
    return {
        "subject": subject,
        "phase": phase.lower(),
        "contrast": CONTRAST_DESCRIPTIONS[contrast_key],
        "channel": channel,
        "n_a": int(len(arm_a)),
        "n_b": int(len(arm_b)),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "sem_a": sem(arm_a),
        "sem_b": sem(arm_b),
        "mean_diff": mean_diff,
        "p_value": float(p_value),
    }


def run_phase_contrast(
    phase_df: pd.DataFrame,
    contrast_key: str,
    subject: str,
    phase: str,
    n_perm: int,
    alpha: float,
    window_stat: str = "mean",
) -> pd.DataFrame:
    if contrast_key == "DecisionVsRepeat":
        mask_a = phase_df["description"] == "Decision"
        mask_b = phase_df["description"] == "Repeat"
        label_a, label_b = "Decision", "Repeat"
    elif contrast_key == "WordVsNonwordDecision":
        mask_a = (phase_df["description"] == "Decision") & (phase_df["lexicality"] == "Word")
        mask_b = (phase_df["description"] == "Decision") & (phase_df["lexicality"] == "Nonword")
        label_a, label_b = "Word", "Nonword"
    elif contrast_key == "WordVsNonwordRepeat":
        mask_a = (phase_df["description"] == "Repeat") & (phase_df["lexicality"] == "Word")
        mask_b = (phase_df["description"] == "Repeat") & (phase_df["lexicality"] == "Nonword")
        label_a, label_b = "Word", "Nonword"
    elif contrast_key == "RepeatVsPassive":
        mask_a = phase_df["description"] == "Repeat"
        mask_b = phase_df["description"] == "Passive"
        label_a, label_b = "Repeat", "Passive"
    else:
        raise ValueError(f"Unknown contrast {contrast_key!r}")

    if not mask_a.any() or not mask_b.any():
        return pd.DataFrame()

    channels = sorted(set(phase_df.loc[mask_a, "channel"]) & set(phase_df.loc[mask_b, "channel"]))
    results: list[dict] = []
    for channel in channels:
        row = analyze_channel_contrast(
            phase_df,
            channel,
            mask_a,
            mask_b,
            label_a,
            label_b,
            subject,
            phase,
            contrast_key,
            n_perm,
        )
        if row is not None:
            results.append(row)

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    result_df["contrast"] = resolve_contrast_description(contrast_key, window_stat)
    result_df = apply_channel_fdr(result_df, alpha)
    return result_df


def save_results(
    df: pd.DataFrame,
    task_name: str,
    reference: str,
    subject: str,
    phase: str,
    contrast_description: str,
    band: str,
) -> None:
    save_path = BIDSPath(
        root=str(hga_results_dir(task_name, reference, ATLAS)),
        datatype="univariate",
        suffix=band,
        task=task_name,
        subject=subject,
        processing=phase,
        description=contrast_description,
        extension=".csv",
        check=False,
    )
    save_path.mkdir(exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.info("Saved %s (%d channels)", save_path, len(df))


def discover_epoch_paths(
    bids_root: str,
    reference: str,
    band: str,
    subject: str,
    recording: str | None = None,
) -> list:
    query_kwargs: dict = {
        "root": os.path.join(bids_root, "derivatives", f"epoch({reference})"),
        "datatype": "epoch(band)(zscore)",
        "suffix": band,
        "subject": subject,
        "extension": ".h5",
        "check": False,
    }
    if recording is not None:
        query_kwargs["recording"] = recording
    return BIDSPath(**query_kwargs).match()


def discover_phases(
    bids_root: str,
    reference: str,
    band: str,
    subject: str,
    recording: str | None = None,
) -> list[str]:
    paths = discover_epoch_paths(bids_root, reference, band, subject, recording)
    return sorted({str(p.processing) for p in paths if p.processing in PHASE_WINDOWS})


def main(
    bids_root: str,
    band: str = DEFAULT_BAND,
    reference: str = "bipolar",
    subject: str | None = None,
    n_perm: int = 5000,
    alpha: float = 0.05,
    contrasts: Sequence[str] | None = None,
    recording: str | None = None,
    window_stat: str = "mean",
    phases: Sequence[str] | None = None,
) -> None:
    if subject is None:
        raise ValueError("--subject is required")

    if window_stat not in WINDOW_STATS:
        raise ValueError(f"Unknown window_stat {window_stat!r}; expected {WINDOW_STATS}")
    contrast_keys = parse_contrast_keys(contrasts)
    needed = descriptions_for_contrasts(contrast_keys)
    found_phases = discover_phases(bids_root, reference, band, subject, recording)
    if phases:
        wanted = [normalize_phase_name(p) for p in phases]
        phases = [p for p in found_phases if p in wanted]
        missing = [p for p in wanted if p not in found_phases]
        if missing:
            logger.warning("Requested phases not found for %s: %s", subject, missing)
    else:
        phases = found_phases
    if not phases:
        logger.warning("No epoch phases found for %s", subject)
        return

    epoch_paths = discover_epoch_paths(bids_root, reference, band, subject, recording)
    if not epoch_paths:
        logger.warning("No epoch files found for %s", subject)
        return
    task_name = epoch_paths[0].task

    logger.info(
        "Processing %s | task=%s | phases=%s | contrasts=%s | descriptions=%s | "
        "recording=%s | window_stat=%s",
        subject,
        task_name,
        phases,
        contrast_keys,
        needed,
        recording,
        window_stat,
    )

    for phase in tqdm(phases, desc=f"{subject} phases"):
        tmin, tmax = PHASE_WINDOWS[phase]
        frames = [
            load_phase_description_means(
                bids_root,
                reference,
                band,
                subject,
                phase,
                description,
                tmin,
                tmax,
                recording=recording,
                window_stat=window_stat,
            )
            for description in needed
        ]
        phase_df = pd.concat(frames, ignore_index=True)
        if phase_df.empty:
            logger.warning("No CORRECT trials for %s %s", subject, phase)
            continue

        for contrast_key in contrast_keys:
            result_df = run_phase_contrast(
                phase_df,
                contrast_key,
                subject,
                phase,
                n_perm,
                alpha,
                window_stat=window_stat,
            )
            if result_df.empty:
                logger.info("No results for %s %s %s", subject, phase, contrast_key)
                continue
            save_results(
                result_df,
                task_name,
                reference,
                subject,
                phase,
                resolve_contrast_description(contrast_key, window_stat),
                band,
            )


def _self_check() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(1.5, 0.2, 20)
    b = rng.normal(0.5, 0.2, 20)
    p, diff = perm_test_two_sample(a, b, n_perm=500, rng=rng)
    assert diff > 0, diff
    assert p < 0.05, p

    fake = pd.DataFrame({"p_value": [0.001, 0.02, 0.04, 0.5, 0.9]})
    corrected = apply_channel_fdr(fake, alpha=0.05)
    n_unc = int((fake["p_value"] < 0.05).sum())
    n_fdr = int(corrected["significant"].sum())
    assert n_fdr <= n_unc, (n_fdr, n_unc)


if __name__ == "__main__":
    _self_check()

    parser = argparse.ArgumentParser(
        description="Phase-window mean univariate contrasts"
    )
    parser.add_argument(
        "--bids_root",
        type=str,
        default="/cwork/ns458/BIDS-1.0_LexicalDecRepNoDelay/BIDS/",
    )
    parser.add_argument("--band", type=str, default=DEFAULT_BAND)
    parser.add_argument("--reference", type=str, default="bipolar")
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--n_perm", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--contrasts",
        action="append",
        default=None,
        help=(
            "Contrast key(s). Repeat the flag or comma-separate. "
            f"Default: {','.join(DEFAULT_CONTRASTS)}."
        ),
    )
    parser.add_argument(
        "--recording",
        default=None,
        help="Optional BIDS recording entity (e.g. sound for PictureNaming).",
    )
    parser.add_argument(
        "--window_stat",
        choices=WINDOW_STATS,
        default="mean",
        help="Per-trial window reduction: mean or 75th percentile (p75).",
    )
    parser.add_argument(
        "--phases",
        action="append",
        default=None,
        help="Phase(s) to run. Repeat the flag or comma-separate. Default: all found.",
    )
    args = parser.parse_args()
    phases = None
    if args.phases:
        phases = []
        for value in args.phases:
            phases.extend(part.strip() for part in str(value).split(",") if part.strip())
    main(
        bids_root=args.bids_root,
        band=args.band,
        reference=args.reference,
        subject=args.subject,
        n_perm=args.n_perm,
        alpha=args.alpha,
        contrasts=args.contrasts,
        recording=args.recording,
        window_stat=args.window_stat,
        phases=phases,
    )
