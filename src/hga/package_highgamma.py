"""Package trial-averaged HGA waveforms to long-format CSV under results/.

Electrode membership is the **subject × modality union** of significant
channels across all packaged conditions and phases (``epoch(band)(sig)``).
Waveforms are read from ``epoch(band)(zscore)`` and trial-averaged with
``nanmean``.  Presence in a phase CSV does **not** mean the electrode was
significant in that phase; use the ``mask`` column (from statistics) for
within-phase significance.
"""

from __future__ import annotations

import argparse
import h5py
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.paths import SUPPORTED_ATLASES, hga_results_dir as results_dir

ENDPOINT_NATIVE_COLS = ("x1", "y1", "z1", "x2", "y2", "z2")
ENDPOINT_TEMPLATE_COLS = ("x1_t", "y1_t", "z1_t", "x2_t", "y2_t", "z2_t")
CONTACT_COLS = ("contact_1", "contact_2", "contact_1_label", "contact_2_label")


SUPPORTED_FAMILIES = ("default", "sentence")


def canonical_task_name(task: str) -> str:
    """Map legacy misspelled BIDS task labels to the canonical name."""
    if task == "PhonemeSequencing":
        return "PhonemeSequence"
    return task


def epoch_derivative_root(bids_root: str, ref: str, family: str = "default") -> str:
    """Return BIDS derivatives root for the requested epoch family."""
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"family must be one of {SUPPORTED_FAMILIES}, got {family!r}")
    if family == "sentence":
        return bids_root + f"derivatives/epoch(sentence)({ref})"
    return bids_root + f"derivatives/epoch({ref})"


def results_task_key(task: str, family: str = "default") -> str:
    """Return results/hga task directory name for the requested family."""
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"family must be one of {SUPPORTED_FAMILIES}, got {family!r}")
    if family == "sentence":
        return f"{task}(sentence)"
    return task


def _swap_epoch_derivative(epoch_root: str, ref: str, target: str) -> str:
    """Map an epoch derivative root to statistics or shared parcellation.

    Sentence-family roots use ``epoch(sentence)({ref})`` and map statistics to
    ``statistics(sentence)``; both families share ``parcellation/``.
    """
    sentence_token = f"epoch(sentence)({ref})"
    if sentence_token in epoch_root:
        if target == "statistics":
            return epoch_root.replace(sentence_token, "statistics(sentence)")
        return epoch_root.replace(sentence_token, target)
    return epoch_root.replace(f"epoch({ref})", target)


def modality_of(epoch_path) -> str:
    return epoch_path.recording if epoch_path.recording is not None else "sound"


def is_baseline(epoch_path) -> bool:
    return epoch_path.description == "baseline" or epoch_path.processing == "baseline"


def sig_union_by_subject_modality(sig_paths) -> dict[tuple[str, str], set[str]]:
    """Union significant channel names per (subject, modality) over all sig files."""
    import mne

    unions: dict[tuple[str, str], set[str]] = defaultdict(set)
    for epoch_path in sig_paths:
        if is_baseline(epoch_path):
            continue
        epochs = mne.read_epochs(epoch_path, preload=False, verbose=False)
        key = (_normalize_subject_id(epoch_path.subject), modality_of(epoch_path))
        unions[key].update(epochs.ch_names)
    return dict(unions)


def pick_union_channels(
    union: dict[tuple[str, str], set[str]],
    subject: str,
    modality: str,
    available: set[str] | list[str],
) -> list[str]:
    """Return sorted intersection of subject×modality union and available channels."""
    key = (_normalize_subject_id(subject), modality)
    allowed = union.get(key, set())
    available_set = set(available)
    return sorted(allowed & available_set)


def stats_path_candidates(epoch_path, ref: str):
    """Yield the statistics h5 path for an epoch (canonical BIDS task label)."""
    yield epoch_path.copy().update(
        root=_swap_epoch_derivative(str(epoch_path.root), ref, "statistics"),
        datatype=ref,
        task=canonical_task_name(epoch_path.task),
        extension=".h5",
    )


def load_stats_mask(epoch_path, ref: str, epochs, df: pd.DataFrame) -> pd.DataFrame:
    candidates = list(stats_path_candidates(epoch_path, ref))
    last_path = None
    for stats_path in candidates:
        last_path = stats_path
        try:
            with h5py.File(stats_path, "r") as stats:
                mask_data = stats["mask"][:]
                ch_names_stats = [
                    chn.decode("utf-8") for chn in stats["ch_names"][:]
                ]

            mask_df = pd.DataFrame(
                index=ch_names_stats,
                columns=epochs.times,
                data=mask_data,
            )
            mask_long = (
                mask_df.reset_index()
                .melt(id_vars="index", var_name="time", value_name="mask")
                .rename(columns={"index": "channel"})
            )
            mask_long = mask_long[mask_long["channel"].isin(df["channel"])]
            df = df.merge(mask_long, on=["channel", "time"], how="left")
            df["mask"] = df["mask"].fillna(False).astype(bool)
            return df
        except FileNotFoundError:
            continue

    logging.warning(
        "Stats file not found for epoch %s (tried %s), setting mask to False",
        epoch_path,
        last_path,
    )
    df["mask"] = False
    return df


def load_parcellation(epoch_path, ref: str, atlas: str = "aparc2009s") -> pd.DataFrame:
    if atlas not in SUPPORTED_ATLASES:
        raise ValueError(f"atlas must be one of {SUPPORTED_ATLASES}, got {atlas!r}")
    parc_matches = epoch_path.copy().update(
        root=_swap_epoch_derivative(str(epoch_path.root), ref, "parcellation"),
        datatype=ref,
        task=None,
        description=None,
        recording=None,
        processing=None,
        suffix=atlas,
        extension=".csv",
    ).match()
    if not parc_matches:
        raise IndexError("no parcellation file matched")
    return pd.read_csv(parc_matches[0])


def parcellation_subset(parc: pd.DataFrame) -> pd.DataFrame:
    """Return channel-keyed aparc geometry for HGA merge.

    ``x``, ``y``, ``z`` remain template (cvs_avg35 / ``*_t``) display coords.
    Native midpoint and bipolar endpoint coords are emitted when present in the
    parcellation CSV; missing endpoint fields are left as NaN.
    """
    parc = parc.rename(columns={"name": "channel", "center": "label"})
    base_cols = ["channel", "label", "roi", "hemi"]
    missing_base = [col for col in base_cols if col not in parc.columns]
    if missing_base:
        raise ValueError(f"parcellation table missing required columns: {missing_base}")

    out = parc[base_cols].copy()

    if {"x_t", "y_t", "z_t"}.issubset(parc.columns):
        out["x"] = parc["x_t"]
        out["y"] = parc["y_t"]
        out["z"] = parc["z_t"]
    else:
        logging.warning(
            "parcellation missing template midpoint coords (x_t/y_t/z_t); "
            "leaving x/y/z as NaN"
        )
        out["x"] = np.nan
        out["y"] = np.nan
        out["z"] = np.nan

    if {"x", "y", "z"}.issubset(parc.columns):
        out["x_native"] = parc["x"]
        out["y_native"] = parc["y"]
        out["z_native"] = parc["z"]
    else:
        out["x_native"] = np.nan
        out["y_native"] = np.nan
        out["z_native"] = np.nan

    if not set(ENDPOINT_NATIVE_COLS).issubset(parc.columns):
        logging.warning(
            "parcellation missing bipolar endpoint native coords; "
            "leaving endpoint fields as NaN"
        )

    endpoint_renames = {
        "x1": "x1_native",
        "y1": "y1_native",
        "z1": "z1_native",
        "x2": "x2_native",
        "y2": "y2_native",
        "z2": "z2_native",
        "x1_t": "x1_template",
        "y1_t": "y1_template",
        "z1_t": "z1_template",
        "x2_t": "x2_template",
        "y2_t": "y2_template",
        "z2_t": "z2_template",
    }
    for src, dst in endpoint_renames.items():
        if src in parc.columns:
            out[dst] = parc[src]
        else:
            out[dst] = np.nan

    for col in CONTACT_COLS:
        if col in parc.columns:
            out[col] = parc[col]
        else:
            out[col] = np.nan

    if "mix" in parc.columns:
        out["mix"] = parc["mix"]

    return out


def _normalize_subject_id(subject: str) -> str:
    return subject if subject.startswith("sub-") else f"sub-{subject}"


def _filter_epoch_paths(epoch_paths, subjects: list[str] | None):
    if not subjects:
        return epoch_paths
    allowed = {_normalize_subject_id(subject) for subject in subjects}
    return [
        epoch_path
        for epoch_path in epoch_paths
        if _normalize_subject_id(epoch_path.subject) in allowed
    ]


def main(
    bids_root: str,
    band: str,
    ref: str,
    atlas: str = "aparc2009s",
    subjects: list[str] | None = None,
    family: str = "default",
):
    import mne
    from mne_bids import BIDSPath

    epoch_root = epoch_derivative_root(bids_root, ref, family=family)
    logging.info("Packaging family=%s from %s", family, epoch_root)

    sig_paths = BIDSPath(
        root=epoch_root,
        suffix=band,
        datatype="epoch(band)(sig)",
        extension=".h5",
        check=False,
    ).match()
    sig_paths = _filter_epoch_paths(sig_paths, subjects)
    if not sig_paths:
        raise FileNotFoundError(
            f"No epoch(band)(sig) files under {epoch_root} for band={band!r}"
        )

    logging.info("Building subject×modality sig unions from %d sig files", len(sig_paths))
    union = sig_union_by_subject_modality(sig_paths)
    logging.info(
        "Sig unions: %d subject×modality keys, %d total channel slots",
        len(union),
        sum(len(chs) for chs in union.values()),
    )

    zscore_paths = BIDSPath(
        root=epoch_root,
        suffix=band,
        datatype="epoch(band)(zscore)",
        extension=".h5",
        check=False,
    ).match()
    matched_paths = _filter_epoch_paths(zscore_paths, subjects)
    if subjects:
        logging.info(
            "Subject filter %s -> %d zscore epoch files",
            ", ".join(subjects),
            len(matched_paths),
        )

    for epoch_path in tqdm(matched_paths, desc="Processing epochs"):
        if is_baseline(epoch_path):
            continue

        modality = modality_of(epoch_path)
        try:
            parc = load_parcellation(epoch_path, ref, atlas=atlas)
        except (IndexError, FileNotFoundError) as exc:
            logging.warning(
                "No %s parcellation for subject %s: %s, skipping",
                atlas,
                epoch_path.subject,
                exc,
            )
            continue

        epochs = mne.read_epochs(epoch_path, preload=True, verbose=False)
        picks = pick_union_channels(
            union,
            epoch_path.subject,
            modality,
            epochs.ch_names,
        )
        if not picks:
            continue

        epochs = epochs.copy().pick(picks, verbose=False)
        evoked = epochs.average(method=lambda x: np.nanmean(x, axis=0))
        df = evoked.to_data_frame(
            long_format=True,
            scalings={"seeg": 1},
        )

        df = load_stats_mask(epoch_path, ref, epochs, df)

        df.drop(columns=["ch_type"], inplace=True)
        task = canonical_task_name(epoch_path.task)
        df["subject"] = epoch_path.subject
        df["description"] = epoch_path.description
        df["task"] = task
        df["phase"] = epoch_path.processing
        df["modality"] = modality

        parc_sub = parcellation_subset(parc)
        df = df.merge(parc_sub, on="channel", how="left")

        save_path = BIDSPath(
            root=str(results_dir(results_task_key(task, family), ref, atlas)),
            description=epoch_path.description,
            datatype="HGA",
            suffix="time",
            recording=epoch_path.recording,
            task=task,
            subject=epoch_path.subject,
            processing=epoch_path.processing,
            extension=".csv",
            check=False,
        )
        save_path.mkdir(exist_ok=True)
        df.to_csv(save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bids_root",
        default="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/",
        type=str,
    )
    parser.add_argument(
        "--band",
        type=str,
        default="highgamma",
        choices=["highgamma", "gamma", "beta", "alpha", "theta"],
        help="which frequency band to use",
    )
    parser.add_argument(
        "--ref",
        type=str,
        default="bipolar",
        choices=["bipolar", "car"],
        help="reference channel",
    )
    parser.add_argument(
        "--atlas",
        type=str,
        default="aparc2009s",
        choices=list(SUPPORTED_ATLASES),
        help="parcellation atlas suffix under derivatives/parcellation/",
    )
    parser.add_argument(
        "--family",
        type=str,
        default="default",
        choices=list(SUPPORTED_FAMILIES),
        help=(
            "epoch family: default uses epoch({ref})/statistics; "
            "sentence uses epoch(sentence)({ref})/statistics(sentence) "
            "and writes results/hga/{task}(sentence)/"
        ),
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="optional subject ids to package (e.g. D0094 D0071)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main(**vars(args))
