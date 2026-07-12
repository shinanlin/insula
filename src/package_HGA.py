# take HGA signal and save it to pandas

import argparse
import h5py
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


ENDPOINT_NATIVE_COLS = ("x1", "y1", "z1", "x2", "y2", "z2")
ENDPOINT_TEMPLATE_COLS = ("x1_t", "y1_t", "z1_t", "x2_t", "y2_t", "z2_t")
CONTACT_COLS = ("contact_1", "contact_2", "contact_1_label", "contact_2_label")
RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"
SUPPORTED_ATLASES = ("aparc2009s", "hammers")

# Epoch and statistics exports may use different BIDS task labels for the same study.
TASK_STATS_ALIASES: dict[str, list[str]] = {
    "PhonemeSequencing": ["PhonemeSequence"],
    "PhonemeSequence": ["PhonemeSequencing"],
}


def stats_path_candidates(epoch_path, ref: str):
    """Yield statistics h5 paths for an epoch, including known task aliases."""
    base = epoch_path.copy().update(
        root=str(epoch_path.root).replace(f"epoch({ref})", "statistics"),
        datatype=ref,
        extension=".h5",
    )
    seen: set[str] = set()

    def _yield_unique(path):
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        yield path

    yield from _yield_unique(base)
    for alt_task in TASK_STATS_ALIASES.get(epoch_path.task, []):
        yield from _yield_unique(base.copy().update(task=alt_task))


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
            if candidates and str(stats_path) != str(candidates[0]):
                logging.info(
                    "Using stats alias %s for epoch task %s",
                    stats_path,
                    epoch_path.task,
                )
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


def results_dir(task: str, ref: str, atlas: str) -> Path:
    """Packaged HGA output root for a task/reference/atlas combination."""
    if atlas not in SUPPORTED_ATLASES:
        raise ValueError(f"atlas must be one of {SUPPORTED_ATLASES}, got {atlas!r}")
    return RESULTS_ROOT / f"{task}({ref})({atlas})"


def load_parcellation(epoch_path, ref: str, atlas: str = "aparc2009s") -> pd.DataFrame:
    if atlas not in SUPPORTED_ATLASES:
        raise ValueError(f"atlas must be one of {SUPPORTED_ATLASES}, got {atlas!r}")
    parc_matches = epoch_path.copy().update(
        root=str(epoch_path.root).replace(f"epoch({ref})", "parcellation"),
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
):
    import mne
    from mne_bids import BIDSPath

    epoch_paths = BIDSPath(
        root=bids_root + f"derivatives/epoch({ref})",
        suffix=band,
        datatype="epoch(band)(sig)(effective)",
        extension=".h5",
        check=False,
    )

    matched_paths = _filter_epoch_paths(epoch_paths.match(), subjects)
    if subjects:
        logging.info(
            "Subject filter %s -> %d epoch files",
            ", ".join(subjects),
            len(matched_paths),
        )

    for epoch_path in tqdm(matched_paths, desc="Processing subjects"):
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

        epochs = mne.read_epochs(epoch_path, preload=True)
        evoked = epochs.average(method=lambda x: np.nanmean(x, axis=0))
        df = evoked.to_data_frame(
            long_format=True,
            scalings={"seeg": 1},
        )

        df = load_stats_mask(epoch_path, ref, epochs, df)

        df.drop(columns=["ch_type"], inplace=True)
        df["subject"] = epoch_path.subject
        df["description"] = epoch_path.description
        df["task"] = epoch_path.task
        df["phase"] = epoch_path.processing
        df["modality"] = (
            epoch_path.recording if epoch_path.recording is not None else "sound"
        )

        parc_sub = parcellation_subset(parc)
        df = df.merge(parc_sub, on="channel", how="left")

        save_path = BIDSPath(
            root=str(results_dir(epoch_path.task, ref, atlas)),
            description=epoch_path.description,
            datatype="HGA",
            suffix="time",
            recording=epoch_path.recording,
            task=epoch_path.task,
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
        "--subjects",
        nargs="*",
        default=None,
        help="optional subject ids to package (e.g. D0094 D0071)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main(**vars(args))
