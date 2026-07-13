"""Export per-epoch electrode coordinates for all zscore channels (Fig 1 coverage)."""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.hga.package_highgamma import (
    _filter_epoch_paths,
    load_parcellation,
    parcellation_subset,
)
from src.paths import SUPPORTED_ATLASES, hga_results_dir

APARC_ROI_MERGE = {
    "PrG": "SMC",
    "PoG": "SMC",
    "Subcentral": "SMC",
}


def _is_baseline(epoch_path) -> bool:
    return epoch_path.description == "baseline" or epoch_path.processing == "baseline"


def _apply_aparc_roi_merge(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for src, dst in APARC_ROI_MERGE.items():
        out.loc[out["roi"] == src, "roi"] = dst
    return out


def _mean_hga_by_channel(epochs) -> dict[str, float]:
    tmin = epochs.tmin + 0.5
    tmax = epochs.tmax - 0.5
    if tmax <= tmin:
        return {ch: np.nan for ch in epochs.ch_names}
    cropped = epochs.crop(tmin, tmax)
    means = cropped.get_data().mean(axis=(0, -1))
    return dict(zip(epochs.ch_names, means))


def _load_sig_channels(epoch_path):
    sig_matches = epoch_path.copy().update(
        datatype="epoch(band)(sig)",
        extension=".h5",
    ).match()
    if not sig_matches:
        return set()
    import mne

    sig_epochs = mne.read_epochs(sig_matches[0], preload=True, verbose=False)
    return set(sig_epochs.ch_names)


def build_coord_dataframe(
    epochs,
    parc_sub: pd.DataFrame,
    sig_channels: set[str],
    *,
    band: str,
    subject: str,
    task: str,
    description: str,
    phase: str,
    modality: str,
    atlas: str,
) -> pd.DataFrame:
    """One row per channel in ``epochs`` with parcellation geometry merged."""
    df = pd.DataFrame({"channel": list(epochs.ch_names)})
    df = df.merge(parc_sub, on="channel", how="left")

    hga_by_channel = _mean_hga_by_channel(epochs)
    df["HGA"] = df["channel"].map(hga_by_channel)
    df["significant"] = df["channel"].isin(sig_channels)
    df["subject"] = subject
    df["task"] = task
    df["band"] = band
    df["description"] = description
    df["phase"] = phase
    df["modality"] = modality

    if atlas == "aparc2009s":
        df = _apply_aparc_roi_merge(df)
    return df


def main(
    bids_root: str,
    band: str,
    ref: str,
    atlas: str = "hammers",
    subjects: list[str] | None = None,
):
    import mne
    from mne_bids import BIDSPath

    if atlas not in SUPPORTED_ATLASES:
        raise ValueError(f"atlas must be one of {SUPPORTED_ATLASES}, got {atlas!r}")

    epoch_paths = BIDSPath(
        root=bids_root + f"derivatives/epoch({ref})",
        suffix=band,
        datatype="epoch(band)(zscore)",
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

    for epoch_path in tqdm(matched_paths, desc="Processing epochs"):
        if _is_baseline(epoch_path):
            continue

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
        parc_sub = parcellation_subset(parc)
        sig_channels = _load_sig_channels(epoch_path)

        df = build_coord_dataframe(
            epochs,
            parc_sub,
            sig_channels,
            band=band,
            subject=epoch_path.subject,
            task=epoch_path.task,
            description=epoch_path.description,
            phase=epoch_path.processing,
            modality=(
                epoch_path.recording if epoch_path.recording is not None else "sound"
            ),
            atlas=atlas,
        )

        save_path = BIDSPath(
            root=str(hga_results_dir(epoch_path.task, ref, atlas)),
            description=epoch_path.description,
            datatype="HGA",
            suffix="coord",
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
    parser = argparse.ArgumentParser(
        description="Package all zscore-channel electrode coords for Fig 1 coverage maps."
    )
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
        default="hammers",
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
