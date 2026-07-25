"""Strict Hammersmith Insula seed definitions."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import pandas as pd


# These are the six Hammers n30r95 Insula structures. Hemisphere suffixes are
# removed before matching. Do not replace this list with derived AIC/PIC ROIs.
HAMMERS_INSULA_SUBREGIONS: dict[str, str] = {
    "insula anterior short gyrus": "ASG",
    "insula middle short gyrus": "MSG",
    "insula posterior short gyrus": "PSG",
    "insula anterior pole": "AP",
    "insula anterior inferior cortex": "AP",
    "insula anterior long gyrus": "ALG",
    "insula posterior long gyrus": "PLG",
}

# Original Hammers n30r95 bilateral IDs. The CSVs currently expose names, not
# numeric IDs, so these constants document and test the intended atlas set.
HAMMERS_INSULA_IDS: dict[str, tuple[int, int]] = {
    "ASG": (86, 87),
    "MSG": (88, 89),
    "PSG": (90, 91),
    "AP": (92, 93),
    "ALG": (94, 95),
    "PLG": (20, 21),
}


def normalize_hammers_label(value: object) -> str:
    """Normalize a Hammers label while preserving exact anatomical wording."""

    if value is None or pd.isna(value):
        return ""
    label = re.sub(r"\s+", " ", str(value).strip().lower())
    return re.sub(r"\s+[lr]$", "", label)


def hammers_insula_subregion(value: object) -> str | None:
    """Return the six-region abbreviation for an exact Hammers label."""

    return HAMMERS_INSULA_SUBREGIONS.get(normalize_hammers_label(value))


@dataclass(frozen=True)
class SeedRecord:
    channel: str
    contact_1_label: str
    contact_2_label: str
    center_label: str
    contact_1_subregion: str
    contact_2_subregion: str
    center_subregion: str | None
    seed_subregion_mix: bool
    hemi: str | None


def strict_hammers_seed_records(
    parcellation: pd.DataFrame,
    available_channels: Iterable[str] | None = None,
) -> list[SeedRecord]:
    """Select bipolar seeds whose two physical contacts are Hammers Insula.

    The derived ``roi`` and ``mix`` columns are deliberately ignored.
    """

    required = {"name", "contact_1_label", "contact_2_label"}
    missing = sorted(required.difference(parcellation.columns))
    if missing:
        raise ValueError(f"Hammers parcellation missing columns: {missing}")

    allowed = None if available_channels is None else set(available_channels)
    records: list[SeedRecord] = []
    for row in parcellation.itertuples(index=False):
        channel = str(getattr(row, "name"))
        if allowed is not None and channel not in allowed:
            continue
        label_1 = getattr(row, "contact_1_label")
        label_2 = getattr(row, "contact_2_label")
        subregion_1 = hammers_insula_subregion(label_1)
        subregion_2 = hammers_insula_subregion(label_2)
        if subregion_1 is None or subregion_2 is None:
            continue
        center = getattr(row, "center", "")
        records.append(
            SeedRecord(
                channel=channel,
                contact_1_label=str(label_1),
                contact_2_label=str(label_2),
                center_label=str(center),
                contact_1_subregion=subregion_1,
                contact_2_subregion=subregion_2,
                center_subregion=hammers_insula_subregion(center),
                seed_subregion_mix=subregion_1 != subregion_2,
                hemi=(
                    None
                    if not hasattr(row, "hemi") or pd.isna(getattr(row, "hemi"))
                    else str(getattr(row, "hemi"))
                ),
            )
        )
    return records


def strict_hammers_seed_frame(
    parcellation: pd.DataFrame,
    available_channels: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return strict seed metadata as a table."""

    records = strict_hammers_seed_records(parcellation, available_channels)
    columns = list(SeedRecord.__dataclass_fields__)
    return pd.DataFrame(
        [record.__dict__ for record in records], columns=columns
    )
