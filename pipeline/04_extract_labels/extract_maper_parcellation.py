#!/usr/bin/env python3
"""Shared MAPER parcellation coordinate helpers.

This module intentionally keeps the small, tested helper surface used by the
native slice QC tooling. The full cohort MAPER extraction outputs are already
materialized under BIDS derivatives; slice QC reads those tables and uses these
helpers only for contact-table coordinate handling.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


CONTACT_NAME_COLUMNS = ("name", "contact", "electrode", "label")
COORD_COLUMNS = ("x", "y", "z")


def strip_subject_prefix(value: object, subject: str) -> str:
    """Return an electrode/contact name without a leading subject prefix."""
    text = str(value).strip()
    if not text:
        return text
    prefixes = {subject, f"sub-{subject}"}
    for prefix in prefixes:
        for separator in ("_", "-"):
            token = f"{prefix}{separator}"
            if text.startswith(token):
                return text[len(token):]
    return text


def split_bipolar_name(name: object, subject: str) -> tuple[str, str]:
    """Split a bipolar channel name into two monopolar endpoint names.

    Handles both canonical names such as ``D0044_RI1-2`` and already stripped
    names such as ``RI1-2``. The second endpoint inherits the alpha prefix from
    the first endpoint when needed.
    """
    channel = strip_subject_prefix(name, subject)
    if "-" not in channel:
        raise ValueError(f"Cannot split non-bipolar channel name: {name}")
    first, second = channel.split("-", 1)
    first = first.strip()
    second = second.strip()
    if not first or not second:
        raise ValueError(f"Cannot split malformed bipolar channel name: {name}")
    if not re.search(r"[A-Za-z]", second):
        prefix = re.match(r"^(.*?)(\d+[A-Za-z]*)$", first)
        if prefix:
            second = f"{prefix.group(1)}{second}"
    return first, second


def coordinate_scale_to_mm(values: np.ndarray) -> float:
    """Infer whether a whole coordinate table is stored in meters or mm.

    Existing BIDS electrode tables in this workspace use meters when every
    finite absolute coordinate is smaller than 10. MAPER derivative CSVs use mm.
    The decision is deliberately table-level rather than point-level so contacts
    near the AC are not accidentally scaled by 1000 independently.
    """
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    if not finite.any():
        return 1.0
    return 1000.0 if float(np.nanmax(np.abs(array[finite]))) < 10.0 else 1.0


def validate_mm(values: np.ndarray) -> np.ndarray:
    """Return finite coordinates as float millimeters."""
    coords = np.asarray(values, dtype=float)
    if coords.shape[-1] != 3:
        raise ValueError(f"Expected 3 coordinates, got shape {coords.shape}")
    if not np.isfinite(coords).all():
        raise ValueError(f"Non-finite coordinate values: {coords}")
    return coords


def _contact_name_column(table: pd.DataFrame) -> str:
    for column in CONTACT_NAME_COLUMNS:
        if column in table.columns:
            return column
    raise ValueError(f"Contact table must include one of {CONTACT_NAME_COLUMNS}")


def load_contacts(path: Path, subject: str) -> dict[str, np.ndarray]:
    """Load a BIDS contacts/electrodes TSV as ``name -> xyz_mm``."""
    table = pd.read_csv(path, sep="\t")
    name_column = _contact_name_column(table)
    missing = [column for column in COORD_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"{path} is missing coordinate columns: {missing}")
    scale = coordinate_scale_to_mm(table[list(COORD_COLUMNS)].to_numpy(float))
    contacts: dict[str, np.ndarray] = {}
    for row in table.itertuples(index=False):
        row_dict = row._asdict()
        name = strip_subject_prefix(row_dict[name_column], subject)
        raw = np.array([row_dict["x"], row_dict["y"], row_dict["z"]], dtype=float) * scale
        if not np.isfinite(raw).all():
            continue
        coords = validate_mm(raw)
        contacts[name] = coords
    return contacts
