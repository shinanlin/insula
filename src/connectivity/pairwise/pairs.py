"""Insula-to-all bipolar pair enumeration."""

from __future__ import annotations

from itertools import combinations
import re
from typing import Iterable

import pandas as pd


def parse_bipolar_contacts(channel: str) -> tuple[str, int, int] | None:
    """Parse ``<shaft><contact>-<contact>`` from a bipolar channel name."""

    match = re.match(r"^(.*?)(\d+)-(\d+)$", str(channel))
    if match is None:
        return None
    return match.group(1), int(match.group(2)), int(match.group(3))


def shares_physical_contact(channel_a: str, channel_b: str) -> bool:
    """Return whether two bipolar channels on one shaft share a contact."""

    first = parse_bipolar_contacts(channel_a)
    second = parse_bipolar_contacts(channel_b)
    if first is None or second is None:
        return False
    stem_a, a1, a2 = first
    stem_b, b1, b2 = second
    return stem_a == stem_b and bool({a1, a2}.intersection({b1, b2}))


def _metadata_by_name(parcellation: pd.DataFrame) -> dict[str, dict[str, object]]:
    if "name" not in parcellation.columns:
        raise ValueError("parcellation requires a 'name' column")
    keep = [
        column
        for column in (
            "name",
            "contact_1_label",
            "contact_2_label",
            "center",
            "roi",
            "mix",
            "hemi",
            "x",
            "y",
            "z",
        )
        if column in parcellation.columns
    ]
    deduped = parcellation[keep].drop_duplicates("name", keep="first")
    return {
        str(row["name"]): row.to_dict()
        for _, row in deduped.iterrows()
    }


def enumerate_insula_to_all_pairs(
    channel_names: Iterable[str],
    seed_frame: pd.DataFrame,
    parcellation: pd.DataFrame,
    effective_channels: set[str] | None = None,
    effective_annotation_available: bool = False,
) -> pd.DataFrame:
    """Enumerate unique pairs with an Insula seed normalized as source."""

    channels = list(channel_names)
    channel_set = set(channels)
    seeds = set(seed_frame.get("channel", pd.Series(dtype=str)).astype(str))
    seeds.intersection_update(channel_set)
    if not seeds:
        raise ValueError("No strict Hammers Insula seeds among available channels")

    seed_meta = seed_frame.set_index("channel", drop=False).to_dict(
        orient="index"
    )
    anatomy = _metadata_by_name(parcellation)
    effective = set() if effective_channels is None else set(effective_channels)

    rows: list[dict[str, object]] = []
    for first, second in combinations(channels, 2):
        first_seed = first in seeds
        second_seed = second in seeds
        if not (first_seed or second_seed):
            continue
        if shares_physical_contact(first, second):
            continue

        if first_seed and not second_seed:
            source, target = first, second
        elif second_seed and not first_seed:
            source, target = second, first
        else:
            source, target = sorted((first, second))

        row: dict[str, object] = {
            "pair_id": f"{source}__{target}",
            "source": source,
            "target": target,
            "source_is_seed": True,
            "target_is_seed": target in seeds,
            "source_effective": (
                source in effective if effective_annotation_available else pd.NA
            ),
            "target_effective": (
                target in effective if effective_annotation_available else pd.NA
            ),
            "effective_annotation_available": effective_annotation_available,
        }
        source_seed = seed_meta[source]
        for key in (
            "contact_1_subregion",
            "contact_2_subregion",
            "center_subregion",
            "seed_subregion_mix",
        ):
            row[f"source_{key}"] = source_seed.get(key)

        for role, channel in (("source", source), ("target", target)):
            meta = anatomy.get(channel, {})
            row[f"{role}_contact_1_label"] = meta.get("contact_1_label")
            row[f"{role}_contact_2_label"] = meta.get("contact_2_label")
            row[f"{role}_center_label"] = meta.get("center")
            row[f"{role}_hemi"] = meta.get("hemi")
        rows.append(row)

    if not rows:
        raise ValueError("No eligible Insula-to-all pairs after exclusions")
    frame = pd.DataFrame(rows)
    return frame.sort_values(["source", "target"]).reset_index(drop=True)
