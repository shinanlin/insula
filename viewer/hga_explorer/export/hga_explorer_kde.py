"""Build ROI-level static KDE source points for the Insula HGA Explorer."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def electrode_hga_weight(electrode: dict, task: str = "all") -> float | None:
    hga_by_task = electrode.get("hga_by_task") or {}
    if task == "all":
        values = [abs(value) for value in hga_by_task.values() if value is not None]
    else:
        value = hga_by_task.get(task)
        values = [abs(value)] if value is not None else []
    if not values:
        return None
    return sum(values) / len(values)


def build_roi_mean_sources(
    electrodes: list[dict],
    task: str = "all",
) -> dict[str, Any]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for electrode in electrodes:
        grouped[electrode["roi"]].append(electrode)

    sources = []
    for roi, members in sorted(grouped.items()):
        weights = [
            weight
            for member in members
            if (weight := electrode_hga_weight(member, task=task)) is not None
        ]
        if not weights:
            continue
        weight = sum(weights) / len(weights)
        sources.append({
            "roi": roi,
            "x": sum(member["x"] for member in members) / len(members),
            "y": sum(member["y"] for member in members) / len(members),
            "z": sum(member["z"] for member in members) / len(members),
            "weight": weight,
            "n_electrodes": len(members),
        })

    max_weight = max((source["weight"] for source in sources), default=1.0)
    if max_weight <= 0:
        max_weight = 1.0
    for source in sources:
        source["weight"] = source["weight"] / max_weight

    return {"sources": sources}
