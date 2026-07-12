"""Shared output paths for insula analysis pipelines."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"

SUPPORTED_ATLASES = ("aparc2009s", "hammers")


def hga_results_dir(task: str, ref: str, atlas: str) -> Path:
    """Packaged HGA output root for a task/reference/atlas combination."""
    if atlas not in SUPPORTED_ATLASES:
        raise ValueError(f"atlas must be one of {SUPPORTED_ATLASES}, got {atlas!r}")
    return RESULTS_ROOT / f"{task}({ref})({atlas})"
