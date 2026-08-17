"""Shared output paths for insula analysis pipelines."""

from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
IMG_ROOT = PROJECT_ROOT / "img"

SUPPORTED_ATLASES = ("aparc2009s", "hammers")


def img_dir(name: str) -> Path:
    """Publication figure directory under ``img/<name>/`` (created if missing)."""
    path = IMG_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_svg(fig: plt.Figure, path: Path, *, dpi: int = 300, close: bool = False) -> Path:
    """Save a matplotlib figure as SVG only."""
    out = Path(path).with_suffix(".svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return out


def hga_results_dir(
    task: str,
    ref: str = "bipolar",
    atlas: str = "hammers",
) -> Path:
    """Packaged HGA output root: ``results/hga/{task}/``.

    Packaging always uses Hammers parcellation CSVs (``--atlas hammers``).
    ``ref`` and ``atlas`` are accepted for API compatibility but do not appear
    in the output path.
    """
    del ref, atlas  # study default: bipolar + Hammers
    return RESULTS_ROOT / "hga" / task


def decoding_results_dir() -> Path:
    """Canonical decoding output root: ``results/decoding/``."""
    return RESULTS_ROOT / "decoding"


def decoding_task_dir(task: str) -> Path:
    """Task-specific decoding output root: ``results/decoding/{task}/``."""
    return decoding_results_dir() / task


def resolve_decoding_task_root(
    results_root: Path | str,
    task: str,
    *,
    ref: str = "bipolar",
) -> Path | None:
    """Resolve a task score root under method-first or legacy layouts.

    Tries, in order:
    1. ``{results_root}/{task}`` (``results_root`` already ``…/decoding``)
    2. ``{results_root}/decoding/{task}`` (``results_root`` is ``…/results``)
    3. ``{results_root}/{task}(roi)({ref})`` (legacy anatomical / old functional)

    Returns the first existing directory, or ``None`` if none exist.
    """
    base = Path(results_root)
    candidates = (
        base / task,
        base / "decoding" / task,
        base / f"{task}(roi)({ref})",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def parcellation_qc_dir(atlas: str, subject: str) -> Path:
    """Native MRI slice QC for parcellation (Stage 3), keyed by atlas and subject."""
    if atlas not in SUPPORTED_ATLASES:
        raise ValueError(f"atlas must be one of {SUPPORTED_ATLASES}, got {atlas!r}")
    subj = subject if subject.startswith("sub-") else f"sub-{subject}"
    return RESULTS_ROOT / "qc" / atlas / subj


def nmf_results_dir() -> Path:
    """Canonical NMF output root: ``results/nmf/``."""
    return RESULTS_ROOT / "nmf"


def nmf_assignments_path() -> Path:
    """Downstream entry point: electrode → functional_cluster CSV.

    Produced by concat multi-phase NMF and written to
    ``results/nmf/channel_assignments.csv``. Downstream analyses must read this
    path only.
    """
    return nmf_results_dir() / "channel_assignments.csv"


def nmf_exclude_channels_path() -> Path:
    """Channels to drop before concat-NMF (one name per line)."""
    return nmf_results_dir() / "exclude_channels.txt"


def nmf_chosen_k(default: int = 3) -> int:
    """Read k from ``chosen_k.json`` or ``nmf_manifest.json``, else ``default``."""

    import json

    chosen = nmf_results_dir() / "chosen_k.json"
    if chosen.is_file():
        payload = json.loads(chosen.read_text(encoding="utf-8"))
        if "k" in payload:
            return int(payload["k"])
    manifest = nmf_results_dir() / "nmf_manifest.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if "k" in payload:
            return int(payload["k"])
    return default


def nmf_run_dir() -> Path:
    """Canonical NMF run directory (flat ``results/nmf/``)."""
    return nmf_results_dir()


def nmf_nnls_dir() -> Path:
    """Fixed-W NNLS projection cache: ``results/nmf/nnls_projection/``."""
    return nmf_results_dir() / "nnls_projection"
