"""Shared native MRI slice rendering for parcellation QC."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

DEFAULT_FOV_MM = 80.0
MAPER_INSULA_IDS = frozenset({20, 21, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95})
DESTRIEUX_INSULA_FALLBACK_IDS = frozenset({
    11117, 11118, 11148, 11149, 11150,
    12117, 12118, 12148, 12149, 12150,
})
APARC_LUT_CANDIDATES = (
    Path(os.environ.get("FREESURFER_HOME", "")) / "FreeSurferColorLUT.txt",
    Path("/hpc/group/coganlab/nanlinshi/seeg-preprocessing-worktrees/lexical_delay/common/FreeSurferColorLUT.txt"),
    Path("/usr/local/freesurfer/FreeSurferColorLUT.txt"),
)


@dataclass(frozen=True)
class ContourStyle:
    color: str
    linestyle: str
    legend: str


MAPER_CONTOUR = ContourStyle("#c51b29", "--", "insula (MAPER Hammersmith)")
APARC_CONTOUR = ContourStyle("#1764ab", "-", "insula (aparc2009s)")


@dataclass(frozen=True)
class SliceCase:
    subject: str
    name: str
    roi: str
    center_label: str
    mix: bool
    contact_1: str
    contact_2: str
    contact_1_xyz: tuple[float, float, float]
    center_xyz: tuple[float, float, float]
    contact_2_xyz: tuple[float, float, float]
    orig: Path
    fused: Path | None = None


def canonical_bool_mask(path: Path, ids: frozenset[int]) -> tuple[np.ndarray, np.ndarray]:
    image = nib.load(path)
    data = np.asarray(image.dataobj).squeeze()
    mask = np.isin(data.astype(int), list(ids)).astype(np.uint8)
    canonical = nib.as_closest_canonical(nib.Nifti1Image(mask, image.affine))
    return np.asarray(canonical.dataobj).astype(bool), canonical.affine


def load_freesurfer_insula_ids(lut_path: Path | None = None) -> frozenset[int]:
    candidates = [lut_path] if lut_path else list(APARC_LUT_CANDIDATES)
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        ids: set[int] = set()
        for line in candidate.read_text(encoding="utf-8", errors="ignore").splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0].isdigit() and "insula" in fields[1].lower():
                ids.add(int(fields[0]))
        if ids:
            return frozenset(ids)
    return DESTRIEUX_INSULA_FALLBACK_IDS


def default_aparc_seg_path(orig_path: Path) -> Path:
    return orig_path.parent / "aparc.a2009s+aseg.mgz"


def _load_brain_and_mask(
    orig_path: Path,
    mask_path: Path,
    mask_ids: frozenset[int],
    mask_name: str,
) -> tuple[nib.Nifti1Image, np.ndarray, np.ndarray]:
    orig = nib.load(orig_path)
    canonical_orig = nib.as_closest_canonical(orig)
    brain = np.asarray(canonical_orig.dataobj, dtype=float)
    insula_mask, insula_affine = canonical_bool_mask(mask_path, mask_ids)
    if brain.shape != insula_mask.shape:
        raise ValueError(
            f"Geometry mismatch for {orig_path}: orig={brain.shape}, {mask_name}={insula_mask.shape}"
        )
    if not np.allclose(canonical_orig.affine, insula_affine, atol=1e-4):
        raise ValueError(f"Canonical affine mismatch for {orig_path} vs {mask_name}")
    return orig, brain, insula_mask


def load_subject_brain_and_maper_insula_mask(
    orig_path: Path,
    fused_path: Path,
) -> tuple[nib.Nifti1Image, np.ndarray, np.ndarray]:
    return _load_brain_and_mask(orig_path, fused_path, MAPER_INSULA_IDS, "MAPER insula")


def load_subject_brain_and_aparc_insula_mask(
    orig_path: Path,
    aparc_seg_path: Path | None = None,
    aparc_ids: frozenset[int] | None = None,
) -> tuple[nib.Nifti1Image, np.ndarray, np.ndarray]:
    seg_path = aparc_seg_path or default_aparc_seg_path(orig_path)
    if not seg_path.is_file():
        raise FileNotFoundError(f"missing aparc segmentation: {seg_path}")
    ids = aparc_ids or load_freesurfer_insula_ids()
    return _load_brain_and_mask(orig_path, seg_path, ids, "aparc insula")


# Backward-compatible alias used by early imports/tests.
load_subject_brain_and_insula_mask = load_subject_brain_and_maper_insula_mask


def tkras_to_canonical_ijk(orig: nib.Nifti1Image, xyz_mm: tuple[float, float, float]) -> np.ndarray:
    canonical_orig = nib.as_closest_canonical(orig)
    native_ijk = (np.linalg.inv(orig.header.get_vox2ras_tkr()) @ np.r_[np.asarray(xyz_mm, float), 1.0])[:3]
    scanner_ras = nib.affines.apply_affine(orig.affine, native_ijk)
    return nib.affines.apply_affine(np.linalg.inv(canonical_orig.affine), scanner_ras)


def view_limits(
    point: np.ndarray,
    shape: tuple[int, ...],
    affine: np.ndarray,
    dims: tuple[int, int],
    fov_mm: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    voxel_sizes = np.linalg.norm(np.asarray(affine, float)[:3, :3], axis=0)
    limits = []
    for dim in dims:
        half_voxels = fov_mm / (2.0 * float(voxel_sizes[dim]))
        lo = max(-0.5, float(point[dim] - half_voxels))
        hi = min(float(shape[dim] - 0.5), float(point[dim] + half_voxels))
        limits.append((lo, hi))
    return limits[0], limits[1]


def draw_slice_case(
    case: SliceCase,
    orig: nib.Nifti1Image,
    brain: np.ndarray,
    insula_mask: np.ndarray,
    fov_mm: float = DEFAULT_FOV_MM,
    contour: ContourStyle = MAPER_CONTOUR,
) -> plt.Figure:
    canonical_affine = nib.as_closest_canonical(orig).affine
    c1 = tkras_to_canonical_ijk(orig, case.contact_1_xyz)
    center = tkras_to_canonical_ijk(orig, case.center_xyz)
    c2 = tkras_to_canonical_ijk(orig, case.contact_2_xyz)
    slice_center = np.rint(center).astype(int)
    planes = [
        (0, int(slice_center[0]), (1, 2), "sagittal"),
        (1, int(slice_center[1]), (0, 2), "coronal"),
        (2, int(slice_center[2]), (0, 1), "axial"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.4), layout="constrained")
    for ax, (axis, index, dims, title) in zip(axes, planes):
        background = np.take(brain, index, axis=axis).T
        insula_slice = np.take(insula_mask, index, axis=axis).T
        ax.imshow(background, cmap="gray", origin="lower")
        if insula_slice.any():
            ax.contour(
                insula_slice,
                levels=[0.5],
                colors=contour.color,
                linewidths=1.2,
                linestyles=contour.linestyle,
                origin="lower",
            )
        ax.plot(
            [c1[dims[0]], c2[dims[0]]],
            [c1[dims[1]], c2[dims[1]]],
            color="#f0f0f0",
            linewidth=0.9,
            alpha=0.85,
            zorder=3,
        )
        ax.scatter(
            c1[dims[0]], c1[dims[1]], s=82,
            facecolors="none", edgecolors="#00d5ff", linewidths=1.8, zorder=5,
        )
        ax.scatter(
            c2[dims[0]], c2[dims[1]], s=82,
            facecolors="none", edgecolors="#ff4fc3", linewidths=1.8, zorder=5,
        )
        ax.scatter(
            center[dims[0]], center[dims[1]], s=24,
            c="#ffe600", edgecolors="black", linewidths=0.4, zorder=6,
        )
        xlim, ylim = view_limits(center, brain.shape, canonical_affine, dims, fov_mm)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=13)
        ax.axis("off")

    mix_text = "mix=False" if not case.mix else "mix=True"
    linestyle_name = {"-": "solid", "--": "dashed", "-.": "dashdot", ":": "dotted"}.get(
        contour.linestyle, contour.linestyle,
    )
    figure.suptitle(
        f"{case.subject} {case.name} | roi={case.roi} | {mix_text} | center={case.center_label}\n"
        f"{contour.legend} ({linestyle_name}) | cyan=c1 magenta=c2 yellow=midpoint",
        fontsize=11,
    )
    return figure
