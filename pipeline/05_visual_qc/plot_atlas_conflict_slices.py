#!/usr/bin/env python3
"""Native slice QC for aparc-only and MAPER-only bipolar electrodes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import nibabel as nib
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
EXTRACTOR_PATH = REPO_ROOT / "pipeline" / "04_extract_labels" / "extract_maper_parcellation.py"
SPEC = importlib.util.spec_from_file_location("extract_maper_parcellation", EXTRACTOR_PATH)
EXTRACTOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = EXTRACTOR
SPEC.loader.exec_module(EXTRACTOR)

DEFAULT_MANIFEST = Path("/cwork/ns458/maper_run/manifests/maper_extract_all_20260708T191620Z.tsv")
DEFAULT_OUTPUT_ROOT = Path("/cwork/ns458/maper_run/qc_slices")
DEFAULT_FOV_MM = 80.0
CONFLICT_AGREEMENTS = ("aparc_only", "maper_only")
MAPER_INSULA_IDS = frozenset({20, 21, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95})
DESTRIEUX_INSULA_FALLBACK_IDS = frozenset({
    11117, 11118, 11148, 11149, 11150,
    12117, 12118, 12148, 12149, 12150,
})
DESTRIEUX_CIRCULAR_INSULA_FALLBACK_IDS = frozenset({
    11148, 11149, 11150,
    12148, 12149, 12150,
})
# Destrieux opercular cortex overlying the insula. These are not insula labels,
# but drawing them helps spot electrodes sitting on IFG/STG opercular lids.
DESTRIEUX_OPERCULUM_IDS = frozenset({
    11104, 11112, 11114, 11135, 11136,
    12104, 12112, 12114, 12135, 12136,
})
APARC_LUT_CANDIDATES = (
    Path(os.environ.get("FREESURFER_HOME", "")) / "FreeSurferColorLUT.txt",
    Path("/hpc/group/coganlab/nanlinshi/seeg-preprocessing-worktrees/lexical_delay/common/FreeSurferColorLUT.txt"),
    Path("/usr/local/freesurfer/FreeSurferColorLUT.txt"),
)


@dataclass(frozen=True)
class QCCase:
    subject: str
    name: str
    tasks: tuple[str, ...]
    agreement: str
    roi: str
    aparc_label: str
    maper_status: str
    maper_region6: str
    maper_ap: str
    confidence: float
    contact_1: str
    contact_2: str
    contact_1_xyz: tuple[float, float, float]
    center_xyz: tuple[float, float, float]
    contact_2_xyz: tuple[float, float, float]
    orig: Path
    fused: Path
    warning: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--agreements", nargs="+", default=list(CONFLICT_AGREEMENTS))
    parser.add_argument("--channel-list", type=Path)
    parser.add_argument("--max-per-class", type=int)
    parser.add_argument("--fov-mm", type=float, default=DEFAULT_FOV_MM)
    parser.add_argument("--aparc-lut", type=Path)
    return parser.parse_args()


def safe_name(value: object) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))


def clean_text(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def pure_aparc_insula(roi: object) -> bool:
    return clean_text(roi).strip() == "INS"


def aparc_insula_roi(roi: object) -> bool:
    return "INS" in [piece.strip() for piece in clean_text(roi).split("–")]


def pure_conflict_row(row: pd.Series) -> bool:
    agreement = clean_text(row.get("maper_atlas_agreement", ""))
    if agreement == "aparc_only":
        return pure_aparc_insula(row.get("roi", ""))
    if agreement == "maper_only":
        return clean_text(row.get("maper_insula_status", "")).strip() == "core"
    if agreement == "concordant_insula":
        return (
            aparc_insula_roi(row.get("roi", ""))
            and clean_text(row.get("maper_insula_status", "")).strip() == "core"
        )
    return False


def load_channel_list(path: Path) -> set[str]:
    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path)
        if "channel" not in table.columns:
            raise ValueError(f"{path} must contain a 'channel' column")
        return set(table["channel"].astype(str).str.strip())
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def filter_rows_by_channel_list(rows: pd.DataFrame, channels: set[str]) -> pd.DataFrame:
    if rows.empty or not channels:
        return rows
    return rows[rows["name"].astype(str).isin(channels)].copy()


def missing_channels(requested: set[str], rows: pd.DataFrame) -> list[str]:
    if not requested:
        return []
    if rows.empty or "name" not in rows.columns:
        return sorted(requested)
    found = set(rows["name"].astype(str))
    return sorted(requested - found)


def endpoint_names(row: pd.Series, subject: str) -> tuple[str, str]:
    if {"contact_1", "contact_2"}.issubset(row.index) and pd.notna(row["contact_1"]) and pd.notna(row["contact_2"]):
        return (
            EXTRACTOR.strip_subject_prefix(row["contact_1"], subject),
            EXTRACTOR.strip_subject_prefix(row["contact_2"], subject),
        )
    return EXTRACTOR.split_bipolar_name(row["name"], subject)


def load_freesurfer_ids_by_label_substring(
    substring: str,
    lut_path: Path | None = None,
    fallback: frozenset[int] = frozenset(),
) -> frozenset[int]:
    candidates = [lut_path] if lut_path else list(APARC_LUT_CANDIDATES)
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        ids: set[int] = set()
        for line in candidate.read_text(encoding="utf-8", errors="ignore").splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0].isdigit() and substring in fields[1].lower():
                ids.add(int(fields[0]))
        if ids:
            return frozenset(ids)
    return fallback


def load_freesurfer_insula_ids(lut_path: Path | None = None) -> frozenset[int]:
    return load_freesurfer_ids_by_label_substring(
        "insula",
        lut_path=lut_path,
        fallback=DESTRIEUX_INSULA_FALLBACK_IDS,
    )


def load_freesurfer_circular_insula_ids(lut_path: Path | None = None) -> frozenset[int]:
    return load_freesurfer_ids_by_label_substring(
        "circular_insula",
        lut_path=lut_path,
        fallback=DESTRIEUX_CIRCULAR_INSULA_FALLBACK_IDS,
    )


def read_conflict_rows(
    manifest_path: Path,
    agreements: set[str],
    subjects: set[str] | None = None,
    channel_list: set[str] | None = None,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, sep="\t")
    ready = manifest[manifest["status"].eq("ready")].copy()
    if subjects is not None:
        ready = ready[ready["subject"].isin(subjects)]

    frames: list[pd.DataFrame] = []
    required = {
        "subject", "name", "x", "y", "z", "roi", "center",
        "contact_1", "contact_2", "maper_insula_status",
        "maper_region6_consensus", "maper_ap_consensus",
        "maper_atlas_agreement", "maper_center_winner_vote_fraction",
        "maper_center_insula_vote_fraction",
    }
    for row in ready.itertuples(index=False):
        header = pd.read_csv(row.output, nrows=0).columns
        table = pd.read_csv(row.output, usecols=[column for column in header if column in required])
        table = table[table["maper_atlas_agreement"].isin(agreements)].copy()
        if not table.empty:
            table = table[table.apply(pure_conflict_row, axis=1)].copy()
        if table.empty:
            continue
        table["task"] = row.task
        table["orig"] = row.orig
        table["fused"] = row.fused
        table["contacts_tsv"] = row.contacts_tsv
        table["source_output"] = row.output
        frames.append(table)
    if not frames:
        return pd.DataFrame()
    rows = pd.concat(frames, ignore_index=True)
    if channel_list:
        rows = filter_rows_by_channel_list(rows, channel_list)
    return rows


def add_endpoint_coordinates(rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.copy()
    contact_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    midpoint_scale_cache: dict[str, float] = {}
    endpoint_rows: list[dict[str, object]] = []

    for row in output.itertuples(index=False):
        row_dict = row._asdict()
        subject = str(row.subject)
        contacts_path = str(row.contacts_tsv)
        key = (subject, contacts_path)
        if key not in contact_cache:
            contact_cache[key] = EXTRACTOR.load_contacts(Path(contacts_path), subject)
        contact_1, contact_2 = endpoint_names(pd.Series(row_dict), subject)
        contacts = contact_cache[key]
        if contact_1 not in contacts or contact_2 not in contacts:
            raise KeyError(f"Missing endpoints for {subject} {row.name}: {contact_1}, {contact_2}")
        scale_key = str(row.source_output)
        if scale_key not in midpoint_scale_cache:
            subject_rows = pd.read_csv(scale_key, usecols=["x", "y", "z"])
            midpoint_scale_cache[scale_key] = EXTRACTOR.coordinate_scale_to_mm(
                subject_rows[["x", "y", "z"]].to_numpy(float)
            )
        center = EXTRACTOR.validate_mm(
            np.array([row.x, row.y, row.z], dtype=float) * midpoint_scale_cache[scale_key]
        )
        endpoint_rows.append({
            "contact_1_clean": contact_1,
            "contact_2_clean": contact_2,
            "contact_1_x": contacts[contact_1][0],
            "contact_1_y": contacts[contact_1][1],
            "contact_1_z": contacts[contact_1][2],
            "center_mm_x": center[0],
            "center_mm_y": center[1],
            "center_mm_z": center[2],
            "contact_2_x": contacts[contact_2][0],
            "contact_2_y": contacts[contact_2][1],
            "contact_2_z": contacts[contact_2][2],
        })
    return pd.concat([output.reset_index(drop=True), pd.DataFrame(endpoint_rows)], axis=1)


def _max_coordinate_delta(group: pd.DataFrame, columns: list[str]) -> float:
    values = group[columns].to_numpy(float)
    if len(values) <= 1:
        return 0.0
    return float(np.nanmax(np.ptp(values, axis=0)))


def _row_to_case(row: pd.Series, tasks: tuple[str, ...], warning: str = "") -> QCCase:
    confidence = row.get("maper_center_winner_vote_fraction", np.nan)
    if pd.isna(confidence):
        confidence = row.get("maper_center_insula_vote_fraction", np.nan)
    return QCCase(
        subject=str(row["subject"]),
        name=str(row["name"]),
        tasks=tasks,
        agreement=str(row["maper_atlas_agreement"]),
        roi=clean_text(row.get("roi", "")),
        aparc_label=clean_text(row.get("center", "")),
        maper_status=clean_text(row.get("maper_insula_status", "")),
        maper_region6=clean_text(row.get("maper_region6_consensus", "")),
        maper_ap=clean_text(row.get("maper_ap_consensus", "")),
        confidence=float(confidence) if pd.notna(confidence) else np.nan,
        contact_1=str(row["contact_1_clean"]),
        contact_2=str(row["contact_2_clean"]),
        contact_1_xyz=tuple(float(row[f"contact_1_{axis}"]) for axis in "xyz"),
        center_xyz=tuple(float(row[f"center_mm_{axis}"]) for axis in "xyz"),
        contact_2_xyz=tuple(float(row[f"contact_2_{axis}"]) for axis in "xyz"),
        orig=Path(str(row["orig"])),
        fused=Path(str(row["fused"])),
        warning=warning,
    )


def build_qc_cases(rows: pd.DataFrame, max_per_class: int | None = None) -> list[QCCase]:
    if rows.empty:
        return []
    cases: list[QCCase] = []
    coordinate_columns = [
        "contact_1_x", "contact_1_y", "contact_1_z",
        "center_mm_x", "center_mm_y", "center_mm_z",
        "contact_2_x", "contact_2_y", "contact_2_z",
    ]
    field_columns = [
        "maper_atlas_agreement", "roi", "center", "maper_insula_status",
        "maper_region6_consensus", "maper_ap_consensus",
        "contact_1_clean", "contact_2_clean", "orig", "fused",
    ]
    for (_, _), group in rows.groupby(["subject", "name"], sort=True):
        group = group.sort_values("task")
        field_mismatch = any(group[column].astype(str).nunique(dropna=False) > 1 for column in field_columns if column in group)
        coordinate_mismatch = _max_coordinate_delta(group, coordinate_columns) > 1e-6
        if field_mismatch or coordinate_mismatch:
            for _, row in group.iterrows():
                cases.append(_row_to_case(row, tasks=(str(row["task"]),), warning="task_specific_coordinate_or_label_mismatch"))
        else:
            row = group.iloc[0]
            cases.append(_row_to_case(row, tasks=tuple(group["task"].astype(str).unique())))

    if max_per_class is not None:
        kept: list[QCCase] = []
        counts: dict[str, int] = {}
        for case in cases:
            count = counts.get(case.agreement, 0)
            if count < max_per_class:
                kept.append(case)
                counts[case.agreement] = count + 1
        cases = kept
    return cases


def canonical_bool_mask(path: Path, ids: frozenset[int]) -> tuple[np.ndarray, np.ndarray]:
    image = nib.load(path)
    data = np.asarray(image.dataobj).squeeze()
    mask = np.isin(data.astype(int), list(ids)).astype(np.uint8)
    canonical = nib.as_closest_canonical(nib.Nifti1Image(mask, image.affine))
    return np.asarray(canonical.dataobj).astype(bool), canonical.affine


def load_subject_images(
    case: QCCase,
    aparc_ids: frozenset[int],
    circular_insula_ids: frozenset[int],
) -> tuple[nib.Nifti1Image, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    orig = nib.load(case.orig)
    canonical_orig = nib.as_closest_canonical(orig)
    brain = np.asarray(canonical_orig.dataobj, dtype=float)
    aparc_path = case.orig.parent / "aparc.a2009s+aseg.mgz"
    if not aparc_path.is_file():
        raise FileNotFoundError(f"Missing aparc.a2009s+aseg.mgz next to {case.orig}")
    aparc_mask, aparc_affine = canonical_bool_mask(aparc_path, aparc_ids)
    circular_mask, circular_affine = canonical_bool_mask(aparc_path, circular_insula_ids)
    maper_mask, maper_affine = canonical_bool_mask(case.fused, MAPER_INSULA_IDS)
    operc_mask, operc_affine = canonical_bool_mask(aparc_path, DESTRIEUX_OPERCULUM_IDS)
    if brain.shape != aparc_mask.shape or brain.shape != circular_mask.shape or brain.shape != maper_mask.shape or brain.shape != operc_mask.shape:
        raise ValueError(
            f"Geometry mismatch for {case.subject}: orig={brain.shape}, aparc={aparc_mask.shape}, "
            f"circular={circular_mask.shape}, maper={maper_mask.shape}, operc={operc_mask.shape}"
        )
    if not np.allclose(canonical_orig.affine, aparc_affine, atol=1e-4) or not np.allclose(canonical_orig.affine, circular_affine, atol=1e-4) or not np.allclose(canonical_orig.affine, maper_affine, atol=1e-4) or not np.allclose(canonical_orig.affine, operc_affine, atol=1e-4):
        raise ValueError(f"Canonical affine mismatch for {case.subject}")
    return orig, brain, aparc_mask, circular_mask, maper_mask, operc_mask


def tkras_to_canonical_ijk(orig: nib.Nifti1Image, xyz_mm: tuple[float, float, float]) -> np.ndarray:
    canonical_orig = nib.as_closest_canonical(orig)
    native_ijk = (np.linalg.inv(orig.header.get_vox2ras_tkr()) @ np.r_[np.asarray(xyz_mm, float), 1.0])[:3]
    scanner_ras = nib.affines.apply_affine(orig.affine, native_ijk)
    return nib.affines.apply_affine(np.linalg.inv(canonical_orig.affine), scanner_ras)


def view_limits(point: np.ndarray, shape: tuple[int, ...], affine: np.ndarray, dims: tuple[int, int], fov_mm: float) -> tuple[tuple[float, float], tuple[float, float]]:
    voxel_sizes = np.linalg.norm(np.asarray(affine, float)[:3, :3], axis=0)
    limits = []
    for dim in dims:
        half_voxels = fov_mm / (2.0 * float(voxel_sizes[dim]))
        lo = max(-0.5, float(point[dim] - half_voxels))
        hi = min(float(shape[dim] - 0.5), float(point[dim] + half_voxels))
        limits.append((lo, hi))
    return limits[0], limits[1]


def draw_case(
    case: QCCase,
    orig: nib.Nifti1Image,
    brain: np.ndarray,
    aparc_mask: np.ndarray,
    circular_mask: np.ndarray,
    maper_mask: np.ndarray,
    operc_mask: np.ndarray,
    fov_mm: float,
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
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    for ax, (axis, index, dims, title) in zip(axes, planes):
        background = np.take(brain, index, axis=axis).T
        aparc_slice = np.take(aparc_mask, index, axis=axis).T
        circular_slice = np.take(circular_mask, index, axis=axis).T
        maper_slice = np.take(maper_mask, index, axis=axis).T
        operc_slice = np.take(operc_mask, index, axis=axis).T
        ax.imshow(background, cmap="gray", origin="lower")
        if operc_slice.any():
            ax.contour(operc_slice, levels=[0.5], colors="#2ca02c", linewidths=1.2, linestyles=":", origin="lower")
        if aparc_slice.any():
            ax.contour(aparc_slice, levels=[0.5], colors="#1764ab", linewidths=1.2, origin="lower")
        if circular_slice.any():
            ax.contour(circular_slice, levels=[0.5], colors="#ff8c00", linewidths=1.35, linestyles="-.", origin="lower")
        if maper_slice.any():
            ax.contour(maper_slice, levels=[0.5], colors="#c51b29", linewidths=1.2, linestyles="--", origin="lower")
        ax.plot([c1[dims[0]], c2[dims[0]]], [c1[dims[1]], c2[dims[1]]], color="#f0f0f0", linewidth=0.9, alpha=0.85, zorder=3)
        ax.scatter(c1[dims[0]], c1[dims[1]], s=82, facecolors="none", edgecolors="#00d5ff", linewidths=1.8, zorder=5)
        ax.scatter(c2[dims[0]], c2[dims[1]], s=82, facecolors="none", edgecolors="#ff4fc3", linewidths=1.8, zorder=5)
        ax.scatter(center[dims[0]], center[dims[1]], s=24, c="#ffe600", edgecolors="black", linewidths=0.4, zorder=6)
        xlim, ylim = view_limits(center, brain.shape, canonical_affine, dims, fov_mm)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=13)
        ax.axis("off")

    confidence = "NA" if not np.isfinite(case.confidence) else f"{case.confidence:.2f}"
    title = (
        f"{case.subject} {case.name} | {case.agreement} | aparc={case.roi}/{case.aparc_label} | "
        f"MAPER={case.maper_status} {case.maper_region6 or 'noninsula'} {case.maper_ap}"
    )
    subtitle = (
        f"tasks={','.join(case.tasks)} | conf={confidence} | "
        "blue solid=aparc insula | red dashed=MAPER insula | "
        "orange dashdot=aparc circular sulcus | green dotted=aparc operculum | "
        "cyan=c1 magenta=c2 yellow=midpoint"
    )
    if case.warning:
        subtitle += f" | warning={case.warning}"
    figure.suptitle(title + "\n" + textwrap.shorten(subtitle, width=190, placeholder="..."), fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    return figure


def case_index_row(case: QCCase, output_path: Path, error: str = "") -> dict[str, object]:
    return {
        "subject": case.subject,
        "channel": case.name,
        "tasks": ";".join(case.tasks),
        "agreement": case.agreement,
        "aparc_roi": case.roi,
        "aparc_label": case.aparc_label,
        "maper_insula_status": case.maper_status,
        "maper_region6_consensus": case.maper_region6,
        "maper_ap_consensus": case.maper_ap,
        "maper_center_confidence": case.confidence,
        "contact_1": case.contact_1,
        "contact_1_x": case.contact_1_xyz[0],
        "contact_1_y": case.contact_1_xyz[1],
        "contact_1_z": case.contact_1_xyz[2],
        "midpoint_x": case.center_xyz[0],
        "midpoint_y": case.center_xyz[1],
        "midpoint_z": case.center_xyz[2],
        "contact_2": case.contact_2,
        "contact_2_x": case.contact_2_xyz[0],
        "contact_2_y": case.contact_2_xyz[1],
        "contact_2_z": case.contact_2_xyz[2],
        "orig": str(case.orig),
        "fused": str(case.fused),
        "png": str(output_path) if output_path else "",
        "warning": case.warning,
        "error": error,
    }


def write_outputs(
    cases: list[QCCase],
    output_dir: Path,
    fov_mm: float,
    aparc_ids: frozenset[int],
    circular_insula_ids: frozenset[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    png_root = output_dir / "png" / "by_subject"
    pdf_root = output_dir / "pdf" / "by_subject"
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    png_root.mkdir(parents=True, exist_ok=True)
    pdf_root.mkdir(parents=True, exist_ok=True)

    image_cache: dict[tuple[Path, Path], tuple[nib.Nifti1Image, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    index_rows: list[dict[str, object]] = []
    for subject, subject_cases in pd.Series(cases).groupby([case.subject for case in cases], sort=True):
        subject_png_dir = png_root / safe_name(subject)
        subject_png_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = pdf_root / f"{safe_name(subject)}_atlas_conflicts.pdf"
        with PdfPages(pdf_path) as pdf:
            for case in subject_cases.tolist():
                output_path = subject_png_dir / f"{safe_name(case.name)}_{safe_name('_'.join(case.tasks))}_{case.agreement}.png"
                try:
                    cache_key = (case.orig, case.fused)
                    if cache_key not in image_cache:
                        image_cache[cache_key] = load_subject_images(case, aparc_ids, circular_insula_ids)
                    figure = draw_case(case, *image_cache[cache_key], fov_mm=fov_mm)
                    figure.savefig(output_path, dpi=180, bbox_inches="tight")
                    pdf.savefig(figure, bbox_inches="tight")
                    plt.close(figure)
                    index_rows.append(case_index_row(case, output_path))
                except Exception as exc:  # noqa: BLE001 - failed electrodes belong in the index.
                    index_rows.append(case_index_row(case, Path(""), error=f"{type(exc).__name__}: {exc}"))

    index = pd.DataFrame(index_rows)
    if index.empty:
        summary = pd.DataFrame([{"cases": 0, "successful_png": 0, "failed": 0}])
    else:
        summary = (
            index.groupby(["agreement", "maper_insula_status"], dropna=False)
            .size().rename("cases").reset_index()
        )
        overview = pd.DataFrame([{
            "agreement": "ALL",
            "maper_insula_status": "ALL",
            "cases": len(index),
            "successful_png": int(index["error"].eq("").sum()),
            "failed": int(index["error"].ne("").sum()),
            "subjects": int(index["subject"].nunique()),
        }])
        summary = pd.concat([overview, summary], ignore_index=True)
    index.to_csv(output_dir / "atlas_conflict_slice_index.csv", index=False)
    summary.to_csv(output_dir / "atlas_conflict_slice_summary.csv", index=False)
    return index, summary


def main() -> None:
    args = parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (args.output_root / stamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = set(args.subjects) if args.subjects else None
    channel_list = load_channel_list(args.channel_list) if args.channel_list else None
    rows = read_conflict_rows(args.manifest, set(args.agreements), subjects, channel_list=channel_list)
    missing = missing_channels(channel_list or set(), rows)
    if missing:
        missing_path = output_dir / "channel_list_missing.csv"
        pd.DataFrame({"channel": missing}).to_csv(missing_path, index=False)
        print(f"Warning: {len(missing)} channel(s) from --channel-list not found in manifest/filtered rows")
        print(f"Wrote: {missing_path}")
    rows = add_endpoint_coordinates(rows) if not rows.empty else rows
    cases = build_qc_cases(rows, max_per_class=args.max_per_class)
    aparc_ids = load_freesurfer_insula_ids(args.aparc_lut)
    circular_insula_ids = load_freesurfer_circular_insula_ids(args.aparc_lut)
    index, summary = write_outputs(cases, output_dir, args.fov_mm, aparc_ids, circular_insula_ids)
    if missing:
        summary = pd.concat([
            summary,
            pd.DataFrame([{
                "agreement": "channel_list_missing",
                "maper_insula_status": "ALL",
                "cases": len(missing),
                "successful_png": 0,
                "failed": len(missing),
                "subjects": pd.NA,
            }]),
        ], ignore_index=True)
        summary.to_csv(output_dir / "atlas_conflict_slice_summary.csv", index=False)
    print(f"Output: {output_dir}")
    print(f"Cases: {len(cases)}")
    print(summary.to_string(index=False))
    failed = int(index["error"].ne("").sum()) if not index.empty else 0
    if failed:
        raise SystemExit(f"{failed} slice QC cases failed; see atlas_conflict_slice_index.csv")


if __name__ == "__main__":
    main()
