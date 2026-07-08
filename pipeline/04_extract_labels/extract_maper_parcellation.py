#!/usr/bin/env python3
"""Extract full Hammers n30r95 labels at bipolar endpoints and midpoint.

The fused MAPER segmentation, tissue-class image, and 30 propagated atlas
segmentations already live in the subject's native FreeSurfer conformed grid.
Input electrode coordinates are native FreeSurfer tkRAS millimetres and are
converted with ``orig.mgz``'s ``vox2ras_tkr`` only.

The aparc table is passed through unchanged. All new columns use a ``maper_``
prefix so Hammers/MAPER remains a parallel atlas rather than overwriting the
FreeSurfer/Destrieux result.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


INSULA_IDS = frozenset({20, 21, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95})
REGION6 = {
    20: "PLG", 21: "PLG",
    86: "ASG", 87: "ASG",
    88: "MSG", 89: "MSG",
    90: "PSG", 91: "PSG",
    92: "pole", 93: "pole",
    94: "ALG", 95: "ALG",
}
AP = {
    20: "Posterior", 21: "Posterior",
    86: "Anterior", 87: "Anterior",
    88: "Anterior", 89: "Anterior",
    90: "Anterior", 91: "Anterior",
    92: "Anterior", 93: "Anterior",
    94: "Posterior", 95: "Posterior",
}

# Validated against D0044 FreeSurfer aparc/aseg: CSF code 1 agreed 94.7%,
# cortical/subcortical GM code 2 agreed 73.9%, and WM code 3 agreed 92.4%.
TISSUE_CODES = {0: "Outside", 1: "CSF", 2: "GM", 3: "WM"}
CORPUS_CALLOSUM_ID = 44
VENTRICLE_IDS = frozenset({45, 46, 47, 48, 49})
LOCATIONS = ("contact_1", "center", "contact_2")


def load_lut(path: Path) -> dict[int, str]:
    lut: dict[int, str] = {}
    pattern = re.compile(r"<index>(\d+)</index><name>(.*?)</name>")
    with path.open() as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                lut[int(match.group(1))] = match.group(2)
    missing = sorted(set(range(1, 96)) - set(lut))
    if missing:
        raise ValueError(f"LUT is missing Hammers IDs: {missing}")
    return lut


def coordinate_scale_to_mm(xyz: np.ndarray) -> float:
    """Infer one unit scale for an entire coordinate table.

    BIDS tables in this cohort are internally uniform but may use metres or
    millimetres.  Inferring once per table avoids misclassifying a valid
    millimetre point close to the AC as metres.
    """
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError(f"Invalid coordinate table with shape {xyz.shape}")
    return 1000.0 if np.max(np.abs(xyz)) < 10 else 1.0


def validate_mm(xyz: np.ndarray) -> np.ndarray:
    """Validate coordinates already guaranteed to be in millimetres."""
    xyz = np.asarray(xyz, dtype=float)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError(f"Invalid millimetre coordinate: {xyz!r}")
    return xyz


def strip_subject_prefix(name: object, subject: str) -> str:
    value = str(name)
    prefix = f"{subject}_"
    return value[len(prefix):] if value.startswith(prefix) else value


def split_bipolar_name(name: object, subject: str) -> tuple[str, str]:
    """Recover physical endpoint names from a bipolar channel name."""
    bare = strip_subject_prefix(name, subject)
    if "-" not in bare:
        raise ValueError(f"Cannot identify bipolar contacts from {name!r}")
    left, right = bare.rsplit("-", 1)
    if right.isdigit():
        match = re.match(r"^(.*?)(\d+)$", left)
        if match is None:
            raise ValueError(f"Cannot identify bipolar contacts from {name!r}")
        return left, f"{match.group(1)}{right}"
    return left, right


def load_contacts(path: Path, subject: str) -> dict[str, np.ndarray]:
    table = pd.read_csv(path, sep="\t")
    required = {"name", "x", "y", "z"}
    if not required.issubset(table.columns):
        raise ValueError(f"{path} lacks columns {sorted(required - set(table.columns))}")
    table = table.dropna(subset=["x", "y", "z"])
    scale = coordinate_scale_to_mm(table[["x", "y", "z"]].to_numpy(float))
    names = table["name"].map(lambda value: strip_subject_prefix(value, subject))
    if names.duplicated().any():
        duplicates = sorted(names[names.duplicated(keep=False)].unique())
        raise ValueError(f"Duplicate physical contacts in {path}: {duplicates}")
    return {
        name: validate_mm(row[["x", "y", "z"]].to_numpy(float) * scale)
        for name, (_, row) in zip(names, table.iterrows())
    }


def coordinate_to_voxel(xyz_mm: np.ndarray, inv_tkr: np.ndarray) -> np.ndarray:
    return (inv_tkr @ np.r_[np.asarray(xyz_mm, float), 1.0])[:3]


def rounded_index(voxel: np.ndarray, shape: tuple[int, ...]) -> tuple[int, int, int]:
    index = np.rint(voxel).astype(int)
    if np.any(index < 0) or np.any(index >= np.asarray(shape[:3])):
        raise ValueError(f"Coordinate maps outside native grid: IJK={index.tolist()}")
    return tuple(int(value) for value in index)


def tissue_name(label_id: int, tissue_code: int) -> str:
    if label_id == CORPUS_CALLOSUM_ID:
        return "WM"
    if label_id in VENTRICLE_IDS:
        return "CSF"
    if tissue_code not in TISSUE_CODES:
        raise ValueError(f"Unexpected tc3crisp tissue code: {tissue_code}")
    if label_id == 0 and tissue_code == 2:
        return "Unclassified-GM"
    return TISSUE_CODES[tissue_code]


def point_label(
    xyz_mm: np.ndarray,
    inv_tkr: np.ndarray,
    segmentation: np.ndarray,
    tissue: np.ndarray,
    lut: dict[int, str],
) -> dict[str, object]:
    voxel = coordinate_to_voxel(xyz_mm, inv_tkr)
    index = rounded_index(voxel, segmentation.shape)
    label_id = int(segmentation[index])
    tissue_code = int(tissue[index])
    tissue_value = tissue_name(label_id, tissue_code)
    if label_id:
        name = lut[label_id]
    else:
        name = tissue_value
    valid = label_id != 0 and tissue_value == "GM"
    return {
        "xyz": np.asarray(xyz_mm, float),
        "voxel": voxel,
        "index": index,
        "id": label_id,
        "name": name,
        "tissue": tissue_value,
        "valid": bool(valid),
        "region6": REGION6.get(label_id, ""),
        "ap": AP.get(label_id, ""),
        "is_insula": bool(valid and label_id in INSULA_IDS),
    }


def ordered_unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def consensus(points: list[dict[str, object]]) -> dict[str, object]:
    valid_names = ordered_unique([
        str(point["name"]) for point in points if bool(point["valid"])
    ])
    valid_points = sum(bool(point["valid"]) for point in points)
    insula_points = sum(bool(point["is_insula"]) for point in points)

    if valid_names:
        roi = "–".join(valid_names)
        mix = len(valid_names) > 1
    else:
        tissues = [str(point["tissue"]) for point in points]
        # Match common/parcellation.py exactly: when no tissue ROI remains,
        # White Matter is more informative than all other non-tissue labels;
        # otherwise the general fallback is Unknown.  The point-level tissue
        # columns retain CSF/Outside detail for QC.
        roi = "WM" if "WM" in tissues else "Unknown"
        mix = False

    insula_region6 = ordered_unique([
        str(point["region6"]) for point in points if bool(point["is_insula"])
    ])
    ap_values = ordered_unique([
        str(point["ap"]) for point in points if bool(point["is_insula"])
    ])
    if set(ap_values) == {"Anterior", "Posterior"}:
        ap_consensus = "Anterior–Posterior"
    else:
        ap_consensus = "–".join(ap_values)
    return {
        "maper_roi": roi,
        "maper_mix": bool(mix),
        "maper_valid_points": int(valid_points),
        "maper_insula_points": int(insula_points),
        "maper_insula_status": (
            "core" if insula_points == 3
            else "partial" if insula_points > 0
            else "none"
        ),
        "maper_region6_consensus": "–".join(insula_region6),
        "maper_ap_consensus": ap_consensus,
        "maper_ap_mix": len(ap_values) > 1,
    }


def aparc_is_insula(roi: object) -> bool:
    return "INS" in [piece.strip() for piece in str(roi).split("–")]


def agreement_label(aparc_insula: bool, insula_points: int) -> str:
    maper_insula = insula_points > 0
    if maper_insula:
        return "concordant_insula" if aparc_insula else "maper_only"
    return "aparc_only" if aparc_insula else "concordant_noninsula"


def sphere_offsets(vox2ras_tkr: np.ndarray, radius_mm: float) -> list[np.ndarray]:
    linear = np.asarray(vox2ras_tkr, float)[:3, :3]
    minimum_step = min(np.linalg.norm(linear[:, axis]) for axis in range(3))
    extent = int(np.ceil(radius_mm / minimum_step))
    offsets: list[np.ndarray] = []
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                offset = np.array([i, j, k], dtype=int)
                if np.linalg.norm(linear @ offset) <= radius_mm + 1e-9:
                    offsets.append(offset)
    return offsets


def sphere_summary(
    point: dict[str, object],
    segmentation: np.ndarray,
    tissue: np.ndarray,
    offsets: list[np.ndarray],
) -> dict[str, object]:
    center = np.rint(np.asarray(point["voxel"], float)).astype(int)
    shape = np.asarray(segmentation.shape[:3])
    ids: list[int] = []
    total = 0
    for offset in offsets:
        index = center + offset
        if np.any(index < 0) or np.any(index >= shape):
            continue
        total += 1
        idx = tuple(int(value) for value in index)
        label_id = int(segmentation[idx])
        if (
            label_id in INSULA_IDS
            and tissue_name(label_id, int(tissue[idx])) == "GM"
        ):
            ids.append(label_id)

    if ids:
        counts = Counter(ids)
        winner_id, winner_count = sorted(
            counts.items(), key=lambda item: (-item[1], item[0]))[0]
        winner_region = REGION6[winner_id]
        winner_fraction = winner_count / len(ids)
    else:
        winner_region = ""
        winner_fraction = 0.0
    return {
        "sphere_total_voxels": int(total),
        "sphere_insula_voxels": int(len(ids)),
        "sphere_insula_fraction": float(len(ids) / total) if total else 0.0,
        "sphere_winner_region6": winner_region,
        "sphere_winner_fraction_within_insula": float(winner_fraction),
    }


def propagated_paths(directory: Path, subject: str) -> list[Path]:
    paths = sorted(
        directory.glob(f"*-{subject}/seg/seg95.nii.gz"),
        key=lambda path: int(path.parts[-3].split("-")[0].lstrip("a")),
    )
    if len(paths) != 30:
        raise ValueError(f"Expected 30 propagated segmentations, found {len(paths)}")
    return paths


def vote_summaries(
    paths: list[Path],
    indices: list[tuple[int, int, int]],
    shape: tuple[int, ...],
    lut: dict[int, str],
) -> list[dict[str, object]]:
    votes: list[list[int]] = [[] for _ in indices]
    for path in paths:
        image = nib.load(path)
        if image.shape[:3] != shape[:3]:
            raise ValueError(f"Propagated segmentation grid mismatch: {path}")
        data = np.asanyarray(image.dataobj).squeeze()
        for position, index in enumerate(indices):
            votes[position].append(int(data[index]))

    summaries: list[dict[str, object]] = []
    for values in votes:
        counts = Counter(values)
        winner_id, winner_count = sorted(
            counts.items(), key=lambda item: (-item[1], item[0]))[0]
        summaries.append({
            "insula_vote_fraction": sum(value in INSULA_IDS for value in values) / 30.0,
            "anterior_vote_fraction": sum(AP.get(value) == "Anterior" for value in values) / 30.0,
            "posterior_vote_fraction": sum(AP.get(value) == "Posterior" for value in values) / 30.0,
            "winner_id": int(winner_id),
            "winner_label": lut.get(winner_id, "Outside" if winner_id == 0 else str(winner_id)),
            "winner_vote_fraction": winner_count / 30.0,
        })
    return summaries


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    lut = load_lut(args.lut)
    aparc = pd.read_csv(args.parcellation_csv)
    required = {"subject", "name", "x", "y", "z", "roi"}
    if not required.issubset(aparc.columns):
        raise ValueError(
            f"{args.parcellation_csv} lacks columns {sorted(required - set(aparc.columns))}")
    if aparc["name"].duplicated().any():
        raise ValueError("Duplicate bipolar channel names in aparc table")

    # A rerun must start from a clean aparc table, not recursively append old
    # MAPER fields from a previous derivative.
    forbidden = [column for column in aparc if column.startswith("maper_")]
    if forbidden:
        raise ValueError(f"Input already contains MAPER columns: {forbidden}")

    contacts = load_contacts(args.contacts_tsv, args.subject)
    midpoint_scale = coordinate_scale_to_mm(
        aparc[["x", "y", "z"]].to_numpy(float))
    fused_image = nib.load(args.fused)
    tissue_image = nib.load(args.tissue)
    orig_image = nib.load(args.orig)
    segmentation = np.asanyarray(fused_image.dataobj).squeeze().astype(np.int16)
    tissue = np.asanyarray(tissue_image.dataobj).squeeze().astype(np.int8)
    if segmentation.shape != tissue.shape or segmentation.shape != orig_image.shape[:3]:
        raise ValueError(
            f"Native grid mismatch: fused={segmentation.shape}, tissue={tissue.shape}, "
            f"orig={orig_image.shape[:3]}")
    tissue_values = set(np.unique(tissue).tolist())
    if not tissue_values.issubset(TISSUE_CODES):
        raise ValueError(f"Unexpected tissue codes: {sorted(tissue_values)}")

    inv_tkr = np.linalg.inv(orig_image.header.get_vox2ras_tkr())
    offsets = sphere_offsets(orig_image.header.get_vox2ras_tkr(), args.sphere_radius)

    records: list[dict[str, object]] = []
    sensitivity: list[dict[str, object]] = []
    point_refs: list[dict[str, object]] = []
    vote_indices: list[tuple[int, int, int]] = []
    midpoint_errors: list[float] = []

    for row_index, row in aparc.iterrows():
        if {"contact_1", "contact_2"}.issubset(aparc.columns):
            contact_1 = strip_subject_prefix(row["contact_1"], args.subject)
            contact_2 = strip_subject_prefix(row["contact_2"], args.subject)
        else:
            contact_1, contact_2 = split_bipolar_name(row["name"], args.subject)
        if contact_1 not in contacts or contact_2 not in contacts:
            raise KeyError(f"Missing endpoints for {row['name']}: {contact_1}, {contact_2}")
        xyz_1 = contacts[contact_1]
        xyz_2 = contacts[contact_2]
        midpoint = validate_mm(
            row[["x", "y", "z"]].to_numpy(float) * midpoint_scale)
        midpoint_errors.append(float(np.max(np.abs((xyz_1 + xyz_2) / 2.0 - midpoint))))

        xyz_by_location = {
            "contact_1": xyz_1,
            "center": midpoint,
            "contact_2": xyz_2,
        }
        points = [
            point_label(xyz_by_location[location], inv_tkr, segmentation, tissue, lut)
            for location in LOCATIONS
        ]
        update: dict[str, object] = {
            "task": args.task,
            "reference": "bipolar",
        }
        for location, point in zip(LOCATIONS, points):
            prefix = f"maper_{location}"
            update.update({
                f"{prefix}_id": point["id"],
                f"{prefix}_name": point["name"],
                f"{prefix}_tissue": point["tissue"],
                f"{prefix}_region6": point["region6"],
                f"{prefix}_ap": point["ap"],
            })
            point_refs.append({
                "row_index": row_index,
                "location": location,
                "prefix": prefix,
            })
            vote_indices.append(point["index"])
            sphere = sphere_summary(point, segmentation, tissue, offsets)
            sensitivity.append({
                "task": args.task,
                "subject": row["subject"],
                "reference": "bipolar",
                "name": row["name"],
                "location": location,
                "x": point["xyz"][0],
                "y": point["xyz"][1],
                "z": point["xyz"][2],
                **sphere,
            })

        update.update(consensus(points))
        aparc_insula = aparc_is_insula(row["roi"])
        update["aparc_is_insula"] = aparc_insula
        update["maper_atlas_agreement"] = agreement_label(
            aparc_insula, int(update["maper_insula_points"]))
        records.append(update)

    max_midpoint_error = max(midpoint_errors, default=0.0)
    if max_midpoint_error > 1e-9:
        raise ValueError(
            f"Stored bipolar midpoint differs from endpoint mean by {max_midpoint_error} mm")

    paths = propagated_paths(args.propagated_dir, args.subject)
    votes = vote_summaries(paths, vote_indices, segmentation.shape, lut)
    for reference, vote in zip(point_refs, votes):
        row_index = int(reference["row_index"])
        prefix = str(reference["prefix"])
        records[row_index].update({
            f"{prefix}_{key}": value for key, value in vote.items()
        })

    result = pd.concat([aparc.reset_index(drop=True), pd.DataFrame(records)], axis=1)
    sensitivity_table = pd.DataFrame(sensitivity)
    if not np.isfinite(
        sensitivity_table[[
            "sphere_insula_fraction", "sphere_winner_fraction_within_insula"
        ]].to_numpy(float)
    ).all():
        raise ValueError("Non-finite sphere fractions produced")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.sensitivity_output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    sensitivity_table.to_csv(args.sensitivity_output, index=False)
    print(
        f"WROTE {args.output} rows={len(result)}; "
        f"{args.sensitivity_output} rows={len(sensitivity_table)}; "
        f"max_midpoint_error_mm={max_midpoint_error:.3g}")
    return result, sensitivity_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--tissue", type=Path, required=True)
    parser.add_argument("--orig", type=Path, required=True)
    parser.add_argument("--propagated-dir", type=Path, required=True)
    parser.add_argument("--parcellation-csv", type=Path, required=True)
    parser.add_argument("--contacts-tsv", type=Path, required=True)
    parser.add_argument("--lut", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sensitivity-output", type=Path, required=True)
    parser.add_argument("--sphere-radius", type=float, default=2.0)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
