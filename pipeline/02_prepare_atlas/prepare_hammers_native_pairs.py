#!/usr/bin/env python3
"""Prepare geometry-corrected Hammers n30r95 MRI/label pairs.

ONE-TIME step, shared across all subjects. Run once per cluster/atlas
download; the corrected labels in derivatives/individual_native_pairs/
are then reused as MAPER ancillaries for every subsequent subject.

Bug fixed by this script (see pipeline/D44_MAPER_worklog.md section 4):
the distributed ``*_IFHW_r95_zy.nii.gz`` label volumes have the same voxel
arrays as their paired T1 images but a historical, displaced NIfTI header
affine (16-30mm pure-translation offset across all 30 atlases, sform_code 2
vs 1 on the T1). If used as-is, MAPER's registration/label-propagation will
apply the correct deformation field but read the label values from the
WRONG physical location, silently corrupting every fused segmentation
built from these ancillaries.

This utility never edits the licensed raw archives. It verifies array shape
and foreground alignment (voxel-space centroid delta < 5 voxels; a raise
here means something is actually wrong, not just headers), then writes
label copies with the paired T1's affine/header for future multi-atlas
propagation.

Usage:
    python prepare_hammers_native_pairs.py --atlas-root /cwork/<user>/atlases/Hammersmith_n30r95

Expects under --atlas-root:
    raw/Hammers-MRIs-n30.tar
    raw/Hammers-labelsets-n30r95.tar

Writes under --atlas-root/derivatives/individual_native_pairs/:
    extracted/                       (raw tar contents)
    geometry_corrected_labels/aNN_labels_r95_geometry-fixed.nii.gz  (x30)
    geometry_correction_manifest.json   (audit trail: per-atlas affine deltas)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tarfile

import nibabel as nib
import numpy as np


DEFAULT_ROOT = Path("/cwork/ns458/atlases/Hammersmith_n30r95")


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        root = destination.resolve()
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.name}")
        tar.extractall(destination)


def foreground_center(data: np.ndarray, label: bool) -> np.ndarray:
    if label:
        mask = data > 0
    else:
        positive = data[data > 0]
        threshold = max(1.0, float(np.percentile(positive, 5)))
        mask = data > threshold
    return np.argwhere(mask).mean(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    raw = args.atlas_root / "raw"
    output = args.atlas_root / "derivatives" / "individual_native_pairs"
    extracted = output / "extracted"
    corrected = output / "geometry_corrected_labels"
    safe_extract(raw / "Hammers-MRIs-n30.tar", extracted)
    safe_extract(raw / "Hammers-labelsets-n30r95.tar", extracted)
    corrected.mkdir(parents=True, exist_ok=True)

    records = []
    for index in range(1, 31):
        atlas_id = f"a{index:02d}"
        t1_path = extracted / f"{atlas_id}.nii.gz"
        label_path = extracted / f"{atlas_id}_IFHW_r95_zy.nii.gz"
        t1 = nib.load(t1_path)
        label = nib.load(label_path)
        if t1.shape[:3] != label.shape[:3]:
            raise ValueError(f"Shape mismatch for {atlas_id}: {t1.shape} vs {label.shape}")
        t1_data = np.asarray(t1.dataobj)
        label_data = np.asarray(label.dataobj).squeeze()
        if not np.allclose(label_data, np.rint(label_data)) or label_data.min() < 0 or label_data.max() > 95:
            raise ValueError(f"Invalid label values for {atlas_id}")
        center_delta_vox = foreground_center(label_data, True) - foreground_center(t1_data, False)
        if np.linalg.norm(center_delta_vox) > 5:
            raise ValueError(
                f"MRI/label arrays do not appear index-aligned for {atlas_id}: {center_delta_vox}")

        destination = corrected / f"{atlas_id}_labels_r95_geometry-fixed.nii.gz"
        header = nib.Nifti1Header.from_header(t1.header)
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(label_data.astype(np.uint8), t1.affine, header), destination)
        check = nib.load(destination)
        if check.shape[:3] != t1.shape[:3] or not np.allclose(check.affine, t1.affine):
            raise AssertionError(f"Corrected geometry validation failed for {atlas_id}")
        records.append({
            "atlas": atlas_id,
            "t1": str(t1_path),
            "source_label": str(label_path),
            "corrected_label": str(destination),
            "source_affine_translation": label.affine[:3, 3].tolist(),
            "t1_affine_translation": t1.affine[:3, 3].tolist(),
            "foreground_center_delta_vox": center_delta_vox.tolist(),
        })

    (output / "geometry_correction_manifest.json").write_text(
        json.dumps(records, indent=2) + "\n")
    print(f"Prepared {len(records)} geometry-corrected pairs in {output}")


if __name__ == "__main__":
    main()
