#!/usr/bin/env python3
"""Registration, coordinate, anatomical, and stability QC for Faillenot pilot."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("faillenot_pilot", HERE / "faillenot_pilot.py")
PILOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PILOT)


def apply_points(points_ras: np.ndarray, transform_args: list[str]) -> np.ndarray:
    points_lps = np.asarray(points_ras, float) * np.array([-1.0, -1.0, 1.0])
    frame = pd.DataFrame(points_lps, columns=["x", "y", "z"])
    frame["t"] = 0
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="faillenot_qc_", ignore_cleanup_errors=True) as tmp:
        source = Path(tmp) / "in.csv"
        target = Path(tmp) / "out.csv"
        frame.to_csv(source, index=False)
        command = [str(PILOT.ANTS_APPLY_POINTS), "-d", "3", "-i", str(source), "-o", str(target)]
        for transform in transform_args:
            command.extend(["-t", transform])
        subprocess.run(command, check=True)
        result_lps = pd.read_csv(target)[["x", "y", "z"]].to_numpy(float)
    return result_lps * np.array([-1.0, -1.0, 1.0])


def scanner_ras(points_tkr: np.ndarray, orig) -> np.ndarray:
    ijk = np.vstack([PILOT.tkr_to_voxel(point, orig) for point in points_tkr])
    return nib.affines.apply_affine(orig.affine, ijk)


def plot_registration(template, warped, destination: Path) -> None:
    fixed = np.asarray(template.dataobj, float).squeeze()
    moving = np.asarray(warped.dataobj, float).squeeze()
    if fixed.shape != moving.shape:
        raise ValueError(f"Registration QC geometry mismatch: {fixed.shape} vs {moving.shape}")
    mask = moving > np.percentile(moving[moving > 0], 10)
    center = np.rint(np.argwhere(mask).mean(0)).astype(int)
    planes = [(0, center[0], "sagittal"), (1, center[1], "coronal"), (2, center[2], "axial")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for column, (axis, index, title) in enumerate(planes):
        f = np.take(fixed, index, axis=axis).T
        m = np.take(moving, index, axis=axis).T
        axes[0, column].imshow(f, cmap="gray", origin="lower")
        axes[0, column].imshow(m, cmap="magma", origin="lower", alpha=0.38)
        axes[0, column].set_title(f"{title}: template + warped D44")
        axes[1, column].imshow(f, cmap="Blues", origin="lower", alpha=0.75)
        axes[1, column].imshow(m, cmap="Reds", origin="lower", alpha=0.42)
        axes[1, column].set_title("edge/orientation inspection")
        for row in range(2):
            axes[row, column].axis("off")
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def plot_native_insula(orig, atlas_dir: Path, subject: str, table: pd.DataFrame,
                       destination: Path) -> None:
    canonical_orig = nib.as_closest_canonical(orig)
    brain = np.asarray(canonical_orig.dataobj, float)
    total_native = np.zeros(orig.shape[:3], float)
    for hemi in ("L", "R"):
        _, arrays = PILOT.native_map_arrays(atlas_dir, subject, "full", hemi)
        total_native += sum(arrays.values())
    total_image = nib.Nifti1Image(total_native, orig.affine)
    total = np.asarray(nib.as_closest_canonical(total_image).dataobj, float)
    insula = table[table["roi"].astype(str).str.contains("INS", case=False, na=False)]
    native_ijk = np.vstack([
        PILOT.tkr_to_voxel(row[["x", "y", "z"]].to_numpy(float), orig)
        for _, row in insula.iterrows()
    ])
    scanner_points = nib.affines.apply_affine(orig.affine, native_ijk)
    points = nib.affines.apply_affine(np.linalg.inv(canonical_orig.affine), scanner_points)
    center = np.rint(np.median(points, axis=0)).astype(int)
    planes = [(0, center[0], (1, 2), "sagittal"),
              (1, center[1], (0, 2), "coronal"),
              (2, center[2], (0, 1), "axial")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, (axis, index, dims, title) in zip(axes, planes):
        ax.imshow(np.take(brain, index, axis=axis).T, cmap="gray", origin="lower")
        overlay = np.take(total, index, axis=axis).T
        ax.imshow(np.ma.masked_where(overlay <= 0.01, overlay), cmap="turbo",
                  origin="lower", alpha=0.55, vmin=0, vmax=1)
        # Array display is transposed: horizontal is first retained dim, vertical second.
        near_slice = np.abs(points[:, axis] - index) <= 1.0
        ax.scatter(points[near_slice, dims[0]], points[near_slice, dims[1]], s=18, c="white",
                   edgecolors="black", linewidths=0.4)
        ax.set_title(f"native {title}: points within 1 mm")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def plot_electrode_exact_slices(orig, aparc, atlas_dir: Path, subject: str,
                                table: pd.DataFrame, destination: Path,
                                insula_ids: list[int]) -> None:
    """One exact orthogonal triplet per native-INS bipolar midpoint."""
    destination.mkdir(parents=True, exist_ok=True)
    canonical_orig = nib.as_closest_canonical(orig)
    brain = np.asarray(canonical_orig.dataobj, float)
    total_native = np.zeros(orig.shape[:3], float)
    for hemi in ("L", "R"):
        _, arrays = PILOT.native_map_arrays(atlas_dir, subject, "full", hemi)
        total_native += sum(arrays.values())
    total = np.asarray(nib.as_closest_canonical(
        nib.Nifti1Image(total_native, orig.affine)).dataobj, float)
    destrieux_native = np.isin(np.asarray(aparc.dataobj).astype(int), insula_ids).astype(np.uint8)
    destrieux = np.asarray(nib.as_closest_canonical(
        nib.Nifti1Image(destrieux_native, aparc.affine)).dataobj, bool)

    insula = table[table["roi"].astype(str).str.contains("INS", case=False, na=False)]
    for _, row in insula.iterrows():
        native_ijk = PILOT.tkr_to_voxel(row[["x", "y", "z"]].to_numpy(float), orig)
        scanner = nib.affines.apply_affine(orig.affine, native_ijk)
        point = nib.affines.apply_affine(np.linalg.inv(canonical_orig.affine), scanner)
        center = np.rint(point).astype(int)
        planes = [(0, center[0], (1, 2), "sagittal"),
                  (1, center[1], (0, 2), "coronal"),
                  (2, center[2], (0, 1), "axial")]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        for ax, (axis, index, dims, title) in zip(axes, planes):
            background = np.take(brain, index, axis=axis).T
            probability = np.take(total, index, axis=axis).T
            native_mask = np.take(destrieux, index, axis=axis).T
            ax.imshow(background, cmap="gray", origin="lower")
            ax.imshow(np.ma.masked_where(probability <= 0.01, probability),
                      cmap="turbo", origin="lower", alpha=0.55, vmin=0, vmax=1)
            if native_mask.any():
                ax.contour(native_mask, levels=[0.5], colors="lime", linewidths=0.8,
                           origin="lower")
            ax.scatter(point[dims[0]], point[dims[1]], s=42, c="white",
                       edgecolors="black", linewidths=0.8)
            ax.set_title(f"{title}, exact midpoint slice")
            ax.axis("off")
        fig.suptitle(f"{row['name']} | Faillenot heatmap | Destrieux INS green")
        fig.tight_layout()
        safe_name = str(row["name"]).replace("/", "_")
        fig.savefig(destination / f"{safe_name}.png", dpi=180)
        plt.close(fig)


def direct_probabilities(template_points: np.ndarray, maps: Path,
                         hemis: list[str]) -> list[dict[str, float]]:
    images = {}
    arrays = {}
    for hemi in ("L", "R"):
        for region in PILOT.REGION_KEYS:
            path = PILOT.map_path(maps, "full", region, hemi)
            image = nib.load(path)
            images[(hemi, region)] = image
            arrays[(hemi, region)] = np.asarray(image.dataobj, float).squeeze() / 30.0
    output = []
    for point, hemi in zip(template_points, hemis):
        image = images[(hemi, PILOT.REGION_KEYS[0])]
        ijk = nib.affines.apply_affine(np.linalg.inv(image.affine), point)
        output.append({region: float(map_coordinates(
            arrays[(hemi, region)], ijk[:, None], order=1,
            mode="constant", cval=0.0, prefilter=False)[0])
                       for region in PILOT.REGION_KEYS})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="D0044")
    parser.add_argument("--reference", default="bipolar", choices=("bipolar", "car"))
    parser.add_argument("--atlas-root", type=Path, default=PILOT.DEFAULT_ATLAS_ROOT)
    parser.add_argument("--bids-root", type=Path, default=PILOT.DEFAULT_BIDS_ROOT)
    parser.add_argument("--recon-root", type=Path, default=PILOT.DEFAULT_RECON_ROOT)
    args = parser.parse_args()

    atlas = PILOT.prepare_atlas(args.atlas_root)
    transforms = PILOT.registration_paths(args.bids_root, args.subject)
    atlas_dir = args.bids_root / "derivatives" / "faillenot" / f"sub-{args.subject}" / "atlas"
    result_path = (args.bids_root / "derivatives" / "faillenot" / f"sub-{args.subject}" /
                   args.reference / f"sub-{args.subject}_desc-faillenot_insula.csv")
    table = pd.read_csv(result_path)
    qc_dir = args.bids_root / "derivatives" / "faillenot" / f"sub-{args.subject}" / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    template_path = PILOT.build_registration_target(args.atlas_root, atlas["maps"])
    template = nib.load(template_path)
    warped = nib.load(transforms["warped"])
    orig = nib.load(args.recon_root / PILOT.recon_subject(args.subject) / "mri" / "orig.mgz")
    plot_registration(template, warped, qc_dir / "registration_overlay.png")
    plot_native_insula(orig, atlas_dir, args.subject, table, qc_dir / "native_insula_electrodes.png")

    point_lookup = PILOT.load_points(args.subject, args.bids_root, table, args.reference)
    records = []
    for _, row in table.iterrows():
        hemi = str(row["hemi"])
        if hemi not in ("L", "R"):
            hemi = "L" if float(row["x"]) < 0 else "R"
        for point_name in PILOT.POINTS:
            point = point_lookup[row["name"]][point_name]
            if point is not None:
                records.append((row["name"], point_name, hemi, np.asarray(point, float)))
    native_tkr = np.vstack([record[3] for record in records])
    native_scanner = scanner_ras(native_tkr, orig)
    template_scanner = apply_points(native_scanner, [
        f"[{transforms['affine']},1]", str(transforms["inverse_warp"])
    ])
    roundtrip = apply_points(template_scanner, [str(transforms["warp"]), str(transforms["affine"])])
    errors = np.linalg.norm(roundtrip - native_scanner, axis=1)
    direct = direct_probabilities(template_scanner, atlas["maps"], [record[2] for record in records])

    comparison = []
    for record, probabilities, error in zip(records, direct, errors):
        name, point_name, hemi, _ = record
        row = table.loc[table["name"].eq(name)].iloc[0]
        native_values = {region: float(row[f"faillenot_{point_name}_p_{region}"])
                         for region in PILOT.REGION_KEYS}
        direct_summary = PILOT.probability_summary(probabilities)
        native_summary = PILOT.probability_summary(native_values)
        comparison.append({
            "name": name, "point": point_name, "hemi": hemi,
            "roundtrip_error_mm": error,
            "native_ap": native_summary["ap"], "direct_ap": direct_summary["ap"],
            "ap_match": native_summary["ap"] == direct_summary["ap"],
            "max_probability_difference": max(abs(native_values[k] - probabilities[k])
                                                  for k in PILOT.REGION_KEYS),
        })
    comparison_frame = pd.DataFrame(comparison)
    comparison_frame.to_csv(qc_dir / "coordinate_crosscheck.csv", index=False)

    insula = table[table["roi"].astype(str).str.contains("INS", case=False, na=False)].copy()
    def expected(label: str) -> str:
        if "G_insular_short" in label:
            return "Anterior"
        if "G_Ins_lg_and_S_cent_ins" in label:
            return "Posterior"
        if "circular_insula" in label:
            return "Boundary"
        return "Other"
    insula["destrieux_expected_ap"] = insula["center"].astype(str).map(expected)
    insula["destrieux_faillenot_match"] = [
        expected_value == observed if expected_value in ("Anterior", "Posterior") else np.nan
        for expected_value, observed in zip(
            insula["destrieux_expected_ap"], insula["faillenot_center_ap"])
    ]
    insula["point_vs_sphere_changed"] = insula["insula_ap"] != insula["insula_ap_sphere2mm"]
    insula["full_vs_gm_changed"] = insula["insula_ap"] != insula["insula_ap_gm"]
    insula.to_csv(qc_dir / "insula_anatomical_stability_qc.csv", index=False)
    evaluable_anatomy = insula["destrieux_expected_ap"].isin(["Anterior", "Posterior"])
    anatomical_match_rate = (
        float(insula.loc[evaluable_anatomy, "destrieux_faillenot_match"].astype(bool).mean())
        if evaluable_anatomy.any() else None
    )

    # Independent native-anatomy check. This catches a globally plausible but
    # deeply displaced registration, which point round trips cannot detect.
    aparc = nib.load(args.recon_root / PILOT.recon_subject(args.subject) /
                     "mri" / "aparc.a2009s+aseg.mgz")
    aparc_data = np.asarray(aparc.dataobj).astype(int)
    lut_path = (Path("/hpc/group/coganlab/nanlinshi/seeg-preprocessing-worktrees/") /
                "lexical_delay/common/FreeSurferColorLUT.txt")
    insula_ids = []
    for line in lut_path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].isdigit() and "insula" in fields[1].lower():
            insula_ids.append(int(fields[0]))
    plot_electrode_exact_slices(
        orig, aparc, atlas_dir, args.subject, table,
        qc_dir / "electrode_exact_slices", insula_ids)
    destrieux_mask = np.isin(aparc_data, insula_ids)
    total_probability = np.zeros(aparc_data.shape, float)
    for hemi in ("L", "R"):
        _, arrays = PILOT.native_map_arrays(atlas_dir, args.subject, "full", hemi)
        total_probability += sum(arrays.values())
    faillenot_mask = total_probability > 0.01
    intersection = int(np.count_nonzero(destrieux_mask & faillenot_mask))
    dice = 2 * intersection / (int(destrieux_mask.sum()) + int(faillenot_mask.sum()))
    mask_rows = []
    scanner_x = nib.affines.apply_affine(
        aparc.affine, np.column_stack([np.arange(aparc.shape[0]),
                                      np.zeros(aparc.shape[0]), np.zeros(aparc.shape[0])]))[:, 0]
    for hemi, selector in (("L", scanner_x[:, None, None] < 0),
                           ("R", scanner_x[:, None, None] > 0)):
        dmask = destrieux_mask & selector
        fmask = faillenot_mask & selector
        dcom = nib.affines.apply_affine(aparc.affine, np.argwhere(dmask).mean(axis=0))
        fcom = nib.affines.apply_affine(aparc.affine, np.argwhere(fmask).mean(axis=0))
        mask_rows.append({"hemi": hemi, "destrieux_com_x": dcom[0],
                          "destrieux_com_y": dcom[1], "destrieux_com_z": dcom[2],
                          "faillenot_com_x": fcom[0], "faillenot_com_y": fcom[1],
                          "faillenot_com_z": fcom[2], "com_distance_mm": np.linalg.norm(fcom-dcom)})
    pd.DataFrame(mask_rows).to_csv(qc_dir / "native_atlas_alignment.csv", index=False)

    jacobian_candidates = [transforms["jacobian"], Path(str(transforms["jacobian"]) + ".nii.gz")]
    jacobian_path = next((path for path in jacobian_candidates if path.is_file()), None)
    negative_jacobian = None
    if jacobian_path:
        jacobian = np.asarray(nib.load(jacobian_path).dataobj, float)
        negative_jacobian = int(np.count_nonzero(jacobian <= 0))

    maximum_com_distance = max(float(row["com_distance_mm"]) for row in mask_rows)
    review_reasons = []
    if maximum_com_distance > 5.0:
        review_reasons.append("native atlas center-of-mass distance exceeds 5 mm")
    if anatomical_match_rate is not None and anatomical_match_rate < 0.8:
        review_reasons.append("native Destrieux short/long agreement is below 80%")
    summary = {
        "subject": args.subject,
        "reference": args.reference,
        "rows": len(table),
        "insula_rows": len(insula),
        "roundtrip_max_mm": float(errors.max()),
        "roundtrip_median_mm": float(np.median(errors)),
        "native_vs_direct_ap_match_rate": float(comparison_frame["ap_match"].mean()),
        "native_vs_direct_max_probability_difference": float(comparison_frame["max_probability_difference"].max()),
        "negative_or_zero_jacobian_voxels": negative_jacobian,
        "native_destrieux_faillenot_dice_at_p01": float(dice),
        "native_atlas_com_distance_mm": {row["hemi"]: float(row["com_distance_mm"])
                                          for row in mask_rows},
        "destrieux_short_long_evaluable_channels": int(evaluable_anatomy.sum()),
        "destrieux_short_long_ap_match_rate": anatomical_match_rate,
        "point_vs_sphere_changed_insula": int(insula["point_vs_sphere_changed"].sum()),
        "full_vs_gm_changed_insula": int(insula["full_vs_gm_changed"].sum()),
        "insula_ap_counts": insula["insula_ap"].value_counts(dropna=False).to_dict(),
        "pilot_status": "REVIEW_REQUIRED" if review_reasons else "PASS",
        "review_reasons": review_reasons,
    }
    (qc_dir / "qc_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
