#!/usr/bin/env python3
"""Faillenot/Hammersmith insula native-space pilot.

This project-specific stage leaves the general Destrieux parcellation intact.
It registers a subject's FreeSurfer native T1 to the SPM canonical MNI space,
pulls the Faillenot probability maps back to native space, and appends
probabilistic six-region and anterior/posterior assignments to a copy of the
existing parcellation table.

Coordinate spaces are deliberately explicit:

* input ``x/y/z``: subject-native FreeSurfer tkRAS, millimetres;
* registration fixed image: SPM canonical ``avg152T1.nii``;
* Faillenot maps: MNI152 grid, 1.5-mm voxels, values 0..30 subjects;
* native maps: the subject ``orig.mgz`` voxel grid, values converted to 0..1.

The downloaded file name ``anterior_pole`` corresponds to the paper's
"anterior inferior cortex" (AIC); it is one component of broad anterior
insula, not a synonym for the whole anterior insula.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy.ndimage import binary_closing, binary_fill_holes, label as connected_components
from nibabel.processing import resample_from_to


ANTS_DIR = Path("/opt/apps/rhel8/ANTs-2.4.2/bin")
ANTS_REGISTRATION = ANTS_DIR / "antsRegistrationSyNQuick.sh"
ANTS_APPLY = ANTS_DIR / "antsApplyTransforms"
ANTS_APPLY_POINTS = ANTS_DIR / "antsApplyTransformsToPoints"
ANTS_JACOBIAN = ANTS_DIR / "CreateJacobianDeterminantImage"

DEFAULT_ATLAS_ROOT = Path("/cwork/ns458/atlases/Hammersmith_n30r95")
DEFAULT_BIDS_ROOT = Path("/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS")
DEFAULT_RECON_ROOT = Path("/cwork/ns458/ECoG_Recon")

REGIONS = (
    ("asg", "anterior_short_gyrus", "Anterior"),
    ("msg", "middle_short_gyrus", "Anterior"),
    ("psg", "posterior_short_gyrus", "Anterior"),
    ("aic", "anterior_pole", "Anterior"),
    ("alg", "anterior_long_gyrus", "Posterior"),
    ("plg", "posterior_long_gyrus", "Posterior"),
)
REGION_KEYS = tuple(item[0] for item in REGIONS)
ANTERIOR_KEYS = ("asg", "msg", "psg", "aic")
POSTERIOR_KEYS = ("alg", "plg")
POINTS = ("contact_1", "center", "contact_2")


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True, env=env)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        root = destination.resolve()
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Unsafe archive member: {member.name}")
        tar.extractall(destination)


def prepare_atlas(atlas_root: Path) -> dict[str, Path]:
    """Create a reproducible working copy while preserving raw archives."""
    raw = atlas_root / "raw"
    work = atlas_root / "derivatives" / "faillenot_spm_mni152"
    maps = work / "probability_maps"
    metadata = work / "metadata"
    maps.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)

    archives = {
        "probmaps": raw / "Hammers-newInsula_regions-probmaps-gm+full.tar",
        "maxprob": raw / "Hammers-n30r95-maxprob-MNI152.tar",
        "metadata": raw / "Hammers-metadata-n30r95.tar.gz",
    }
    for path in archives.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    if not list(maps.glob("probmap-*.nii.gz")):
        safe_extract(archives["probmaps"], maps)
    if not list(maps.glob("Hammers-n30r95-*.nii.gz")):
        safe_extract(archives["maxprob"], maps)
    if not list(metadata.iterdir()):
        safe_extract(archives["metadata"], metadata)

    expected = []
    for mask in ("full", "gm"):
        for _, file_region, _ in REGIONS:
            for hemi in ("L", "R"):
                expected.append(maps / f"probmap-{mask}-insula_{file_region}_{hemi}.nii.gz")
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing probability maps:\n" + "\n".join(missing))

    geometry = None
    ranges = {}
    for path in expected:
        image = nib.load(path)
        data = np.asarray(image.dataobj)
        current = (image.shape[:3], image.affine)
        if geometry is None:
            geometry = current
        elif current[0] != geometry[0] or not np.allclose(current[1], geometry[1]):
            raise ValueError(f"Probability-map geometry mismatch: {path}")
        minimum, maximum = float(np.nanmin(data)), float(np.nanmax(data))
        if minimum < 0 or maximum > 30:
            raise ValueError(f"Unexpected probability count range in {path}: {minimum}, {maximum}")
        ranges[path.name] = [minimum, maximum]

    manifest = {
        "space": "Faillenot_SPM_MNI152",
        "probability_denominator": 30,
        "shape": list(geometry[0]),
        "affine": np.asarray(geometry[1]).tolist(),
        "archives": {name: {"path": str(path), "sha256": sha256(path)}
                     for name, path in archives.items()},
        "ranges": ranges,
        "region_mapping": {
            key: {"file_name": file_region, "ap": ap}
            for key, file_region, ap in REGIONS
        },
        "terminology_note": (
            "anterior_pole in the distributed files corresponds to the paper's "
            "anterior inferior cortex; broad anterior insula is ASG+MSG+PSG+AIC"
        ),
        "source": "https://soundray.org/hammers-n30r95/",
    }
    (work / "atlas_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"work": work, "maps": maps, "metadata": metadata}


def build_registration_target(atlas_root: Path, maps: Path) -> Path:
    """Mask SPM avg152T1 with the exact-space Hammers whole-brain support.

    Registering a skull-stripped FreeSurfer brainmask to the unmasked SPM
    canonical image can yield a superficially plausible whole-brain overlay
    while displacing the deep insula.  The MPM supplies the matching brain
    support in precisely the Faillenot coordinate frame.
    """
    reference_dir = atlas_root / "reference"
    source = reference_dir / "avg152T1.nii"
    destination = reference_dir / "avg152T1_Hammers_brain.nii.gz"
    maxprob = maps / "Hammers-n30r95-maxprob-full-MNI152.nii.gz"
    if not source.is_file() or not maxprob.is_file():
        raise FileNotFoundError(f"Missing registration-target input: {source} or {maxprob}")
    if destination.is_file() and destination.stat().st_mtime >= max(
        source.stat().st_mtime, maxprob.stat().st_mtime
    ):
        return destination

    template = nib.load(source)
    maxprob_image = nib.load(maxprob)
    maxprob_3d = nib.Nifti1Image(np.asarray(maxprob_image.dataobj).squeeze(), maxprob_image.affine)
    resampled = resample_from_to(maxprob_3d, (template.shape[:3], template.affine), order=0)
    mask = np.asarray(resampled.dataobj) > 0
    mask = binary_fill_holes(binary_closing(mask, iterations=1))
    components, count = connected_components(mask)
    if count:
        sizes = np.bincount(components.ravel())
        mask = components == (np.argmax(sizes[1:]) + 1)
    data = np.asarray(template.dataobj, np.float32) * mask.astype(np.float32)
    nib.save(nib.Nifti1Image(data, template.affine), destination)
    return destination


def recon_subject(subject: str) -> str:
    return f"D{int(subject[1:])}"


def convert_mgz_to_nifti(source: Path, destination: Path) -> nib.spatialimages.SpatialImage:
    image = nib.load(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(image.dataobj, np.float32), image.affine), destination)
    return image


def registration_paths(bids_root: Path, subject: str) -> dict[str, Path]:
    directory = bids_root / "derivatives" / "faillenot_transforms" / f"sub-{subject}"
    prefix = directory / f"sub-{subject}_from-native_to-Faillenot_SPM_MNI152_"
    return {
        "directory": directory,
        "prefix": prefix,
        "moving": directory / f"sub-{subject}_space-native_brainmask.nii.gz",
        "affine": Path(f"{prefix}0GenericAffine.mat"),
        "warp": Path(f"{prefix}1Warp.nii.gz"),
        "inverse_warp": Path(f"{prefix}1InverseWarp.nii.gz"),
        "warped": Path(f"{prefix}Warped.nii.gz"),
        "inverse_warped": Path(f"{prefix}InverseWarped.nii.gz"),
        "jacobian": directory / f"sub-{subject}_space-Faillenot_SPM_MNI152_jacobian.nii.gz",
    }


def register_subject(subject: str, bids_root: Path, recon_root: Path,
                     template: Path, threads: int) -> dict[str, Path]:
    if not template.is_file():
        raise FileNotFoundError(f"SPM canonical template missing: {template}")
    paths = registration_paths(bids_root, subject)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    brainmask = recon_root / recon_subject(subject) / "mri" / "brainmask.mgz"
    if not brainmask.is_file():
        raise FileNotFoundError(brainmask)
    if not paths["moving"].is_file() or paths["moving"].stat().st_mtime < brainmask.stat().st_mtime:
        convert_mgz_to_nifti(brainmask, paths["moving"])

    required = ("affine", "warp", "inverse_warp", "warped", "inverse_warped")
    newest_input = max(template.stat().st_mtime, paths["moving"].stat().st_mtime)
    current = all(paths[key].is_file() and paths[key].stat().st_mtime >= newest_input
                  for key in required)
    if not current:
        env = os.environ.copy()
        env["ANTSPATH"] = f"{ANTS_DIR}/"
        env["PATH"] = f"{ANTS_DIR}:{env.get('PATH', '')}"
        run([
            str(ANTS_REGISTRATION), "-d", "3", "-f", str(template),
            "-m", str(paths["moving"]), "-o", str(paths["prefix"]),
            "-t", "s", "-n", str(threads), "-p", "f",
        ], env=env)
    if not all(paths[key].is_file() for key in required):
        raise RuntimeError("ANTs registration outputs are incomplete")

    if ANTS_JACOBIAN.is_file() and not paths["jacobian"].is_file():
        run([str(ANTS_JACOBIAN), "3", str(paths["warp"]), str(paths["jacobian"]), "0", "1"])
    return paths


def map_path(maps: Path, mask: str, region: str, hemi: str) -> Path:
    file_region = dict((key, filename) for key, filename, _ in REGIONS)[region]
    return maps / f"probmap-{mask}-insula_{file_region}_{hemi}.nii.gz"


def warp_maps_to_native(subject: str, bids_root: Path, recon_root: Path,
                        maps: Path, transforms: dict[str, Path]) -> Path:
    output = bids_root / "derivatives" / "faillenot" / f"sub-{subject}" / "atlas"
    output.mkdir(parents=True, exist_ok=True)
    orig = recon_root / recon_subject(subject) / "mri" / "orig.mgz"
    if not orig.is_file():
        raise FileNotFoundError(orig)

    for mask in ("full", "gm"):
        for region in REGION_KEYS:
            for hemi in ("L", "R"):
                source = map_path(maps, mask, region, hemi)
                target = output / f"sub-{subject}_space-native_desc-{mask}_faillenot-{region}_{hemi}.nii.gz"
                if target.is_file() and target.stat().st_mtime >= max(
                    source.stat().st_mtime, transforms["affine"].stat().st_mtime,
                    transforms["inverse_warp"].stat().st_mtime,
                ):
                    continue
                run([
                    str(ANTS_APPLY), "-d", "3", "-i", str(source), "-r", str(orig),
                    "-o", str(target), "-n", "Linear",
                    "-t", f"[{transforms['affine']},1]",
                    "-t", str(transforms["inverse_warp"]),
                ])

    for kind in ("full", "gm"):
        source = maps / f"Hammers-n30r95-maxprob-{kind}-MNI152.nii.gz"
        target = output / f"sub-{subject}_space-native_desc-{kind}_Hammers-maxprob.nii.gz"
        if not target.is_file() or target.stat().st_mtime < max(
            source.stat().st_mtime, transforms["affine"].stat().st_mtime,
            transforms["inverse_warp"].stat().st_mtime,
        ):
            run([
                str(ANTS_APPLY), "-d", "3", "-i", str(source), "-r", str(orig),
                "-o", str(target), "-n", "NearestNeighbor",
                "-t", f"[{transforms['affine']},1]",
                "-t", str(transforms["inverse_warp"]),
            ])
    return output


def tkr_to_voxel(point_mm: np.ndarray, image) -> np.ndarray:
    return (np.linalg.inv(image.header.get_vox2ras_tkr()) @ np.r_[point_mm, 1.0])[:3]


def sample_point(data: np.ndarray, ijk: np.ndarray) -> float:
    return float(map_coordinates(data, np.asarray(ijk)[:, None], order=1,
                                 mode="constant", cval=0.0, prefilter=False)[0])


def sphere_offsets(radius_mm: float, affine: np.ndarray) -> np.ndarray:
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    extent = np.ceil(radius_mm / spacing).astype(int)
    offsets = []
    for i in range(-extent[0], extent[0] + 1):
        for j in range(-extent[1], extent[1] + 1):
            for k in range(-extent[2], extent[2] + 1):
                offset = np.array([i, j, k], float)
                displacement = affine[:3, :3] @ offset
                if np.linalg.norm(displacement) <= radius_mm + 1e-9:
                    offsets.append(offset)
    return np.asarray(offsets)


def sample_sphere(data: np.ndarray, ijk: np.ndarray, offsets: np.ndarray) -> float:
    coordinates = (ijk[None, :] + offsets).T
    values = map_coordinates(data, coordinates, order=1, mode="constant",
                            cval=0.0, prefilter=False)
    return float(np.mean(values))


def probability_summary(probabilities: dict[str, float]) -> dict[str, object]:
    values = {key: float(np.clip(probabilities[key], 0.0, 1.0)) for key in REGION_KEYS}
    top = max(values, key=values.get)
    top_probability = values[top]
    anterior = sum(values[key] for key in ANTERIOR_KEYS)
    posterior = sum(values[key] for key in POSTERIOR_KEYS)
    total = anterior + posterior
    if total <= 0:
        ap = "Unclassified"
        fraction = np.nan
        margin = 0.0
        top_label = "Unclassified"
    else:
        ap = "Anterior" if anterior >= posterior else "Posterior"
        fraction = anterior / total
        margin = abs(anterior - posterior) / total
        top_label = top.upper()
    return {
        **values,
        "label": top_label,
        "label_probability": top_probability,
        "p_anterior": anterior,
        "p_posterior": posterior,
        "ap": ap,
        "ap_anterior_fraction": fraction,
        "ap_margin": margin,
    }


def split_bipolar(name: str, subject: str) -> tuple[str, str]:
    bare = name[len(subject) + 1:] if name.startswith(subject + "_") else name
    left, right = bare.rsplit("-", 1)
    if right.isdigit():
        match = re.match(r"^(.*?)(\d+)$", left)
        if not match:
            raise ValueError(name)
        return left, f"{match.group(1)}{right}"
    return left, right


def load_points(subject: str, bids_root: Path, table: pd.DataFrame,
                reference: str) -> dict[str, dict[str, np.ndarray | None]]:
    if reference == "car":
        return {
            row["name"]: {"contact_1": row[["x", "y", "z"]].to_numpy(float),
                          "center": row[["x", "y", "z"]].to_numpy(float),
                          "contact_2": None}
            for _, row in table.iterrows()
        }
    car_path = (bids_root / "derivatives" / "parcellation" / f"sub-{subject}" /
                "car" / f"sub-{subject}_aparc2009s.csv")
    car = pd.read_csv(car_path)
    lookup = {}
    for _, row in car.iterrows():
        bare = row["name"][len(subject) + 1:] if row["name"].startswith(subject + "_") else row["name"]
        lookup[bare] = row[["x", "y", "z"]].to_numpy(float)
    result = {}
    for _, row in table.iterrows():
        first, second = split_bipolar(str(row["name"]), subject)
        result[row["name"]] = {
            "contact_1": lookup[first],
            "center": row[["x", "y", "z"]].to_numpy(float),
            "contact_2": lookup[second],
        }
    return result


def native_map_arrays(atlas_dir: Path, subject: str, mask: str, hemi: str):
    arrays = {}
    reference = None
    for region in REGION_KEYS:
        path = atlas_dir / f"sub-{subject}_space-native_desc-{mask}_faillenot-{region}_{hemi}.nii.gz"
        image = nib.load(path)
        if reference is None:
            reference = image
        elif image.shape != reference.shape or not np.allclose(image.affine, reference.affine):
            raise ValueError(f"Native probability-map geometry mismatch: {path}")
        arrays[region] = np.asarray(image.dataobj, dtype=float).squeeze() / 30.0
    return reference, arrays


def add_summary_columns(row: dict, prefix: str, summary: dict[str, object]) -> None:
    for region in REGION_KEYS:
        row[f"{prefix}_p_{region}"] = summary[region]
    for key in ("label", "label_probability", "p_anterior", "p_posterior",
                "ap", "ap_anterior_fraction", "ap_margin"):
        row[f"{prefix}_{key}"] = summary[key]


def consensus_ap(labels: list[str]) -> tuple[str, bool]:
    valid = []
    for label in labels:
        if label in ("Anterior", "Posterior") and label not in valid:
            valid.append(label)
    if not valid:
        return "Unclassified", False
    if len(valid) == 1:
        return valid[0], False
    return "Anterior–Posterior", True


def label_electrodes(subject: str, reference: str, bids_root: Path,
                     recon_root: Path, atlas_dir: Path) -> Path:
    source = (bids_root / "derivatives" / "parcellation" / f"sub-{subject}" /
              reference / f"sub-{subject}_aparc2009s.csv")
    table = pd.read_csv(source)
    original_columns = list(table.columns)
    points = load_points(subject, bids_root, table, reference)
    orig = nib.load(recon_root / recon_subject(subject) / "mri" / "orig.mgz")

    loaded = {}
    for mask in ("full", "gm"):
        for hemi in ("L", "R"):
            loaded[(mask, hemi)] = native_map_arrays(atlas_dir, subject, mask, hemi)
    sphere = sphere_offsets(2.0, orig.affine)

    output_rows = []
    for _, source_row in table.iterrows():
        result = source_row.to_dict()
        hemi = str(source_row.get("hemi", ""))
        if hemi not in ("L", "R"):
            hemi = "L" if float(source_row["x"]) < 0 else "R"
        ap_labels = []
        sphere_labels = []
        gm_labels = []
        for point_name in POINTS:
            point = points[source_row["name"]][point_name]
            output_prefix = f"faillenot_{point_name}"
            if point is None:
                for suffix in ("", "_sphere2mm", "_gm"):
                    add_summary_columns(result, output_prefix + suffix,
                                        probability_summary({key: 0.0 for key in REGION_KEYS}))
                continue
            ijk = tkr_to_voxel(np.asarray(point, float), orig)
            _, full_arrays = loaded[("full", hemi)]
            _, gm_arrays = loaded[("gm", hemi)]
            point_summary = probability_summary({
                key: sample_point(full_arrays[key], ijk) for key in REGION_KEYS
            })
            sphere_summary = probability_summary({
                key: sample_sphere(full_arrays[key], ijk, sphere) for key in REGION_KEYS
            })
            gm_summary = probability_summary({
                key: sample_point(gm_arrays[key], ijk) for key in REGION_KEYS
            })
            add_summary_columns(result, output_prefix, point_summary)
            add_summary_columns(result, output_prefix + "_sphere2mm", sphere_summary)
            add_summary_columns(result, output_prefix + "_gm", gm_summary)
            ap_labels.append(str(point_summary["ap"]))
            sphere_labels.append(str(sphere_summary["ap"]))
            gm_labels.append(str(gm_summary["ap"]))
        result["insula_ap"], result["insula_ap_mix"] = consensus_ap(ap_labels)
        result["insula_ap_sphere2mm"], result["insula_ap_sphere2mm_mix"] = consensus_ap(sphere_labels)
        result["insula_ap_gm"], result["insula_ap_gm_mix"] = consensus_ap(gm_labels)
        output_rows.append(result)

    output = pd.DataFrame(output_rows)
    if list(output.columns[:len(original_columns)]) != original_columns or len(output) != len(table):
        raise AssertionError("Original parcellation columns or row count changed")
    probability_columns = [column for column in output if "_p_" in column or column.endswith("_probability")]
    numeric = output[probability_columns].to_numpy(float)
    if not np.isfinite(numeric).all() or np.any(numeric < -1e-8) or np.any(numeric > 1 + 1e-8):
        raise ValueError("Output probabilities are non-finite or outside 0..1")

    destination = (bids_root / "derivatives" / "faillenot" / f"sub-{subject}" /
                   reference / f"sub-{subject}_desc-faillenot_insula.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="D0044")
    parser.add_argument("--reference", choices=("bipolar", "car", "both"), default="bipolar")
    parser.add_argument("--atlas-root", type=Path, default=DEFAULT_ATLAS_ROOT)
    parser.add_argument("--bids-root", type=Path, default=DEFAULT_BIDS_ROOT)
    parser.add_argument("--recon-root", type=Path, default=DEFAULT_RECON_ROOT)
    parser.add_argument("--template", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    atlas = prepare_atlas(args.atlas_root)
    if args.template is None:
        args.template = build_registration_target(args.atlas_root, atlas["maps"])
    transforms = register_subject(args.subject, args.bids_root, args.recon_root,
                                  args.template, args.threads)
    native_atlas = warp_maps_to_native(args.subject, args.bids_root, args.recon_root,
                                       atlas["maps"], transforms)
    references = ("bipolar", "car") if args.reference == "both" else (args.reference,)
    for reference in references:
        destination = label_electrodes(args.subject, reference, args.bids_root,
                                       args.recon_root, native_atlas)
        print(f"Saved {destination}")


if __name__ == "__main__":
    main()
