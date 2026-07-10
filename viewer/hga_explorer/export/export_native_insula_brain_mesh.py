#!/usr/bin/env python3
"""Export per-subject native insula sub-mesh, vertex mask, and camera meta."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_EXPORT_DIR = Path(__file__).resolve().parent
if str(_EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPORT_DIR))

from export_insula_brain_mesh import (
    build_decimated_insula_mask,
    extract_insula_faces,
    get_insula_center,
    insula_vertex_set,
    read_insula_labels,
)
from export_average_brain_mesh import (
    decimate_mesh,
    export_glb,
    merge_hemispheres,
    read_hemisphere_mesh,
)
from export_native_brain_mesh import recon_subject_id
from insula_constants import INSULA_PATTERNS

VIEWER_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = VIEWER_ROOT / "public" / "assets" / "native"
DEFAULT_RECON_ROOT = Path("/cwork/ns458/ECoG_Recon")

VALIDATION_SUBJECTS = ["D0094", "D0071", "D0084"]


def export_subject_native_insula(
    recon_root: Path,
    subject: str,
    output_dir: Path,
    full_target_faces: int = 80_000,
    insula_target_faces: int = 15_000,
) -> Path | None:
    recon_id = recon_subject_id(subject)
    pial_meta_path = output_dir / f"{subject}_pial.meta.json"
    if not pial_meta_path.exists():
        print(f"SKIP {subject}: missing {pial_meta_path} — run export_native_brain_mesh.py first")
        return None

    try:
        labels = read_insula_labels(recon_root, recon_id)
        lh_coords, lh_tris = read_hemisphere_mesh(recon_root, recon_id, "lh")
        rh_coords, rh_tris = read_hemisphere_mesh(recon_root, recon_id, "rh")
    except FileNotFoundError as exc:
        print(f"SKIP {subject}: {exc}")
        return None

    lh_insula = insula_vertex_set(labels, "lh")
    rh_insula = insula_vertex_set(labels, "rh")
    rh_insula_offset = {v + len(lh_coords) for v in rh_insula}
    full_insula_vertices = lh_insula | rh_insula_offset

    coords, faces = merge_hemispheres(lh_coords, lh_tris, rh_coords, rh_tris)
    high_res_coords = coords
    high_res_faces = faces

    decimated_coords, _decimated_faces, decimated_face_count = decimate_mesh(
        coords,
        faces,
        full_target_faces,
    )

    insula_faces = extract_insula_faces(high_res_faces, full_insula_vertices)
    if len(insula_faces) == 0:
        print(f"SKIP {subject}: no insula faces found")
        return None

    insula_coords, insula_faces, insula_face_count = decimate_mesh(
        high_res_coords,
        insula_faces,
        insula_target_faces,
    )

    insula_glb = output_dir / f"{subject}_insula_pial.glb"
    mask_json = output_dir / f"{subject}_pial_insula_mask.json"
    meta_json = output_dir / f"{subject}_insula.meta.json"

    export_glb(insula_coords, insula_faces, insula_glb)

    mask = build_decimated_insula_mask(
        decimated_coords,
        high_res_coords,
        full_insula_vertices,
    )
    mask_json.write_text(
        json.dumps({"mask": mask, "n_vertices": len(mask)}, indent=2) + "\n",
        encoding="utf-8",
    )

    pial_meta = json.loads(pial_meta_path.read_text(encoding="utf-8"))
    expected_vertices = int(pial_meta.get("n_vertices", 0))
    if expected_vertices and expected_vertices != len(mask):
        raise SystemExit(
            f"{subject}: mask length {len(mask)} does not match "
            f"{pial_meta_path.name} n_vertices={expected_vertices}. "
            "Re-run export_native_brain_mesh.py with the same target_faces."
        )

    lh_center = get_insula_center(labels, "lh", lh_coords)
    rh_center = get_insula_center(labels, "rh", rh_coords)
    both_target = None
    if lh_center and rh_center:
        both_target = [
            (lh_center[0] + rh_center[0]) / 2,
            (lh_center[1] + rh_center[1]) / 2,
            (lh_center[2] + rh_center[2]) / 2,
        ]

    meta = {
        "subject": subject,
        "recon_id": recon_id,
        "parc": "aparc.a2009s",
        "coordinate_space": "native_RAS_mm",
        "insula_patterns": INSULA_PATTERNS,
        "n_vertices_mask": len(mask),
        "n_vertices_insula_mesh": int(len(insula_coords)),
        "n_faces_insula_mesh": int(insula_face_count),
        "n_faces_full_decimated": int(decimated_face_count),
        "lh_center": lh_center,
        "rh_center": rh_center,
        "both_target": both_target,
        "camera_hint": {
            "target": both_target,
            "distance": 180,
            "azimuth_deg": 118,
            "elevation_deg": 90,
            "fov": 50,
        },
        "outputs": {
            "insula_glb": str(insula_glb),
            "mask_json": str(mask_json),
            "full_meta": str(pial_meta_path),
        },
    }
    meta_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    insula_mb = insula_glb.stat().st_size / 1e6
    print(
        f"Wrote {insula_glb} ({insula_mb:.2f} MB, {insula_face_count} faces); "
        f"mask {len(mask)} vertices ({sum(mask)} insula)"
    )
    return insula_glb


def load_index_subjects(output_dir: Path) -> list[str]:
    index_path = output_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing native index: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    subjects = index.get("subjects") or []
    if not subjects:
        raise ValueError(f"No subjects listed in {index_path}")
    return subjects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recon_root", type=Path, default=DEFAULT_RECON_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--from-index",
        action="store_true",
        help="Export all subjects listed in output_dir/index.json (full native pial cohort)",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="BIDS-style subject ids (default: validation cohort, or index when --from-index)",
    )
    parser.add_argument("--full_target_faces", type=int, default=80_000)
    parser.add_argument("--insula_target_faces", type=int, default=15_000)
    args = parser.parse_args()

    if args.from_index:
        subjects = load_index_subjects(args.output_dir)
    elif args.subjects:
        subjects = args.subjects
    else:
        subjects = VALIDATION_SUBJECTS

    args.output_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    skipped = []
    for subject in subjects:
        path = export_subject_native_insula(
            args.recon_root,
            subject,
            args.output_dir,
            args.full_target_faces,
            args.insula_target_faces,
        )
        if path is None:
            skipped.append(subject)
        else:
            exported.append(subject)

    index_path = args.output_dir / "index.json"
    index = {}
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    index["insula_pattern"] = "{subject}_insula_pial.glb"
    index["insula_mask_pattern"] = "{subject}_pial_insula_mask.json"
    index["insula_meta_pattern"] = "{subject}_insula.meta.json"
    index["insula_subjects"] = exported
    index["insula_skipped"] = skipped
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    print(f"Exported {len(exported)} native insula meshes; skipped {len(skipped)}")
    print(f"Updated {index_path}")


if __name__ == "__main__":
    main()
