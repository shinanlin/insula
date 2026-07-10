#!/usr/bin/env python3
"""Export insula sub-mesh and per-vertex insula mask for the HGA Explorer viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np
from scipy.spatial import cKDTree

from export_average_brain_mesh import (
    decimate_mesh,
    export_glb,
    merge_hemispheres,
    read_hemisphere_mesh,
    write_meta,
)
from insula_constants import INSULA_PATTERNS, is_insula_label

VIEWER_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = VIEWER_ROOT / "public" / "assets"
DEFAULT_FULL_META = ASSETS_DIR / "cvs_avg35_pial.meta.json"
DEFAULT_INSULA_GLB = ASSETS_DIR / "cvs_avg35_insula_pial.glb"
DEFAULT_MASK_JSON = ASSETS_DIR / "cvs_avg35_pial_insula_mask.json"
DEFAULT_INSULA_META = ASSETS_DIR / "cvs_avg35_insula.meta.json"


def read_insula_labels(recon_dir: Path, subject: str):
    return mne.read_labels_from_annot(
        subject=subject,
        parc="aparc.a2009s",
        surf_name="pial",
        hemi="both",
        subjects_dir=str(recon_dir),
    )


def insula_vertex_set(labels, hemi: str) -> set[int]:
    vertices: set[int] = set()
    for lab in labels:
        if lab.hemi != hemi:
            continue
        if is_insula_label(lab.name):
            vertices.update(int(v) for v in lab.vertices)
    return vertices


def get_insula_center(labels, hemi: str, pial_coords: np.ndarray) -> list[float] | None:
    vertices = sorted(insula_vertex_set(labels, hemi))
    if not vertices:
        return None
    center = pial_coords[vertices].mean(axis=0)
    return center.tolist()


def extract_insula_faces(faces: np.ndarray, insula_vertices: set[int]) -> np.ndarray:
    if not len(faces):
        return faces
    mask = np.fromiter(
        (
            int(faces[i, 0] in insula_vertices
                and faces[i, 1] in insula_vertices
                and faces[i, 2] in insula_vertices)
            for i in range(len(faces))
        ),
        dtype=bool,
        count=len(faces),
    )
    return faces[mask]


def build_decimated_insula_mask(
    decimated_coords: np.ndarray,
    high_res_coords: np.ndarray,
    insula_vertices: set[int],
) -> list[bool]:
    tree = cKDTree(high_res_coords)
    _, nearest = tree.query(decimated_coords, k=1)
    return [int(idx) in insula_vertices for idx in nearest]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recon_dir",
        type=Path,
        default=Path("/cwork/ns458/ECoG_Recon"),
        help="FreeSurfer subjects directory containing cvs_avg35_inMNI152",
    )
    parser.add_argument(
        "--subject",
        default="cvs_avg35_inMNI152",
        help="Average brain subject id",
    )
    parser.add_argument(
        "--full_target_faces",
        type=int,
        default=100_000,
        help="Target face count for full-brain decimation (must match cvs_avg35_pial.glb)",
    )
    parser.add_argument(
        "--insula_target_faces",
        type=int,
        default=15_000,
        help="Target face count for insula sub-mesh decimation",
    )
    parser.add_argument(
        "--insula_glb",
        type=Path,
        default=DEFAULT_INSULA_GLB,
        help="Output insula highlight GLB path",
    )
    parser.add_argument(
        "--mask_json",
        type=Path,
        default=DEFAULT_MASK_JSON,
        help="Output per-vertex insula mask JSON for decimated full brain",
    )
    parser.add_argument(
        "--meta_json",
        type=Path,
        default=DEFAULT_INSULA_META,
        help="Output insula metadata JSON",
    )
    args = parser.parse_args()

    labels = read_insula_labels(args.recon_dir, args.subject)
    lh_coords, lh_tris = read_hemisphere_mesh(args.recon_dir, args.subject, "lh")
    rh_coords, rh_tris = read_hemisphere_mesh(args.recon_dir, args.subject, "rh")

    lh_insula = insula_vertex_set(labels, "lh")
    rh_insula = insula_vertex_set(labels, "rh")
    rh_insula_offset = {v + len(lh_coords) for v in rh_insula}
    full_insula_vertices = lh_insula | rh_insula_offset

    coords, faces = merge_hemispheres(lh_coords, lh_tris, rh_coords, rh_tris)
    high_res_coords = coords
    high_res_faces = faces

    decimated_coords, decimated_faces, decimated_face_count = decimate_mesh(
        coords,
        faces,
        args.full_target_faces,
    )

    insula_faces = extract_insula_faces(high_res_faces, full_insula_vertices)
    if len(insula_faces) == 0:
        raise SystemExit("No insula faces found — check aparc labels and INSULA_PATTERNS.")

    insula_coords = high_res_coords
    insula_coords, insula_faces, insula_face_count = decimate_mesh(
        insula_coords,
        insula_faces,
        args.insula_target_faces,
    )
    export_glb(insula_coords, insula_faces, args.insula_glb)

    mask = build_decimated_insula_mask(
        decimated_coords,
        high_res_coords,
        full_insula_vertices,
    )
    args.mask_json.parent.mkdir(parents=True, exist_ok=True)
    args.mask_json.write_text(
        json.dumps({"mask": mask, "n_vertices": len(mask)}, indent=2) + "\n",
        encoding="utf-8",
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
        "subject": args.subject,
        "parc": "aparc.a2009s",
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
            "insula_glb": str(args.insula_glb),
            "mask_json": str(args.mask_json),
            "full_meta": str(DEFAULT_FULL_META),
        },
    }
    args.meta_json.parent.mkdir(parents=True, exist_ok=True)
    args.meta_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    if DEFAULT_FULL_META.exists():
        full_meta = json.loads(DEFAULT_FULL_META.read_text(encoding="utf-8"))
        expected_vertices = int(full_meta.get("n_vertices", 0))
        if expected_vertices and expected_vertices != len(mask):
            raise SystemExit(
                f"Mask length {len(mask)} does not match cvs_avg35_pial.meta.json "
                f"n_vertices={expected_vertices}. Re-run export_average_brain_mesh.py first."
            )

    insula_mb = args.insula_glb.stat().st_size / 1e6
    print(f"Wrote {args.insula_glb} ({insula_mb:.2f} MB, {insula_face_count} faces)")
    print(f"Wrote {args.mask_json} ({len(mask)} vertices, {sum(mask)} insula)")
    print(f"Wrote {args.meta_json}")


if __name__ == "__main__":
    main()
