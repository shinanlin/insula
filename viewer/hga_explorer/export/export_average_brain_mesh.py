#!/usr/bin/env python3
"""Export cvs_avg35 pial surfaces to a decimated GLB for the HGA Explorer viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np

VIEWER_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = VIEWER_ROOT / "public" / "assets" / "cvs_avg35_pial.glb"


def _require(module_name: str, pip_name: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency '{module_name}'. Install with: pip install {pip_name}"
        ) from exc


def read_hemisphere_mesh(recon_dir: Path, subject: str, hemi: str):
    surf_path = recon_dir / subject / "surf" / f"{hemi}.pial"
    if not surf_path.exists():
        raise FileNotFoundError(f"Pial surface not found: {surf_path}")
    coords, tris = mne.read_surface(str(surf_path))
    return np.asarray(coords, dtype=np.float64), np.asarray(tris, dtype=np.int64)


def merge_hemispheres(lh_coords, lh_tris, rh_coords, rh_tris):
    offset = len(lh_coords)
    coords = np.vstack([lh_coords, rh_coords])
    faces = np.vstack([lh_tris, rh_tris + offset])
    return coords, faces


def decimate_mesh(coords, faces, target_faces: int):
    pv = _require("pyvista", "pyvista")
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64).ravel()
    mesh = pv.PolyData(coords, faces_pv)
    n_faces = int(mesh.n_cells)
    if n_faces <= target_faces:
        return coords, faces, n_faces

    reduction = 1.0 - (target_faces / n_faces)
    reduction = min(max(reduction, 0.0), 0.95)
    reduced = mesh.decimate_pro(reduction)
    reduced_faces = reduced.faces.reshape(-1, 4)[:, 1:]
    return np.asarray(reduced.points), reduced_faces.astype(np.int64), int(reduced.n_cells)


def export_glb(coords, faces, output_path: Path):
    trimesh = _require("trimesh", "trimesh")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=coords, faces=faces, process=False)
    mesh.export(str(output_path))


def write_meta(output_path: Path, meta: dict):
    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta_path


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
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output GLB path",
    )
    parser.add_argument(
        "--target_faces",
        type=int,
        default=100_000,
        help="Approximate target face count after decimation",
    )
    args = parser.parse_args()

    lh_coords, lh_tris = read_hemisphere_mesh(args.recon_dir, args.subject, "lh")
    rh_coords, rh_tris = read_hemisphere_mesh(args.recon_dir, args.subject, "rh")
    coords, faces = merge_hemispheres(lh_coords, lh_tris, rh_coords, rh_tris)
    original_faces = len(faces)

    coords, faces, decimated_faces = decimate_mesh(coords, faces, args.target_faces)
    export_glb(coords, faces, args.output)

    bounds = {
        "xmin": float(coords[:, 0].min()),
        "xmax": float(coords[:, 0].max()),
        "ymin": float(coords[:, 1].min()),
        "ymax": float(coords[:, 1].max()),
        "zmin": float(coords[:, 2].min()),
        "zmax": float(coords[:, 2].max()),
    }
    center = coords.mean(axis=0).tolist()
    meta = {
        "subject": args.subject,
        "surf": "pial",
        "coordinate_space": "RAS_mm",
        "source": {
            "lh_pial": str(args.recon_dir / args.subject / "surf" / "lh.pial"),
            "rh_pial": str(args.recon_dir / args.subject / "surf" / "rh.pial"),
        },
        "n_vertices_original": int(len(lh_coords) + len(rh_coords)),
        "n_faces_original": int(original_faces),
        "n_vertices": int(len(coords)),
        "n_faces": int(decimated_faces),
        "target_faces": args.target_faces,
        "bounds": bounds,
        "center": center,
        "camera_hint": {
            "position": [0, -150, 95],
            "fov": 45,
        },
    }
    meta_path = write_meta(args.output, meta)

    size_mb = args.output.stat().st_size / 1e6
    print(f"Wrote {args.output} ({size_mb:.2f} MB, {decimated_faces} faces)")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
