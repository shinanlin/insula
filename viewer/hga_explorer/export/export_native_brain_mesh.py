#!/usr/bin/env python3
"""Export per-subject native pial surfaces (lh + rh) to decimated GLB files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_EXPORT_DIR = Path(__file__).resolve().parent
if str(_EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPORT_DIR))

from export_average_brain_mesh import (
    decimate_mesh,
    export_glb,
    merge_hemispheres,
    read_hemisphere_mesh,
    write_meta,
)

VIEWER_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = VIEWER_ROOT / "public" / "assets" / "native"
DEFAULT_RECON_ROOT = Path("/cwork/ns458/ECoG_Recon")

FULL_COHORT_SUBJECTS = [
    "D0023", "D0024", "D0028", "D0029", "D0035", "D0042", "D0053", "D0054",
    "D0055", "D0057", "D0059", "D0063", "D0066", "D0068", "D0069", "D0070",
    "D0071", "D0077", "D0079", "D0084", "D0086", "D0094", "D0096", "D0100",
    "D0102", "D0103",
]


def recon_subject_id(subject: str) -> str:
    return f"D{int(subject.lstrip('D0'))}"


def export_subject_native_mesh(
    recon_root: Path,
    subject: str,
    output_dir: Path,
    target_faces: int,
) -> Path | None:
    recon_id = recon_subject_id(subject)
    try:
        lh_coords, lh_tris = read_hemisphere_mesh(recon_root, recon_id, "lh")
        rh_coords, rh_tris = read_hemisphere_mesh(recon_root, recon_id, "rh")
    except FileNotFoundError as exc:
        print(f"SKIP {subject}: {exc}")
        return None

    coords, faces = merge_hemispheres(lh_coords, lh_tris, rh_coords, rh_tris)
    original_faces = len(faces)
    coords, faces, decimated_faces = decimate_mesh(coords, faces, target_faces)

    output_path = output_dir / f"{subject}_pial.glb"
    export_glb(coords, faces, output_path)

    bounds = {
        "xmin": float(coords[:, 0].min()),
        "xmax": float(coords[:, 0].max()),
        "ymin": float(coords[:, 1].min()),
        "ymax": float(coords[:, 1].max()),
        "zmin": float(coords[:, 2].min()),
        "zmax": float(coords[:, 2].max()),
    }
    meta = {
        "subject": subject,
        "recon_id": recon_id,
        "surf": "pial",
        "coordinate_space": "native_RAS_mm",
        "source": {
            "lh_pial": str(recon_root / recon_id / "surf" / "lh.pial"),
            "rh_pial": str(recon_root / recon_id / "surf" / "rh.pial"),
        },
        "n_vertices_original": int(len(lh_coords) + len(rh_coords)),
        "n_faces_original": int(original_faces),
        "n_vertices": int(len(coords)),
        "n_faces": int(decimated_faces),
        "target_faces": target_faces,
        "bounds": bounds,
        "center": coords.mean(axis=0).tolist(),
    }
    write_meta(output_path, meta)
    size_mb = output_path.stat().st_size / 1e6
    print(f"Wrote {output_path} ({size_mb:.2f} MB, {decimated_faces} faces)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recon_root", type=Path, default=DEFAULT_RECON_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=FULL_COHORT_SUBJECTS,
        help="BIDS-style subject ids, e.g. D0094 D0071",
    )
    parser.add_argument("--target_faces", type=int, default=80_000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    skipped = []
    for subject in args.subjects:
        path = export_subject_native_mesh(
            args.recon_root,
            subject,
            args.output_dir,
            args.target_faces,
        )
        if path is None:
            skipped.append(subject)
        else:
            exported.append(subject)

    index = {
        "subjects": exported,
        "skipped": skipped,
        "pattern": "{subject}_pial.glb",
    }
    index_path = args.output_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(exported)} native meshes; skipped {len(skipped)}")
    print(f"Wrote {index_path}")


if __name__ == "__main__":
    main()
