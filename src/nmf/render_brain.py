#!/usr/bin/env python3
"""Render corrected functional NMF labels on the bilateral insula surface.

Run in the ``ieeg`` environment after ``src.nmf.run_waveform_analysis``.
All heavy visualization imports are local so ``--help`` works in a lightweight
analysis environment.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


COLORS = {
    "sustained_ramping": "#A9373B",
    "sensory_transient": "#2369BD",
}
INSULA_PATTERNS = (
    "G_insular_short",
    "G_Ins_lg_and_S_cent_ins",
    "S_circular_insula_ant",
    "S_circular_insula_inf",
    "S_circular_insula_sup",
)


def render(args: argparse.Namespace) -> None:
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne
    import pyvista as pv
    from matplotlib.patches import Patch
    from mne.viz import Brain
    from scipy.spatial import cKDTree

    mne.viz.set_3d_backend("notebook")
    assignments = pd.read_csv(args.assignments)
    required = {"x", "y", "z", "functional_cluster", "dominance"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"Assignment table is missing columns: {sorted(missing)}")

    labels = mne.read_labels_from_annot(
        subject=args.fs_subject,
        parc="aparc.a2009s",
        surf_name="pial",
        hemi="both",
        subjects_dir=args.recon_dir,
    )
    pial = {}
    trees = {}
    centers = {}
    for hemi in ("lh", "rh"):
        coords, _ = mne.read_surface(
            str(args.recon_dir / args.fs_subject / "surf" / f"{hemi}.pial")
        )
        pial[hemi] = coords
        trees[hemi] = cKDTree(coords)
        vertices = []
        for label in labels:
            if label.hemi == hemi and any(
                pattern in label.name for pattern in INSULA_PATTERNS
            ):
                vertices.extend(label.vertices)
        centers[hemi] = coords[np.asarray(vertices, dtype=int)].mean(axis=0)

    screenshots = []
    for hemi, azimuth in (("lh", 180), ("rh", 0)):
        brain = Brain(
            args.fs_subject,
            subjects_dir=args.recon_dir,
            surf="pial",
            hemi=hemi,
            background="white",
            show=False,
            cortex=(0.9, 0.9, 0.9),
            alpha=0.08,
            size=(800, 800),
        )
        for label in labels:
            if label.hemi == hemi and any(
                pattern in label.name for pattern in INSULA_PATTERNS
            ):
                brain.add_label(
                    label, borders=False, color=(0.88, 0.88, 0.88), alpha=0.42
                )

        side = assignments.loc[
            assignments["x"].lt(0) if hemi == "lh" else assignments["x"].gt(0)
        ]
        coords = side[["x", "y", "z"]].to_numpy(float)
        if len(coords):
            _, nearest = trees[hemi].query(coords)
            projected = pial[hemi][nearest]
            sizes = args.size_min + (args.size_max - args.size_min) * side[
                "dominance"
            ].clip(0.5, 1.0).sub(0.5).div(0.5).to_numpy(float)
            for point, functional_cluster, size in zip(
                projected, side["functional_cluster"], sizes
            ):
                cloud = pv.PolyData(point.reshape(1, 3))
                brain._renderer.plotter.add_mesh(
                    cloud,
                    render_points_as_spheres=True,
                    point_size=float(size),
                    color=COLORS[functional_cluster],
                    lighting=False,
                )
        brain.show_view(
            azimuth=azimuth,
            elevation=90,
            distance=180,
            focalpoint=centers[hemi],
        )
        screenshots.append(brain.screenshot(mode="rgb"))
        brain.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    for axis, screenshot, title in zip(
        axes, screenshots, ("Left insula", "Right insula")
    ):
        axis.imshow(screenshot)
        axis.axis("off")
        axis.set_title(title)
    axes[0].legend(
        handles=[
            Patch(facecolor=COLORS["sustained_ramping"], label="Sustained/ramping"),
            Patch(facecolor=COLORS["sensory_transient"], label="Sensory/transient"),
        ],
        loc="upper left",
        framealpha=0.9,
    )
    fig.suptitle(
        f"Stimulus-derived functional clusters (n={len(assignments)}; size=dominance)"
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assignments",
        type=Path,
        default=Path("tmp/nmf_corrected/channel_assignments.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tmp/nmf_corrected/brain_functional_clusters.png"),
    )
    parser.add_argument(
        "--recon-dir", type=Path, default=Path("/cwork/ns458/ECoG_Recon")
    )
    parser.add_argument("--fs-subject", default="cvs_avg35_inMNI152")
    parser.add_argument("--size-min", type=float, default=9.0)
    parser.add_argument("--size-max", type=float, default=20.0)
    return parser.parse_args()


if __name__ == "__main__":
    render(parse_args())
