#!/usr/bin/env python3
"""Exact-slice QC plots for corrected D0044 MAPER Insula labels."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np
import pandas as pd


ORIG = Path("/cwork/ns458/ECoG_Recon/D44/mri/orig.mgz")
DESTRIEUX = Path("/cwork/ns458/ECoG_Recon/D44/mri/aparc.a2009s+aseg.mgz")
MAPER = Path("/cwork/ns458/maper_run/output/f30-seg95-D0044.nii.gz")
PARCELLATION = Path(
    "/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/derivatives/"
    "parcellation/sub-D0044/bipolar/sub-D0044_aparc2009s.csv"
)
MAPER_TABLE = Path("/cwork/ns458/maper_run/sub-D0044_desc-maper_insula_v2.csv")
OUTPUT = Path("/cwork/ns458/maper_run/qc_v2/electrode_exact_slices")

MAPER_INSULA_IDS = [20, 21, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]
MAPER_ORANGE = ListedColormap(["#f26b38"])
DESTRIEUX_INSULA_IDS = [
    11117, 11118, 11148, 11149, 11150,
    12117, 12118, 12148, 12149, 12150,
]


def canonical_data(data: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = nib.as_closest_canonical(nib.Nifti1Image(data, affine))
    return np.asarray(image.dataobj), image.affine


def make_figure(
    name: str,
    point: np.ndarray,
    brain: np.ndarray,
    maper_mask: np.ndarray,
    destrieux_mask: np.ndarray,
    maper_region: str,
) -> plt.Figure:
    center = np.rint(point).astype(int)
    planes = [
        (0, center[0], (1, 2), "sagittal"),
        (1, center[1], (0, 2), "coronal"),
        (2, center[2], (0, 1), "axial"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for axis_object, (axis, index, dimensions, title) in zip(axes, planes):
        background = np.take(brain, index, axis=axis).T
        maper_slice = np.take(maper_mask, index, axis=axis).T
        destrieux_slice = np.take(destrieux_mask, index, axis=axis).T

        axis_object.imshow(background, cmap="gray", origin="lower")
        axis_object.imshow(
            np.ma.masked_where(maper_slice < 0.5, maper_slice),
            cmap=MAPER_ORANGE, origin="lower", alpha=0.62, vmin=0, vmax=1,
        )
        if destrieux_slice.any():
            axis_object.contour(
                destrieux_slice, levels=[0.5], colors="#00ff32",
                linewidths=1.2, origin="lower",
            )
        axis_object.scatter(
            point[dimensions[0]], point[dimensions[1]], s=55,
            c="white", edgecolors="black", linewidths=0.9, zorder=5,
        )
        axis_object.set_title(title, fontsize=15)
        axis_object.axis("off")

    region_text = maper_region if maper_region else "non-Insula at exact point"
    figure.suptitle(
        f"{name} | MAPER: {region_text} | orange MAPER Insula | green Destrieux INS",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    orig = nib.load(ORIG)
    canonical_orig = nib.as_closest_canonical(orig)
    brain = np.asarray(canonical_orig.dataobj)

    maper = nib.load(MAPER)
    maper_native = np.isin(
        np.asarray(maper.dataobj).squeeze().astype(int), MAPER_INSULA_IDS,
    ).astype(np.uint8)
    maper_mask, maper_affine = canonical_data(maper_native, maper.affine)

    destrieux = nib.load(DESTRIEUX)
    destrieux_native = np.isin(
        np.asarray(destrieux.dataobj).astype(int), DESTRIEUX_INSULA_IDS,
    ).astype(np.uint8)
    destrieux_mask, destrieux_affine = canonical_data(
        destrieux_native, destrieux.affine,
    )

    if brain.shape != maper_mask.shape or brain.shape != destrieux_mask.shape:
        raise ValueError(
            f"Canonical geometry mismatch: {brain.shape}, "
            f"{maper_mask.shape}, {destrieux_mask.shape}"
        )
    if not np.allclose(canonical_orig.affine, maper_affine) or not np.allclose(
        canonical_orig.affine, destrieux_affine
    ):
        raise ValueError("Canonical affines do not match")

    parcellation = pd.read_csv(PARCELLATION)
    insula = parcellation[
        parcellation["roi"].astype(str).str.contains("INS", case=False, na=False)
    ].copy()
    maper_table = pd.read_csv(MAPER_TABLE).set_index("name")

    pdf_path = OUTPUT.parent / "D0044_MAPER_vs_Destrieux_native_INS.pdf"
    with PdfPages(pdf_path) as pdf:
        for _, row in insula.iterrows():
            tkras = row[["x", "y", "z"]].to_numpy(float)
            native_ijk = (
                np.linalg.inv(orig.header.get_vox2ras_tkr())
                @ np.r_[tkras, 1.0]
            )[:3]
            scanner_ras = nib.affines.apply_affine(orig.affine, native_ijk)
            canonical_ijk = nib.affines.apply_affine(
                np.linalg.inv(canonical_orig.affine), scanner_ras,
            )
            region = ""
            if row["name"] in maper_table.index:
                value = maper_table.loc[row["name"], "maper_region6"]
                region = "" if pd.isna(value) else str(value)
            figure = make_figure(
                str(row["name"]), canonical_ijk, brain, maper_mask,
                destrieux_mask, region,
            )
            output_path = OUTPUT / f"{row['name']}.png"
            figure.savefig(output_path, dpi=200, bbox_inches="tight")
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)

    print(f"Saved {len(insula)} PNGs to {OUTPUT}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
