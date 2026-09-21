#!/usr/bin/env python3
"""Native-subject SEEG shaft examples: cylinders + Hammers insula on pial and T1."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import rootutils

rootutils.setup_root(
    __file__,
    indicator=".project-root",
    pythonpath=True,
    cwd=True,
)

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import mne
import nibabel as nib
import numpy as np
import pandas as pd
import pyvista as pv
from mne import Label
from mne.viz import Brain

from src.paths import img_dir, save_svg

RECON_DIR = Path("/cwork/ns458/ECoG_Recon")
MAPER_ROOT = Path("/cwork/ns458/maper_run")
BIDS_PARC = Path(
    "/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/derivatives/parcellation"
)
CONTACT_H = 2.0
TIP_EXTEND = 1.5
SKULL_EXTEND = 8.0
SLICE_TOL_MM = 4.0
HAMMERS_INSULA_LEFT = (20, 86, 88, 90, 92, 94)
HAMMERS_INSULA_RIGHT = (21, 87, 89, 91, 93, 95)
INSULA_PATTERNS = (
    "G_insular_short",
    "G_Ins_lg_and_S_cent_ins",
    "S_circular_insula_ant",
    "S_circular_insula_inf",
    "S_circular_insula_sup",
)
RED = "#A9373B"
BLUE = "#2369BD"
INSULA_COLOR = "#D4AF37"
INSULA_ALPHA = 0.6
CM = 1 / 2.54
FONTSIZE = 7
SCALE = {
    "overview": dict(shaft_r=0.45, ring_r=0.68, cortex_alpha=0.1),
    "zoom": dict(shaft_r=0.5, ring_r=0.8, cortex_alpha=0.05),
}

# Clean consecutive AIC/PIC runs; D0064 LAI is kept as the original reference.
CASES = (
    dict(subject="D0061", strip="LAI", why="left AIC, 6 consecutive, mix=0"),
    dict(subject="D0075", strip="RAI", why="right AIC, 6 consecutive"),
    dict(subject="D0079", strip="LPI", why="left PIC, 9 consecutive"),
    dict(subject="D0071", strip="RIP", why="right PIC, 6 consecutive, mix=0"),
    dict(subject="D0064", strip="LAI", why="original Fig1 shaft (for comparison)"),
)


def fs_subject(bids_subject: str) -> str:
    match = re.match(r"D0*(\d+)$", bids_subject)
    if not match:
        raise ValueError(f"unrecognized subject {bids_subject}")
    return f"D{int(match.group(1))}"


def _contact_num(name: str) -> int:
    match = re.search(r"(\d+)$", str(name))
    return int(match.group(1)) if match else 0


def load_shaft_geometry(subject: str, strip: str) -> tuple[list[str], np.ndarray, set[str], set[str]]:
    csv = BIDS_PARC / f"sub-{subject}" / "bipolar" / f"sub-{subject}_hammers.csv"
    parc = pd.read_csv(csv).set_index("name")
    names = parc.index.astype(str)
    mask = names.str.match(rf"{re.escape(subject)}_{re.escape(strip)}\d")
    df = parc.loc[mask]
    if df.empty:
        raise ValueError(f"no channels for {subject} {strip} in {csv}")
    contacts: dict[str, np.ndarray] = {}
    aic: set[str] = set()
    pic: set[str] = set()
    mix = df["mix"].astype(bool) if "mix" in df.columns else pd.Series(False, index=df.index)
    for name, row in df.iterrows():
        c1, c2 = str(row["contact_1"]), str(row["contact_2"])
        p1 = np.array([row["x1"], row["y1"], row["z1"]], float)
        p2 = np.array([row["x2"], row["y2"], row["z2"]], float)
        contacts[c1] = p1
        contacts[c2] = p2
        if (not bool(mix.loc[name])) and row["roi"] == "AIC":
            aic.update((c1, c2))
        if (not bool(mix.loc[name])) and row["roi"] == "PIC":
            pic.update((c1, c2))
    order = sorted(contacts, key=_contact_num)
    coords = np.vstack([contacts[c] for c in order])
    return order, coords, aic, pic


def _add_cylinder(plotter, p0, p1, radius, color, opacity=1.0) -> None:
    direction = np.asarray(p1, float) - np.asarray(p0, float)
    height = float(np.linalg.norm(direction))
    if height < 1e-6:
        return
    plotter.add_mesh(
        pv.Cylinder(
            center=(np.asarray(p0) + np.asarray(p1)) / 2.0,
            direction=direction,
            radius=radius,
            height=height,
            resolution=24,
        ),
        color=color,
        opacity=opacity,
        smooth_shading=True,
    )


def add_seeg_shaft(brain, order, coords, aic_contacts, pic_contacts, scale) -> None:
    plotter = brain._renderer.plotter
    axis = coords[-1] - coords[0]
    axis = axis / np.linalg.norm(axis)
    tip = coords[0] - axis * TIP_EXTEND
    skull = coords[-1] + axis * SKULL_EXTEND
    _add_cylinder(plotter, tip, skull, scale["shaft_r"], "#B8B8B8")
    plotter.add_mesh(
        pv.Sphere(radius=scale["shaft_r"], center=tip, theta_resolution=16, phi_resolution=16),
        color="#B8B8B8",
        smooth_shading=True,
    )
    for name, coord in zip(order, coords):
        if name in aic_contacts:
            color = RED
        elif name in pic_contacts:
            color = BLUE
        else:
            color = "#595959"
        _add_cylinder(
            plotter,
            coord - axis * (CONTACT_H / 2.0),
            coord + axis * (CONTACT_H / 2.0),
            scale["ring_r"],
            color,
        )


def hammers_insula_label(fsid: str, bids_subject: str, hemi: str) -> Label | None:
    pial, _ = mne.read_surface(str(RECON_DIR / fsid / "surf" / f"{hemi}.pial"))
    orig = nib.load(str(RECON_DIR / fsid / "mri" / "orig.mgz"))
    maper_img = nib.load(str(MAPER_ROOT / bids_subject / "output" / f"f30-seg95-{bids_subject}.nii.gz"))
    maper = np.asarray(maper_img.dataobj).squeeze().astype(int)
    orig_vox = nib.affines.apply_affine(
        np.linalg.inv(np.asarray(orig.header.get_vox2ras_tkr(), float)),
        pial,
    )
    scanner = nib.affines.apply_affine(orig.affine, orig_vox)
    ijk = nib.affines.apply_affine(np.linalg.inv(maper_img.affine), scanner).round().astype(int)
    valid = (
        (ijk[:, 0] >= 0)
        & (ijk[:, 0] < maper.shape[0])
        & (ijk[:, 1] >= 0)
        & (ijk[:, 1] < maper.shape[1])
        & (ijk[:, 2] >= 0)
        & (ijk[:, 2] < maper.shape[2])
    )
    ids = HAMMERS_INSULA_LEFT if hemi == "lh" else HAMMERS_INSULA_RIGHT
    hit = np.zeros(len(pial), dtype=bool)
    if valid.any():
        vox = ijk[valid]
        hit[valid] = np.isin(maper[vox[:, 0], vox[:, 1], vox[:, 2]], ids)
    vertices = np.flatnonzero(hit)
    if vertices.size < 50:
        return None
    return Label(vertices=vertices, hemi=hemi, name="insula-hammers", subject=fsid)


def add_insula_structure(brain, fsid: str, hemi: str, hammers_label: Label | None) -> str:
    if hammers_label is not None and len(hammers_label.vertices) >= 50:
        brain.add_label(hammers_label, borders=False, color=INSULA_COLOR, alpha=INSULA_ALPHA)
        return f"hammers:{len(hammers_label.vertices)}"
    labs = mne.read_labels_from_annot(
        subject=fsid,
        parc="aparc.a2009s",
        surf_name="pial",
        hemi=hemi,
        subjects_dir=str(RECON_DIR),
    )
    n_lab = 0
    n_vtx = 0
    for lab in labs:
        if lab.hemi == hemi and any(pattern in lab.name for pattern in INSULA_PATTERNS):
            brain.add_label(lab, borders=False, color=INSULA_COLOR, alpha=INSULA_ALPHA)
            n_lab += 1
            n_vtx += len(lab.vertices)
    return f"destrieux:{n_lab}/{n_vtx}"


def build_brain(fsid: str, hemi: str, order, coords, aic, pic, scale_name: str, insula_label: Label | None):
    scale = SCALE[scale_name]
    brain = Brain(
        fsid,
        subjects_dir=str(RECON_DIR),
        surf="pial",
        hemi=hemi,
        background="white",
        show=False,
        cortex=(0.9, 0.9, 0.9),
        alpha=scale["cortex_alpha"],
        size=(800, 800),
    )
    overlay = add_insula_structure(brain, fsid, hemi, insula_label)
    add_seeg_shaft(brain, order, coords, aic, pic, scale)
    return brain, overlay


def sagittal_plane_tkr(img, x_mm):
    tkr = np.asarray(img.header.get_vox2ras_tkr(), dtype=float)
    vol = np.asarray(img.dataobj, dtype=float)
    index = int(np.round((x_mm - tkr[0, 3]) / tkr[0, 0]))
    index = int(np.clip(index, 0, vol.shape[0] - 1))
    sl = vol[index, ::-1, :]
    k0, k1 = -0.5, vol.shape[2] - 0.5
    j_bottom, j_top = vol.shape[1] - 0.5, -0.5
    extent = [
        float(tkr[1, 2] * k0 + tkr[1, 3]),
        float(tkr[1, 2] * k1 + tkr[1, 3]),
        float(tkr[2, 1] * j_bottom + tkr[2, 3]),
        float(tkr[2, 1] * j_top + tkr[2, 3]),
    ]
    actual_x = float(tkr[0, 0] * index + tkr[0, 3])
    return sl, extent, index, actual_x


def draw_lai_on_slice(ax, x0, order, coords, aic, pic, tol):
    half = CONTACT_H / 2.0
    axis = coords[-1] - coords[0]
    axis = axis / np.linalg.norm(axis)
    yz = np.array([axis[1], axis[2]], float)
    nrm = np.linalg.norm(yz)
    if nrm < 1e-6:
        return 0
    yz = yz / nrm
    near = np.abs(coords[:, 0] - x0) <= tol
    if near.sum() >= 2:
        pts = coords[near][:, 1:3]
        order_pts = np.argsort(pts[:, 1])
        ax.plot(
            pts[order_pts, 0],
            pts[order_pts, 1],
            color="#B8B8B8",
            lw=0.7,
            zorder=4,
            solid_capstyle="round",
        )
    n_drawn = 0
    for name, coord in zip(order, coords):
        if abs(coord[0] - x0) > tol:
            continue
        if name in aic:
            color = RED
        elif name in pic:
            color = BLUE
        else:
            color = "#595959"
        ax.plot(
            [coord[1] - half * yz[0], coord[1] + half * yz[0]],
            [coord[2] - half * yz[1], coord[2] + half * yz[1]],
            color=color,
            lw=2.4,
            solid_capstyle="round",
            zorder=6,
        )
        n_drawn += 1
    return n_drawn


def plot_case(case: dict, out_dir: Path) -> Path:
    subject = case["subject"]
    strip = case["strip"]
    fsid = fs_subject(subject)
    hemi = "lh" if strip.startswith("L") else "rh"
    order, coords, aic, pic = load_shaft_geometry(subject, strip)
    focus = np.vstack([coords[order.index(c)] for c in order if c in aic or c in pic]) if (aic or pic) else coords
    focus_center = focus.mean(axis=0)
    shaft_center = coords.mean(axis=0)

    insula_label = hammers_insula_label(fsid, subject, hemi)
    overview_brain, overlay = build_brain(
        fsid, hemi, order, coords, aic, pic, "overview", insula_label
    )
    zoom_brain, _ = build_brain(fsid, hemi, order, coords, aic, pic, "zoom", insula_label)
    if hemi == "lh":
        overview_brain.show_view(azimuth=180, elevation=80, distance=320, focalpoint=shaft_center)
        zoom_brain.show_view(azimuth=120, elevation=85, distance=100, focalpoint=focus_center)
    else:
        overview_brain.show_view(azimuth=0, elevation=80, distance=320, focalpoint=shaft_center)
        zoom_brain.show_view(azimuth=60, elevation=85, distance=100, focalpoint=focus_center)
    overview = overview_brain.screenshot(mode="rgb")
    zoom = zoom_brain.screenshot(mode="rgb")
    overview_brain.close()
    zoom_brain.close()

    t1_path = RECON_DIR / fsid / "mri" / "T1.mgz"
    if not t1_path.is_file():
        t1_path = RECON_DIR / fsid / "mri" / "orig.mgz"
    t1_img = nib.load(str(t1_path))
    maper_path = MAPER_ROOT / subject / "output" / f"f30-seg95-{subject}.nii.gz"
    slice_x = float(focus_center[0])
    sl, extent, vox_i, actual_x = sagittal_plane_tkr(t1_img, slice_x)
    positive = sl[sl > 0]
    vmax = float(np.percentile(positive, 99)) if positive.size else float(sl.max() or 1.0)
    maper_img = nib.load(str(maper_path))
    maper_sl = np.asarray(maper_img.dataobj).squeeze()[vox_i, ::-1, :].astype(int)
    insula_ids = HAMMERS_INSULA_LEFT if hemi == "lh" else HAMMERS_INSULA_RIGHT
    insula_mask = np.isin(maper_sl, insula_ids)

    fig, axes = plt.subplots(1, 3, figsize=(16.8 * CM, 5 * CM))
    axes[0].imshow(overview)
    axes[0].set_axis_off()
    axes[1].imshow(zoom)
    axes[1].set_axis_off()
    axes[2].imshow(
        sl,
        origin="lower",
        extent=extent,
        cmap="gray",
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    if np.any(insula_mask):
        axes[2].contourf(
            insula_mask.astype(float),
            levels=[0.5, 1.5],
            extent=extent,
            colors=[INSULA_COLOR],
            alpha=0.22,
            origin="lower",
            zorder=2,
        )
        axes[2].contour(
            insula_mask.astype(float),
            levels=[0.5],
            extent=extent,
            colors=[INSULA_COLOR],
            linewidths=0.8,
            origin="lower",
            zorder=5,
        )
    n_slice = draw_lai_on_slice(axes[2], slice_x, order, coords, aic, pic, SLICE_TOL_MM)
    pad = 28.0
    axes[2].set_xlim(focus_center[1] - pad, focus_center[1] + pad)
    axes[2].set_ylim(focus_center[2] - pad, focus_center[2] + pad)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    for spine in axes[2].spines.values():
        spine.set_color("0.35")
        spine.set_linewidth(0.5)
    fig.subplots_adjust(wspace=0.04, top=0.88)
    axes[1].plot([], [], color=RED, lw=1.5, label="AIC contact")
    axes[1].plot([], [], color=BLUE, lw=1.5, label="PIC contact")
    axes[1].plot([], [], color="#595959", lw=1.5, label="other contact")
    axes[1].plot([], [], color=INSULA_COLOR, lw=1.2, label="insula (Hammers)")
    axes[1].legend(
        loc="lower right",
        fontsize=FONTSIZE - 1,
        frameon=False,
        handlelength=1.2,
        borderaxespad=0.2,
    )
    fig.suptitle(
        f"{subject}  {strip}  ·  native {fsid}  ·  {case['why']}",
        fontsize=FONTSIZE,
        y=0.98,
    )
    out = out_dir / f"sub-{subject}_strip-{strip}_seeg.svg"
    save_svg(fig, out, close=True)
    print(
        f"wrote {out} aic={len(aic)} pic={len(pic)} overlay={overlay} "
        f"slice_x={slice_x:.1f}->{actual_x:.1f} n_on_slice={n_slice} "
        f"insula_vox={int(insula_mask.sum())}",
        flush=True,
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=img_dir("fig1") / "seeg_examples",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Optional subject:strip filters, e.g. D0061:LAI D0079:LPI",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plt.rcParams["svg.fonttype"] = "none"
    cases = CASES
    if args.cases:
        wanted = {tuple(item.split(":", 1)) for item in args.cases}
        cases = tuple(c for c in CASES if (c["subject"], c["strip"]) in wanted)
        if not cases:
            raise SystemExit(f"no matching cases in {wanted}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        plot_case(case, args.out_dir)
    print("done", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
