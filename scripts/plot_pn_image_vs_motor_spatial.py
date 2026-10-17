#!/usr/bin/env python3
"""Fig 2-style insula maps for Picture Naming Repeat.

Two rows, left/right hemispheres, frozen NMF colors:

1. Packaged-significant during **image** stimulus ``[0, 0.6]`` s.
2. Packaged-significant during **response**, and **not** during sound stimulus.

Run on a compute node (PyVista / ``mne.viz.Brain``). Login-node table check:

    python scripts/plot_pn_image_vs_motor_spatial.py --tables-only
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.nmf.waveform_analysis import FUNCTION_COLORS
from src.paths import hga_results_dir, img_dir, nmf_assignments_path, save_svg

EXCLUDE_SUBJECTS = {"D0121"}
CLUSTERS = ["sensory", "sustain", "motor"]
DISPLAY_LABELS = {
    "sensory": "sensory",
    "sustain": "scene",
    "motor": "motor",
}
STIM_WINDOW = (0.0, 0.6)
RESP_WINDOW = (-0.5, 1.0)
TASK = "PictureNaming"
DESCRIPTION = "Repeat"

cm = 1 / 2.54
fontsize = 7
POINT_SIZE = 10.0
KDE_BANDWIDTH_PX = 25.0
KDE_LEVEL_FRAC = 0.18
KDE_FILL_ALPHA = 0.12
KDE_MIN_POINTS = 2
CROP_PAD_PX = 24
INSULA_COLOR = (0.9, 0.9, 0.9)
INSULA_LABEL_ALPHA = 0.55
INSULA_PATTERNS = (
    "G_insular_short",
    "G_Ins_lg_and_S_cent_ins",
    "S_circular_insula_ant",
    "S_circular_insula_inf",
    "S_circular_insula_sup",
)
RECON_DIR = "/cwork/ns458/ECoG_Recon/"
TEMPLATE = "cvs_avg35_inMNI152"


def _norm_subject(subject: str) -> str:
    return str(subject).removeprefix("sub-")


def load_assignments() -> pd.DataFrame:
    meta = pd.read_csv(nmf_assignments_path())
    meta = meta.copy()
    meta["subject"] = meta["subject"].map(_norm_subject)
    meta = meta[meta["functional_cluster"].isin(CLUSTERS)]
    meta = meta[~meta["subject"].isin(EXCLUDE_SUBJECTS)]
    return meta.drop_duplicates("channel", keep="first").reset_index(drop=True)


def _glob_hga(proc: str, recording: str) -> list[Path]:
    root = hga_results_dir(TASK)
    pattern = f"sub-*/HGA/*_proc-{proc}_recording-{recording}_desc-{DESCRIPTION}_time.csv"
    return sorted(root.glob(pattern))


def significant_channels(proc: str, recording: str, t0: float, t1: float, wanted: set[str]) -> set[str]:
    hits: set[str] = set()
    for path in _glob_hga(proc, recording):
        subj = path.parent.parent.name.removeprefix("sub-")
        if subj in EXCLUDE_SUBJECTS:
            continue
        df = pd.read_csv(path, usecols=["channel", "time", "mask"])
        df["mask"] = df["mask"].astype(bool)
        keep = (
            df["channel"].isin(wanted)
            & (df["time"] >= t0)
            & (df["time"] <= t1)
            & df["mask"]
        )
        hits.update(df.loc[keep, "channel"].astype(str))
    return hits


def channel_sets(assignments: pd.DataFrame) -> dict[str, set[str]]:
    wanted = set(assignments["channel"].astype(str))
    image_stim = significant_channels("Stimulus", "image", *STIM_WINDOW, wanted)
    sound_stim = significant_channels("Stimulus", "sound", *STIM_WINDOW, wanted)
    resp_image = significant_channels("Response", "image", *RESP_WINDOW, wanted)
    resp_sound = significant_channels("Response", "sound", *RESP_WINDOW, wanted)
    response_any = resp_image | resp_sound
    motor_not_sound = response_any - sound_stim
    return {
        "image_stimulus": image_stim,
        "sound_stimulus": sound_stim,
        "response_any": response_any,
        "motor_not_sound": motor_not_sound,
    }


def summarize_sets(assignments: pd.DataFrame, sets: dict[str, set[str]]) -> pd.DataFrame:
    rows = []
    for name, channels in sets.items():
        sub = assignments[assignments["channel"].isin(channels)]
        counts = sub.groupby("functional_cluster")["channel"].nunique()
        row = {"set": name, "n": int(sub["channel"].nunique())}
        for cluster in CLUSTERS:
            row[cluster] = int(counts.get(cluster, 0))
        rows.append(row)
    return pd.DataFrame(rows)


def membership_table(assignments: pd.DataFrame, sets: dict[str, set[str]]) -> pd.DataFrame:
    out = assignments[
        ["channel", "subject", "roi", "hemi", "x", "y", "z", "functional_cluster"]
    ].copy()
    out["image_stimulus"] = out["channel"].isin(sets["image_stimulus"])
    out["sound_stimulus"] = out["channel"].isin(sets["sound_stimulus"])
    out["response_any"] = out["channel"].isin(sets["response_any"])
    out["motor_not_sound"] = out["channel"].isin(sets["motor_not_sound"])
    keep = out["image_stimulus"] | out["motor_not_sound"] | out["sound_stimulus"] | out["response_any"]
    return out.loc[keep].sort_values(["functional_cluster", "channel"]).reset_index(drop=True)


def _insula_vertices(labels, hemi: str) -> np.ndarray:
    verts = []
    for lab in labels:
        if lab.hemi == hemi and any(p in lab.name for p in INSULA_PATTERNS):
            verts.extend(lab.vertices)
    if not verts:
        raise RuntimeError(f"No insula labels for {hemi}")
    return np.unique(verts)


def _faces_to_polydata(coords, faces):
    import pyvista as pv

    faces_pv = np.hstack(
        [np.full((len(faces), 1), 3, dtype=np.int64), np.asarray(faces, dtype=np.int64)]
    ).ravel()
    return pv.PolyData(np.asarray(coords, dtype=float), faces_pv)


def insula_mesh(labels, hemi: str, pial: np.ndarray, faces: np.ndarray):
    verts = _insula_vertices(labels, hemi)
    keep = np.all(np.isin(faces, verts), axis=1)
    mesh = _faces_to_polydata(pial, faces[keep])
    center = pial[verts].mean(axis=0)
    return mesh, center


def snap_to_pial(coords, tree, pial):
    _, idx = tree.query(coords)
    return pial[idx]


def add_cluster_spheres(brain, coords, clusters, tree, pial, keep_mask=None):
    import pyvista as pv

    coords = np.asarray(coords, dtype=float)
    if coords.size == 0:
        return np.zeros((0, 3), dtype=float)
    snapped = snap_to_pial(coords, tree, pial)
    if keep_mask is None:
        keep_mask = np.ones(len(coords), dtype=bool)
    keep_mask = np.asarray(keep_mask, dtype=bool)
    for cluster in CLUSTERS:
        mask = (np.asarray(clusters) == cluster) & keep_mask
        if not mask.any():
            continue
        cloud = pv.PolyData(np.asarray(snapped[mask], dtype=float))
        brain._renderer.plotter.add_mesh(
            cloud,
            render_points_as_spheres=True,
            point_size=POINT_SIZE,
            color=FUNCTION_COLORS[cluster],
            lighting=False,
        )
    return snapped


def project_xy(plotter, points, image_shape):
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=float)
    if hasattr(plotter, "project_points_to_pixels"):
        pix = np.asarray(plotter.project_points_to_pixels(pts), dtype=float)
        xy = np.asarray(pix[:, :2], dtype=float)
    else:
        import vtk

        renderer = plotter.renderer
        coordinate = vtk.vtkCoordinate()
        coordinate.SetCoordinateSystemToWorld()
        xy = np.zeros((len(pts), 2), dtype=float)
        for i, (x, y, z) in enumerate(pts):
            coordinate.SetValue(float(x), float(y), float(z))
            d = coordinate.GetComputedDisplayValue(renderer)
            xy[i] = (float(d[0]), float(d[1]))
    h = int(image_shape[0])
    xy[:, 1] = (h - 1) - xy[:, 1]
    return xy


def _keep_ring(ring, pts) -> bool:
    if len(pts) < KDE_MIN_POINTS:
        return False
    return int(MplPath(np.asarray(ring, dtype=float)).contains_points(pts).sum()) >= KDE_MIN_POINTS


def kde_contour_rings(xy, image_shape, *, bandwidth_px=KDE_BANDWIDTH_PX, level_frac=KDE_LEVEL_FRAC):
    from scipy.ndimage import gaussian_filter

    pts = np.asarray(xy, dtype=float)
    if len(pts) < 3:
        return []
    h, w = int(image_shape[0]), int(image_shape[1])
    dens = np.zeros((h, w), dtype=float)
    xs = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    for x, y in zip(xs, ys):
        dens[y, x] += 1.0
    dens = gaussian_filter(dens, sigma=float(bandwidth_px))
    peak = float(dens.max())
    if peak <= 0:
        return []
    level = peak * float(level_frac)
    try:
        from skimage.measure import find_contours

        contours = find_contours(dens, level=level)
        rings = []
        for cont in contours:
            if len(cont) < 8:
                continue
            ring = np.column_stack([cont[:, 1], cont[:, 0]])
            ring = np.vstack([ring, ring[0]])
            if _keep_ring(ring, pts):
                rings.append(ring)
        return rings
    except ImportError:
        fig_c, ax_c = plt.subplots()
        yy, xx = np.mgrid[0:h, 0:w]
        cs = ax_c.contour(xx, yy, dens, levels=[level])
        rings = []
        for segs in cs.allsegs:
            for seg in segs:
                if len(seg) < 8:
                    continue
                ring = np.vstack([seg, seg[0]])
                if _keep_ring(ring, pts):
                    rings.append(ring)
        plt.close(fig_c)
        return rings


def draw_rings(ax, rings, color):
    for ring in rings:
        ax.fill(ring[:, 0], ring[:, 1], color=color, alpha=KDE_FILL_ALPHA, linewidth=0, zorder=3)
        ax.plot(
            ring[:, 0],
            ring[:, 1],
            color=color,
            linewidth=0.9,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=4,
        )


def draw_ap_dv(ax, *, anterior_is_left: bool, corner: str):
    if corner == "lower left":
        x0, y0, arm = 0.16, 0.18, 0.08
    else:
        x0, y0, arm = 0.91, 0.16, 0.07
    gap = 0.012
    if anterior_is_left:
        tips = {"A": (x0 - arm, y0), "P": (x0 + arm, y0), "D": (x0, y0 + arm), "V": (x0, y0 - arm)}
    else:
        tips = {"A": (x0 + arm, y0), "P": (x0 - arm, y0), "D": (x0, y0 + arm), "V": (x0, y0 - arm)}
    for label, (tx, ty) in tips.items():
        sx = x0 + gap * np.sign(tx - x0) if tx != x0 else x0
        sy = y0 + gap * np.sign(ty - y0) if ty != y0 else y0
        ax.annotate(
            "",
            xy=(tx, ty),
            xytext=(sx, sy),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="0.25", lw=0.7, mutation_scale=7),
            annotation_clip=False,
        )
        px = 0.03 * np.sign(tx - x0) if tx != x0 else 0.0
        py = 0.035 * np.sign(ty - y0) if ty != y0 else 0.0
        ax.text(
            tx + px,
            ty + py,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=fontsize,
            color="0.25",
            clip_on=False,
        )


def crop_to_content(img, xy, pad=CROP_PAD_PX):
    rgb = np.asarray(img)
    mask = np.any(rgb < 250, axis=2)
    if not mask.any():
        return rgb, np.asarray(xy, dtype=float)
    ys, xs = np.nonzero(mask)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(rgb.shape[0], int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(rgb.shape[1], int(xs.max()) + 1 + pad)
    xy_c = np.asarray(xy, dtype=float).copy()
    if xy_c.size:
        xy_c[:, 0] -= x0
        xy_c[:, 1] -= y0
    return rgb[y0:y1, x0:x1], xy_c


def build_meshes():
    import mne
    from scipy.spatial import cKDTree

    labels = mne.read_labels_from_annot(
        subject=TEMPLATE,
        parc="aparc.a2009s",
        surf_name="pial",
        hemi="both",
        subjects_dir=RECON_DIR,
    )
    lh_pial_coords, lh_faces = mne.read_surface(f"{RECON_DIR}/{TEMPLATE}/surf/lh.pial")
    rh_pial_coords, rh_faces = mne.read_surface(f"{RECON_DIR}/{TEMPLATE}/surf/rh.pial")
    meshes = {
        "labels": labels,
        "lh_pial": lh_pial_coords,
        "rh_pial": rh_pial_coords,
        "lh_tree": cKDTree(lh_pial_coords),
        "rh_tree": cKDTree(rh_pial_coords),
    }
    meshes["lh_insula"], meshes["lh_center"] = insula_mesh(
        labels, "lh", lh_pial_coords, lh_faces
    )
    meshes["rh_insula"], meshes["rh_center"] = insula_mesh(
        labels, "rh", rh_pial_coords, rh_faces
    )
    return meshes


def render_hemi(hemi: str, coords, clusters, meshes, sphere_mask=None):
    import mne
    from mne.viz import Brain

    mne.viz.set_3d_backend("notebook")
    brain = Brain(
        TEMPLATE,
        subjects_dir=RECON_DIR,
        surf="pial",
        hemi=hemi,
        background="white",
        show=False,
        cortex=INSULA_COLOR,
        alpha=0.0,
        size=(800, 800),
    )
    plotter = brain._renderer.plotter
    mesh = meshes["lh_insula"] if hemi == "lh" else meshes["rh_insula"]
    plotter.add_mesh(
        mesh,
        color=INSULA_COLOR,
        opacity=INSULA_LABEL_ALPHA,
        lighting=True,
        smooth_shading=True,
    )
    tree = meshes["lh_tree"] if hemi == "lh" else meshes["rh_tree"]
    pial = meshes["lh_pial"] if hemi == "lh" else meshes["rh_pial"]
    snapped = add_cluster_spheres(brain, coords, clusters, tree, pial, keep_mask=sphere_mask)
    center = meshes["lh_center"] if hemi == "lh" else meshes["rh_center"]
    az = 180 if hemi == "lh" else 0
    brain.show_view(azimuth=az, elevation=90, distance=180, focalpoint=center)
    for _ in range(3):
        plotter.render()
    img = np.asarray(brain.screenshot(mode="rgb"))
    xy = project_xy(plotter, snapped, img.shape[:2])
    brain.close()
    return img, xy


def plot_spatial(assignments: pd.DataFrame, sets: dict[str, set[str]], out_svg: Path) -> Path:
    spatial = assignments.loc[
        assignments["functional_cluster"].isin(CLUSTERS)
        & assignments[["x", "y", "z"]].notna().all(axis=1)
    ].copy()
    cord = spatial[["x", "y", "z"]].to_numpy(float)
    clust = spatial["functional_cluster"].to_numpy()
    channels = spatial["channel"].to_numpy()
    mask_lh = cord[:, 0] < 0
    mask_rh = cord[:, 0] > 0

    rows = (
        ("image_stimulus", "Significant during image"),
        ("motor_not_sound", "Significant during response, not sound"),
    )
    meshes = build_meshes()

    fig, axes = plt.subplots(2, 2, figsize=(8 * cm, 7.2 * cm))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for row_i, (set_name, row_title) in enumerate(rows):
            keep = np.isin(channels, list(sets[set_name]))
            n_row = int(keep.sum())
            for col_i, (hemi, hemi_mask, title) in enumerate(
                (("lh", mask_lh, "Left"), ("rh", mask_rh, "Right"))
            ):
                img, xy = render_hemi(
                    hemi,
                    cord[hemi_mask],
                    clust[hemi_mask],
                    meshes,
                    sphere_mask=keep[hemi_mask],
                )
                img, xy = crop_to_content(img, xy)
                ax = axes[row_i, col_i]
                ax.imshow(img)
                ax.set_xlim(0, img.shape[1])
                ax.set_ylim(img.shape[0], 0)
                ax.axis("off")
                if row_i == 0:
                    ax.set_title(title, fontsize=fontsize)
                in_frame = (
                    (xy[:, 0] >= 0)
                    & (xy[:, 0] < img.shape[1])
                    & (xy[:, 1] >= 0)
                    & (xy[:, 1] < img.shape[0])
                )
                shown = keep[hemi_mask]
                for cluster in CLUSTERS:
                    pts = xy[in_frame & shown & (np.asarray(clust[hemi_mask]) == cluster)]
                    draw_rings(
                        ax,
                        kde_contour_rings(pts, img.shape[:2], bandwidth_px=KDE_BANDWIDTH_PX),
                        FUNCTION_COLORS[cluster],
                    )
                if title == "Left":
                    draw_ap_dv(ax, anterior_is_left=True, corner="lower left")
                    ax.text(
                        -0.04,
                        0.5,
                        f"{row_title}\n(n={n_row})",
                        transform=ax.transAxes,
                        rotation=90,
                        ha="right",
                        va="center",
                        fontsize=fontsize,
                    )
                else:
                    draw_ap_dv(ax, anterior_is_left=False, corner="lower right")

    handles = [
        Line2D(
            [0],
            [0],
            color=FUNCTION_COLORS[cluster],
            linewidth=0.9,
            marker="o",
            markersize=3.5,
            label=DISPLAY_LABELS[cluster],
        )
        for cluster in CLUSTERS
    ]
    axes[0, 0].legend(
        handles=handles,
        frameon=False,
        fontsize=fontsize,
        loc="upper left",
    )
    fig.subplots_adjust(wspace=0.02, hspace=0.06, left=0.14, right=0.98, top=0.94, bottom=0.02)
    out = save_svg(fig, out_svg, close=True)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Write channel tables without the 3D render.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = img_dir("picture_naming_modality")
    assignments = load_assignments()
    sets = channel_sets(assignments)
    summary = summarize_sets(assignments, sets)
    members = membership_table(assignments, sets)
    summary_path = out_dir / "pn_repeat_image_vs_motor_spatial_counts.csv"
    members_path = out_dir / "pn_repeat_image_vs_motor_spatial_channels.csv"
    summary.to_csv(summary_path, index=False)
    members.to_csv(members_path, index=False)
    print(summary.to_string(index=False))
    print("wrote", summary_path)
    print("wrote", members_path)
    if args.tables_only:
        return
    from matplotlib import font_manager

    for ttf in (Path(sys.prefix) / "fonts").glob("arial*.ttf"):
        font_manager.fontManager.addfont(str(ttf))
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    svg = plot_spatial(
        assignments,
        sets,
        out_dir / "pn_repeat_image_vs_motor_spatial.svg",
    )
    print("wrote", svg)


if __name__ == "__main__":
    main()
