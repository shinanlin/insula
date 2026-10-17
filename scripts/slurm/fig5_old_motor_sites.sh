#!/bin/bash
#SBATCH --job-name=fig5_oldmot
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/fig5_old_motor_sites_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/fig5_old_motor_sites_%j.err
#SBATCH --time=00:40:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional/vizpub

set -eo pipefail

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

export PYTHONPATH="/hpc/group/coganlab/nanlinshi/insula-functional${PYTHONPATH:+:${PYTHONPATH}}"
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_PANEL=false
export MNE_3D_BACKEND=notebook
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1

mkdir -p /hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm \
  /hpc/group/coganlab/nanlinshi/insula-functional/img/fig5

python - <<'PY'
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib.lines import Line2D

nb = json.loads(Path("fig5.ipynb").read_text())
cells = {c.get("id"): c for c in nb["cells"]}
ns = {"__name__": "__main__"}
for cid in ("fig5-imports", "fig5-style", "fig5-load"):
    src = "\n".join(
        line
        for line in "".join(cells[cid]["source"]).splitlines()
        if not line.startswith("%")
    )
    exec(src, ns)

spatial = "\n".join(
    line
    for line in "".join(cells["fig5-spatial"]["source"]).splitlines()
    if not line.startswith("%")
)
spatial = spatial.split("hit_unique = ", 1)[0]
exec(spatial, ns)

HIGHLIGHT = {
    "D0084_RAI2-3",
    "D0084_RAI3-4",
    "D0084_RAI4-5",
    "D0090_RIA4-5",
    "D0102_RFAI1-2",
    "D0119_RIA4-5",
}

summary = Path("..") / "results" / "rt" / "summaries"
electrodes = pd.read_csv(summary / "electrodes.csv")
clusters = pd.read_csv(summary / "significant_clusters.csv")
hit = clusters.loc[
    ((clusters.phase == "Delay") & (clusters.end_time > 0))
    | (clusters.phase.isin(["Go", "Response"]) & (clusters.start_time < 0)),
    ["subject", "channel"],
].drop_duplicates()
elec = electrodes.drop_duplicates(["subject", "channel"])
spatial_df = elec.merge(hit, on=["subject", "channel"])
spatial_df = spatial_df.loc[
    spatial_df.functional_cluster.isin(ns["RASTER_CLUSTER_ORDER"])
    & spatial_df[["x_template", "y_template", "z_template"]].notna().all(axis=1)
].copy()
spatial_df["highlight"] = spatial_df.channel.isin(HIGHLIGHT)
print(
    "old assigned",
    spatial_df.functional_cluster.value_counts().to_dict(),
    "highlight",
    int(spatial_df.highlight.sum()),
    "highlight hemi",
    spatial_df.loc[spatial_df.highlight, "hemi"].value_counts().to_dict(),
)

def render_marked(hemi, frame):
    brain_fn = ns["Brain"]
    template = ns["template"]
    recon_dir = ns["recon_dir"]
    brain = brain_fn(
        template,
        subjects_dir=recon_dir,
        surf="pial",
        hemi=hemi,
        background="white",
        show=False,
        cortex=ns["INSULA_COLOR"],
        alpha=0.0,
        size=(800, 800),
    )
    plotter = brain._renderer.plotter
    mesh = ns["lh_insula"] if hemi == "lh" else ns["rh_insula"]
    plotter.add_mesh(
        mesh,
        color=ns["INSULA_COLOR"],
        opacity=ns["INSULA_LABEL_ALPHA"],
        lighting=True,
        smooth_shading=True,
    )
    tree = ns["lh_tree"] if hemi == "lh" else ns["rh_tree"]
    pial = ns["lh_pial_coords"] if hemi == "lh" else ns["rh_pial_coords"]
    side = frame.loc[frame.x_template < 0] if hemi == "lh" else frame.loc[frame.x_template > 0]
    coords = side[["x_template", "y_template", "z_template"]].to_numpy(float)
    ns["add_cluster_spheres"](
        brain, coords, side.functional_cluster.to_numpy(), tree, pial
    )
    marked = side.loc[side.highlight]
    if len(marked):
        snapped = ns["snap_to_pial"](
            marked[["x_template", "y_template", "z_template"]].to_numpy(float),
            tree,
            pial,
        )
        for size, color in ((30, "#111111"), (16, ns["FUNCTION_COLORS"]["motor"])):
            plotter.add_mesh(
                pv.PolyData(np.asarray(snapped, dtype=float)),
                render_points_as_spheres=True,
                point_size=size,
                color=color,
                lighting=False,
            )
    center = ns["lh_insula_center"] if hemi == "lh" else ns["rh_insula_center"]
    az = 180 if hemi == "lh" else 0
    brain.show_view(azimuth=az, elevation=90, distance=180, focalpoint=center)
    for _ in range(3):
        plotter.render()
    img = np.asarray(brain.screenshot(mode="rgb"))
    brain.close()
    img, _ = ns["crop_to_content"](img, np.zeros((0, 2)))
    return img

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    lh_img = render_marked("lh", spatial_df)
    rh_img = render_marked("rh", spatial_df)

plt = ns["plt"]
cm = ns["cm"]
fontsize = ns["fontsize"]
fig, axes = plt.subplots(1, 2, figsize=(6.5 * cm, 3.0 * cm))
for ax, img, title, anterior_left, corner in (
    (axes[0], lh_img, "Left", True, "lower left"),
    (axes[1], rh_img, "Right", False, "lower right"),
):
    ax.imshow(img)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis("off")
    ax.set_title(title, fontsize=fontsize)
    ns["draw_ap_dv"](ax, anterior_is_left=anterior_left, corner=corner)

n_by = spatial_df.groupby("functional_cluster").channel.nunique()
handles = [
    Line2D(
        [0], [0], marker="o", color="none",
        markerfacecolor=ns["FUNCTION_COLORS"][c],
        markeredgecolor=ns["FUNCTION_COLORS"][c],
        markersize=4,
        label=f"{ns['CLUSTER_LABELS'][c]}  (n={int(n_by.get(c, 0))})",
    )
    for c in ns["RASTER_CLUSTER_ORDER"]
]
handles.append(
    Line2D(
        [0], [0], marker="o", color="none",
        markerfacecolor=ns["FUNCTION_COLORS"]["motor"],
        markeredgecolor="#111111",
        markeredgewidth=1.2,
        markersize=5,
        label="old motor, now sustain (n=6)",
    )
)
axes[0].legend(
    handles=handles,
    frameon=False,
    fontsize=fontsize,
    loc="upper right",
    bbox_to_anchor=(-0.04, 1.0),
    borderaxespad=0.0,
)
fig.subplots_adjust(wspace=0.02, left=0.34, right=0.98, top=0.92, bottom=0.02)
out = ns["save_svg"](fig, ns["IMG_DIR"] / "fig5_rt_old_grouping_relabel_sites")
print("Wrote", out)
PY
