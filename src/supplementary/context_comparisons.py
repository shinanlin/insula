"""Native inputs and condition-wise brain maps for the separate Figure 3 SI.

The waveform plotting definitions are copied verbatim into the notebook.
This module supplies validated native data, the retained nominal Passive
Response, and descriptive brain maps of one explicitly labeled phase.
"""
from __future__ import annotations

import ast
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.figure3_context_draft import _trial_mean, _names, digest, write_json
from src.nmf.waveform_analysis import FUNCTION_COLORS

ROOT = Path(__file__).resolve().parents[2]
CLUSTERS = ("sensory", "sustain", "motor")
PHASES = ("stimulus", "delay", "response")
CONTRASTS = (
    dict(key="sound_image", task="PictureNaming", title="Sound–Image · Picture Naming, Repeat",
         field="modality", levels=("sound", "image"), source_spec="input", map_phase="stimulus"),
    dict(key="decision_repeat", task="LexicalDelay", title="Decision–Repeat · Lexical Delay, sound",
         field="description", levels=("Decision", "Repeat"), source_spec="delay", map_phase="delay"),
    dict(key="repeat_passive", task="PictureNaming", title="Repeat–Passive · Picture Naming, sound",
         field="description", levels=("Repeat", "Passive"), source_spec="output", map_phase="stimulus"),
)
DISPLAY_WINDOWS = dict(stimulus=(-.5, .6), delay=(0., 1.), response=(-.5, 1.))


def _nominal_passive_response(subject):
    """Reuse the existing original alignment; never synthesize a vocal onset."""
    base = Path("/cwork/ns458/BIDS-1.3_PictureNaming/BIDS/derivatives")
    filename = f"sub-{subject}_task-PictureNaming_proc-Response_recording-sound_desc-Passive_highgamma.h5"
    epoch = base / "epoch(bipolar)" / f"sub-{subject}" / "epoch(band)(zscore)" / filename
    stats = base / "statistics" / f"sub-{subject}" / "bipolar" / filename
    if not epoch.exists() or not stats.exists():
        return None
    ep = mne.read_epochs(epoch, preload=False, verbose="error")
    trials = len(ep.events)
    del ep
    return dict(condition="Passive", modality="sound", phase="Response", trials=trials,
                zscore_path=str(epoch), statistics_path=str(stats),
                alignment="retained original nominal Passive Response; no actual speech onset")


def load_context_data(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "epoch_means"
    cache.mkdir(exist_ok=True)
    assignments = pd.read_csv(ROOT / "results/nmf/channel_assignments.csv")
    assert len(assignments) == 231 and not assignments.duplicated(["subject", "channel"]).any()
    coverage = pd.read_csv(ROOT / "results/paper_trial_inputs/subject_task_coverage.csv")
    inputs, missing, frames, alignment = [], [], [], []
    for task in ("PictureNaming", "LexicalDelay"):
        for subject, group in assignments.groupby("subject", sort=True):
            status = coverage.loc[coverage.task.eq(task) & coverage.subject.eq(subject)]
            if status.empty or subject == "D0121":
                continue
            path = ROOT / "results/native_preprocessing_restore_2026-10-18" / task / subject / "output_validation.json"
            if not path.exists():
                assert status.iloc[0].status == "native_QC_zero_usable"
                missing.append(dict(task=task, subject=subject, reason="native QC: no usable task sites"))
                continue
            validation = json.loads(path.read_text())
            assert validation["validation_passed"]
            products = validation["products"]
            sig_union, native_channels = set(), None
            for product in products:
                if product["modality"] != "sound":
                    continue
                with h5py.File(product["statistics_path"], "r") as stats:
                    sig_union.update(_names(stats["sig_ch_names"]))
                    names = set(_names(stats["ch_names"]))
                    native_channels = names if native_channels is None else native_channels & names
            # Restrict a legacy nominal Response to the restored good montage
            # and existing task electrode set, as in the original PN splice.
            assigned = set(group.channel)
            context = assigned & sig_union & native_channels
            if task == "PictureNaming" and context:
                legacy = _nominal_passive_response(subject)
                if legacy is not None:
                    products = products + [legacy]
                    alignment.append(dict(subject=subject, **legacy))
                else:
                    missing.append(dict(task=task, subject=subject, phase="response", condition="Passive",
                                        reason="retained nominal Passive Response epoch/statistics unavailable"))
            for spec in CONTRASTS:
                if spec["task"] != task:
                    continue
                requested = assigned & native_channels if spec["field"] == "modality" else context
                if not requested:
                    continue
                for phase in PHASES:
                    bounds = DISPLAY_WINDOWS[phase]
                    if spec["key"] == "decision_repeat" and phase == "delay":
                        bounds = (-.05, 1.)  # Original Figure 3 primary delay display.
                    for level in spec["levels"]:
                        condition = "Repeat" if spec["field"] == "modality" else level
                        modality = level if spec["field"] == "modality" else "sound"
                        matches = [p for p in products if (p["condition"], p["modality"], p["phase"].lower()) == (condition, modality, phase)]
                        if not matches:
                            assert condition == "Passive" and phase == "response"
                            continue
                        assert len(matches) == 1
                        product = matches[0]
                        print(task, subject, spec["key"], phase, level, "sites", len(requested), flush=True)
                        channels, times, mean, mask, n_trials = _trial_mean(product, requested, cache, inputs, missing)
                        keep = (times >= bounds[0]) & (times <= bounds[1])
                        if not channels:
                            continue
                        nt = int(keep.sum())
                        frame = pd.DataFrame(dict(channel=np.repeat(channels, nt), time=np.tile(times[keep], len(channels)),
                            value=mean[:, keep].reshape(-1), mask=mask[:, keep].reshape(-1)))
                        frame["subject"], frame["task"], frame["phase"] = subject, task, phase
                        frame["contrast"], frame["level"] = spec["key"], level
                        frame["description"], frame["modality"] = condition, modality
                        frame["n_trials"] = n_trials
                        frame["functional_cluster"] = frame.channel.map(group.set_index("channel").functional_cluster)
                        frame["electrode"] = subject + "|" + frame.channel
                        frame["mask_observed"] = True
                        frame["alignment"] = product.get("alignment", "validated restored native alignment")
                        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    assert data.functional_cluster.isin(CLUSTERS).all()
    data.to_pickle(output / "waveforms.pkl")
    data.to_csv(output / "waveforms.csv.gz", index=False)
    membership = data[["contrast", "task", "subject", "channel", "functional_cluster", "phase", "level", "n_trials", "alignment"]].drop_duplicates()
    membership.to_csv(output / "waveform_membership.csv", index=False)
    counts = membership.groupby(["contrast", "phase", "functional_cluster", "level"]).agg(
        n_sites=("channel", "nunique"), n_subjects=("subject", "nunique")).reset_index()
    counts.to_csv(output / "waveform_counts.csv", index=False)
    pd.DataFrame(missing).to_csv(output / "missing_inputs.csv", index=False)
    write_json(output / "native_input_manifest.json", dict(
        assignments_sha256=digest(ROOT / "results/nmf/channel_assignments.csv"),
        input_files=inputs, retained_nominal_passive_response=alignment,
        condition_tests="No new condition test; existing baseline permutation masks only",
        native_QC="Original restored good channels; nominal Passive Response cannot reintroduce old bad channels",
        map_scope="One approved displayed phase per contrast; saved baseline masks, not a between-condition difference",
        map_phases={spec['key']:spec['map_phase'] for spec in CONTRASTS},
        display_windows=DISPLAY_WINDOWS, original_primary_LD_delay_window=[-.05, 1.]))
    return data, assignments, counts


def condition_phase_maps(data, assignments, spec, output, xlims):
    frame = data.loc[data.contrast.eq(spec["key"])]
    phase = spec["map_phase"]
    bounds = xlims[phase]
    mapped = frame.loc[frame.phase.eq(phase) & frame.time.between(*bounds)]
    assert not mapped.empty
    keys = frame[["subject", "channel"]].drop_duplicates()
    background = keys.merge(assignments, on=["subject", "channel"], how="left", validate="one_to_one")
    assert background.functional_cluster.notna().all()
    rows = []
    for level in spec["levels"]:
        points = background.copy()
        active = mapped.loc[mapped["level"].eq(level) & mapped["mask"], ["subject", "channel"]].drop_duplicates()
        hits = set(map(tuple, active.to_numpy()))
        available = set(map(tuple, mapped.loc[mapped['level'].eq(level), ['subject','channel']].drop_duplicates().to_numpy()))
        points['phase_observed'] = [(s,c) in available for s,c in zip(points.subject,points.channel)]
        points["significant"] = [(s, c) in hits for s, c in zip(points.subject, points.channel)]
        assert not (points.significant & ~points.phase_observed).any()
        points["pattern"] = np.where(points.significant, 1., np.nan)
        points["condition"] = level
        points["contrast"] = spec["key"]
        points['map_phase'] = phase
        points['map_window_start'] = bounds[0]
        points['map_window_end'] = bounds[1]
        rows.append(points)
        # Direct agreement between map membership and the masks in the source
        # waveform table for the selected phase, before rendering/projection.
        assert set(map(tuple, points.loc[points.significant, ["subject", "channel"]].to_numpy())) == hits
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(Path(output) / f"{spec['key']}_brain_membership.csv", index=False)
    return table


def _figure1_source_cells():
    notebook = json.loads((ROOT / "vizpub/fig1.ipynb").read_text())
    return {cell.get("id"): "".join(cell["source"]) for cell in notebook["cells"]}


def _exec_functions(namespace, source, names):
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    compiled = ast.Module(body=nodes, type_ignores=[])
    exec(compile(compiled, "vizpub/fig1.ipynb", "exec"), namespace)
    return {node.name: hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest() for node in nodes}


def _exec_assignments(namespace, source, names):
    tree = ast.parse(source)
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in names:
                nodes.append(node)
    assert {node.targets[0].id for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "vizpub/fig1.ipynb", "exec"), namespace)


@lru_cache(maxsize=1)
def _figure1_brain_panel():
    """Figure 1 panel B helpers, without executing Figure 1 analysis cells.

    The saved panel draws the insula mesh alone, projects each electrode's
    own x/y/z onto that screenshot, and overlays Hammers gyrus boundaries.
    """
    import mne
    import nibabel as nib
    import pyvista as pv
    from collections import defaultdict
    from mne.viz import Brain
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial import cKDTree

    cells = _figure1_source_cells()
    pv.OFF_SCREEN = True
    mne.viz.set_3d_backend("notebook", verbose="error")
    namespace = dict(np=np, pv=pv, nib=nib, Brain=Brain, cKDTree=cKDTree,
                     defaultdict=defaultdict, gaussian_filter1d=gaussian_filter1d,
                     Path=Path, SOURCE_ROOT=Path("/hpc/group/coganlab/nanlinshi/insula-functional"))
    _exec_assignments(namespace, cells["fig1-style"], {"recon_dir"})
    _exec_assignments(namespace, cells["00031585"], {
        "FS_SUBJECT", "HAMMERS_MAXPROB_NAME", "HAMMERS_MNI152_CANDIDATES",
        "HAMMERS_MAXPROB_TAR", "HAMMERS_INSULA_INDEX", "HAMMERS_INSULA_IDS"})
    _exec_assignments(namespace, cells["7731cdbd"], {
        "INSULA_MESH_COLOR", "INSULA_MESH_ALPHA", "ELECTRODE_OFFSET_MM",
        "HAMMERS_FILL_MAX_MM", "POINT_SIZE_NSIG", "POINT_SIZE_SIG", "NSIG_COLOR",
        "HAMMERS_GYRI", "INSULA_PATTERNS"})
    hashes = {}
    hashes.update(_exec_functions(namespace, cells["00031585"],
        ["resolve_hammers_mni152", "_cvs_tkr_to_mni_ras", "_sample_hammers_labels"]))
    hashes.update(_exec_functions(namespace, cells["7731cdbd"], [
        "_insula_vertices", "_faces_to_polydata", "_insula_outward_normals",
        "_prepare_insula_surface", "_hide_default_pial", "_make_insula_brain",
        "_camera_xyz", "_points_to_pixels", "_hammers_gyrus_code", "_xy_arclength",
        "_pial_hammers_gyrus", "_render_insula_hemi"]))
    hashes.update(_exec_functions(namespace, cells["72a702a6"], [
        "_trace_boundary_network", "_smooth_boundary",
        "overlay_continuous_gyrus_boundaries", "_crop_panel"]))
    namespace["HAMMERS_MNI152_PATH"] = namespace["resolve_hammers_mni152"]()
    namespace["labels"] = mne.read_labels_from_annot(
        subject=namespace["FS_SUBJECT"], parc="aparc.a2009s", surf_name="pial",
        hemi="both", subjects_dir=namespace["recon_dir"])
    for hemi in ("lh", "rh"):
        pial, faces = mne.read_surface(
            f"{namespace['recon_dir']}/{namespace['FS_SUBJECT']}/surf/{hemi}.pial")
        namespace[f"{hemi}_pial_coords"] = pial
        namespace[f"{hemi}_faces"] = faces
        surface = namespace["_prepare_insula_surface"](hemi, pial, faces)
        namespace[f"{hemi}_surf"] = surface
        namespace[f"{hemi}_insula"] = surface["mesh"]
        namespace[f"{hemi}_insula_center"] = surface["center"]
        namespace[f"{hemi}_hammers_gyrus"] = namespace["_pial_hammers_gyrus"](pial, hemi)
    namespace["function_sha256"] = hashes
    namespace["source_cells"] = ["00031585", "7731cdbd", "72a702a6"]
    return namespace


def fit_hga_canvas(fig, *, width_cm=13.0, max_height_cm=8.0, pad_inches=0.1):
    """Scale a finished HGA stack so the saved SVG is ``width_cm`` wide.

    Phase ratios, fonts and line widths stay as drawn. Height is reduced only when
    ``max_height_cm`` is set and the saved canvas would otherwise exceed it.
    ``max_height_cm=None`` leaves the drawn height unchanged.
    """
    target_w = width_cm / 2.54
    max_h = None if max_height_cm is None else max_height_cm / 2.54
    width_samples = []
    height_samples = []
    for _ in range(8):
        fig.canvas.draw()
        box = fig.get_tightbbox(fig.canvas.get_renderer()).padded(pad_inches)
        saved_w, saved_h = float(box.width), float(box.height)
        fig_w, fig_h = (float(value) for value in fig.get_size_inches())
        height_ok = max_h is None or saved_h <= max_h + 0.015
        if abs(saved_w - target_w) <= 0.015 and height_ok:
            print(f"HGA canvas {saved_w * 2.54:.2f} x {saved_h * 2.54:.2f} cm", flush=True)
            return fig
        width_samples.append((fig_w, saved_w))
        height_samples.append((fig_h, saved_h))
        new_w = _canvas_size_for_target(width_samples, target_w, fig_w * target_w / saved_w)
        if max_h is not None and saved_h > max_h:
            new_h = _canvas_size_for_target(height_samples, max_h, fig_h * max_h / saved_h)
        else:
            new_h = fig_h
        fig.set_size_inches(max(new_w, 0.5), max(new_h, 0.5), forward=True)
    fig.canvas.draw()
    box = fig.get_tightbbox(fig.canvas.get_renderer()).padded(pad_inches)
    saved_w, saved_h = float(box.width), float(box.height)
    if abs(saved_w - target_w) > 0.05 or (max_h is not None and saved_h > max_h + 0.05):
        raise RuntimeError(f"HGA canvas fit stopped at {saved_w * 2.54:.2f} x {saved_h * 2.54:.2f} cm")
    print(f"HGA canvas {saved_w * 2.54:.2f} x {saved_h * 2.54:.2f} cm", flush=True)
    return fig


def _canvas_size_for_target(samples, target, fallback):
    """Invert an approximately affine saved-size response from the last two tries."""
    if len(samples) < 2:
        return fallback
    (size_a, saved_a), (size_b, saved_b) = samples[-2], samples[-1]
    slope = (saved_b - saved_a) / (size_b - size_a) if size_b != size_a else 0.0
    if abs(slope) < 1e-4:
        return fallback
    return size_b + (target - saved_b) / slope


def draw_brain_maps(table, spec, destination):
    """One horizontal row of Figure 1 insula panels: mesh, true coordinates, gyrus lines."""
    panel = _figure1_brain_panel()
    coordinates = table[["x", "y", "z"]].to_numpy(float)
    assert np.isfinite(coordinates).all()
    assert not np.any(coordinates[:, 0] == 0)
    phase = spec["map_phase"]
    start, end = table[["map_window_start", "map_window_end"]].drop_duplicates().iloc[0]
    empty_xyz, empty_col = np.zeros((0, 3)), np.array([], dtype=object)
    empty_sz = np.array([], dtype=float)
    rendered = []
    for level in spec["levels"]:
        points = table.loc[table.condition.eq(level)].copy()
        n = int(points.significant.sum())
        counts = points.loc[points.significant].functional_cluster.value_counts()
        for hemi, side in (("lh", "Left"), ("rh", "Right")):
            frame = points.loc[points.x.lt(0) if hemi == "lh" else points.x.gt(0)]
            brain = None
            try:
                image, project, brain = panel["_render_insula_hemi"](hemi, empty_xyz, empty_col, empty_sz)
                overlay = panel["overlay_continuous_gyrus_boundaries"](hemi, project)
                xyz = frame[["x", "y", "z"]].to_numpy(float)
                pixels = project(xyz) if len(xyz) else np.zeros((0, 2))
            finally:
                if brain is not None:
                    brain.close()
            image, pixels, overlay = panel["_crop_panel"](image, pixels, overlay, pad=24)
            rendered.append(dict(level=level, side=side, frame=frame, image=image,
                                 pixels=pixels, overlay=overlay, n=n, counts=counts))
    cm = 1 / 2.54
    panel_w_cm = 3.5
    gap_cm, side_cm, top_cm, bot_cm = 0.22, 0.08, 1.05, 1.45
    heights = [panel_w_cm * item["image"].shape[0] / item["image"].shape[1] for item in rendered]
    panel_h_cm = max(heights)
    width_cm = side_cm * 2 + panel_w_cm * len(rendered) + gap_cm * (len(rendered) - 1)
    height_cm = top_cm + panel_h_cm + bot_cm
    fig = plt.figure(figsize=(width_cm * cm, height_cm * cm))
    fig.suptitle(f"{phase.capitalize()} window: {start:g} to {end:g} s", y=.985, fontsize=7.5, fontweight="bold")
    for index, item in enumerate(rendered):
        image = item["image"]
        left = (side_cm + index * (panel_w_cm + gap_cm)) / width_cm
        bottom = (bot_cm + panel_h_cm - heights[index]) / height_cm
        ax = fig.add_axes([left, bottom, panel_w_cm / width_cm, heights[index] / height_cm])
        ax.imshow(image, aspect="auto", interpolation="nearest")
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.axis("off")
        ax.set_title(f"{item['level'].capitalize()} · {item['side']} Insula", fontsize=6.5, pad=2)
        pixels = item["pixels"]
        significant = item["frame"].significant.to_numpy(dtype=bool)
        clusters = item["frame"].functional_cluster.to_numpy()
        for cluster in CLUSTERS:
            chosen = significant & (clusters == cluster)
            if chosen.any():
                ax.scatter(pixels[chosen, 0], pixels[chosen, 1], s=6,
                           c=FUNCTION_COLORS[cluster], edgecolors="none", zorder=6)
        for line in item["overlay"].get("lines_minor", []):
            if len(line) < 2:
                continue
            ax.plot(line[:, 0], line[:, 1], linestyle=(0, (2.4, 1.8)), color="0.40",
                    lw=0.65, solid_capstyle="round", dash_capstyle="butt", zorder=5)
        if index % 2 == 0:
            label = f"{phase.capitalize()}: n={item['n']}"
        else:
            label = "S / Sust / M: " + " / ".join(str(int(item["counts"].get(c, 0))) for c in CLUSTERS)
        ax.text(0., -0.07, label, transform=ax.transAxes, fontsize=6.0, clip_on=False)
    fig.canvas.draw()
    image_widths = [ax.get_window_extent().width / fig.dpi * 2.54 for ax in fig.axes]
    if any(abs(width - panel_w_cm) > 0.15 for width in image_widths):
        raise RuntimeError(f"insula screenshot widths {image_widths} cm, expected {panel_w_cm}")
    print("insula screenshot widths cm", [round(width, 2) for width in image_widths], flush=True)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=FUNCTION_COLORS[c],
                      markeredgewidth=0, markersize=3.2,
                      label={"sensory": "sensory", "sustain": "sustained", "motor": "motor"}[c])
               for c in CLUSTERS]
    handles.append(Line2D([0], [0], color="0.40", lw=0.65, linestyle=(0, (2.4, 1.8)), label="Gyrus boundary"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .062), fontsize=6.5,
               frameon=False, ncol=4, handletextpad=.25, columnspacing=.7)
    fig.text(.02, .012, f"Points: baseline-responsive in the {phase} window; no between-condition test", fontsize=5.9)
    fig.savefig(destination, format="svg", dpi=300, bbox_inches="tight")
    proof = ROOT / "results/supplementary/context_comparisons/figure1_brain_renderer_provenance.json"
    proof.write_text(json.dumps(dict(
        source_notebook=str(ROOT / "vizpub/fig1.ipynb"),
        source_cells=panel["source_cells"],
        function_sha256=panel["function_sha256"],
        display="Figure 1 panel B: insula-only mesh, true x/y/z scatter, crop pad 24, gyrus boundaries",
        layout="1x4 horizontal; each unilateral insula image is 3.5 cm wide",
        marker_area=6, nonsignificant_points_drawn=False,
        component_colors="frozen NMF palette for baseline-responsive sites",
        analytic_coordinates_changed=False, new_condition_tests=False), indent=2) + "\n")
    return fig


def highlight_mapped_phase(fig, axes, spec):
    """Label the mapped phase without enclosing the plots in a border."""
    column = PHASES.index(spec['map_phase'])
    axes[0, column].set_title(f"{spec['map_phase'].capitalize()}\n(brain maps)", fontsize=7, fontweight='bold')
