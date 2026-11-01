"""Reviewed local-insula functional maps, without obsolete analysis entry points."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd
from src.supplementary.decision_repeat_maps import _figure1_local_insula_renderer
from src.supplementary.selected_brain_contours import _figure2_contour_helpers
ROOT=Path(__file__).resolve().parents[1]

def brain_pair(points, groups, colors, title):
    assert points.electrode.is_unique and set(points.display_group) <= set(groups)
    cm = 1 / 2.54
    fig, axes = plt.subplots(1, 2, figsize=(8.6 * cm, 4.7 * cm))
    fig.subplots_adjust(left=.03, right=.98, bottom=.16, top=.80, wspace=.03)
    fig.suptitle(title, fontsize=7.5, y=.98)
    audit = []
    for ax, hemi, hemi_label in zip(axes, ('lh', 'rh'), ('Left insula', 'Right insula')):
        side = points.x.lt(0) if hemi == 'lh' else points.x.gt(0)
        frame = points.loc[side]
        xyz = frame[['x', 'y', 'z']].to_numpy(float)
        palette = np.asarray([colors[group] for group in frame.display_group], dtype=object)
        brain = None
        try:
            image, project, brain = renderer['_render_insula_hemi'](hemi, xyz, palette, np.full(len(frame), 24.))
            if len(frame):
                camera = renderer['_camera_xyz'](brain)
                surface = renderer[f'{hemi}_surf']
                pial = renderer[f'{hemi}_pial_coords']
                displayed = renderer['_project_to_insula'](xyz, surface, pial, camera)
                direction = camera - displayed
                direction /= np.linalg.norm(direction, axis=1, keepdims=True)
                snapped = displayed - 4. * direction
                residual, _vertices = renderer['cKDTree'](pial[surface['verts']]).query(snapped)
                assert np.max(residual) < 1e-8
                pixels = project(displayed)
                for i, row in enumerate(frame.itertuples()):
                    audit.append(dict(
                        electrode=row.electrode, subject=row.subject, hemisphere=hemi, display_group=row.display_group,
                        pixel_x=float(pixels[i, 0]), pixel_y=float(pixels[i, 1]),
                        surface_residual_mm=float(residual[i]), display_lift_mm=4.))
            else:
                pixels = np.empty((0, 2))
        finally:
            if brain is not None:
                brain.close()
        image, pixels = contour['crop_to_content'](image, pixels)
        ax.imshow(image)
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.axis('off')
        ax.set_title(f'{hemi_label} · n={len(frame)}', fontsize=7, pad=2)
    fig.canvas.draw()
    for ax, shift in zip(axes, (.025, -.025)):
        position = ax.get_position()
        ax.set_position([position.x0 + shift, position.y0, position.width, position.height])
    handles = [Line2D([0], [0], marker='o', linestyle='none', markerfacecolor=colors[group],
                       markeredgecolor=colors[group], markersize=4,
                       label=f'{group} (n={int(points.display_group.eq(group).sum())})') for group in groups]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .02), fontsize=6,
               frameon=False, ncol=len(groups), handletextpad=.3, columnspacing=.7)
    assert {row['electrode'] for row in audit} == set(points.electrode)
    return fig, audit

def renderer_namespace(root=ROOT):
    global renderer,renderer_proof,contour,contour_proof
    assert Path(root).resolve()==ROOT.resolve()
    renderer,renderer_proof=_figure1_local_insula_renderer(ROOT)
    contour,contour_proof=_figure2_contour_helpers(ROOT)
    assert renderer_proof['functions_copied_verbatim'] and contour_proof['copied_verbatim']
    return dict(brain_pair=brain_pair,renderer_proof=renderer_proof,contour_proof=contour_proof)
