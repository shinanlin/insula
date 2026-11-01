"""Selected-site maps: Figure 1 surface and exact Figure 2 display contours."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from src.nmf.waveform_analysis import FUNCTION_COLORS
from src.supplementary.decision_repeat_maps import (
    _figure1_local_insula_renderer, CLUSTERS, LABELS,
)


def _figure2_contour_helpers(root):
    path = Path(root) / 'vizpub/fig2.ipynb'
    notebook = json.loads(path.read_text())
    source = ''.join(next(c for c in notebook['cells']
                          if c.get('id') == 'fig2-spatial-plot')['source'])
    tree = ast.parse(source)
    names = {'_keep_ring', 'kde_contour_rings', 'draw_rings', 'crop_to_content'}
    constant_names = {'KDE_BANDWIDTH_PX', 'KDE_LEVEL_FRAC', 'KDE_FILL_ALPHA',
                      'KDE_MIN_POINTS', 'CROP_PAD_PX'}
    constants = {}
    functions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            functions.append(node)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in constant_names:
                constants[target.id] = ast.literal_eval(node.value)
    assert {f.name for f in functions} == names and set(constants) == constant_names
    namespace = dict(np=np, plt=plt, MplPath=MplPath, gaussian_filter=gaussian_filter, **constants)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    proof = dict(source_notebook=str(path), source_cell='fig2-spatial-plot',
        source_cell_sha256=hashlib.sha256(source.encode()).hexdigest(),
        exact_function_sha256={n.name: hashlib.sha256(ast.get_source_segment(source, n).encode()).hexdigest()
                               for n in functions}, constants=constants,
        copied_verbatim=True, source_notebook_executed=False,
        minimum_sites_per_component_hemisphere=3,
        contour_interpretation='Descriptive spatial density; no statistical inference')
    return namespace, proof


def draw_selected_brain_contours(points, title, *, display_output=None):
    root = Path(__file__).resolve().parents[2]
    renderer, proof = _figure1_local_insula_renderer(root)
    contours, contour_proof = _figure2_contour_helpers(root)
    selected = points.loc[points.significant].copy()
    assert not selected.empty and selected.electrode.is_unique
    assert selected.functional_cluster.isin(CLUSTERS).all()
    audit_rows, contour_rows, ring_rows = [], [], []
    cm = 1 / 2.54
    fig, axes = plt.subplots(1, 2, figsize=(8.6 * cm, 4.7 * cm))
    fig.subplots_adjust(left=.03, right=.98, bottom=.26, top=.80, wspace=.03)
    fig.suptitle(title, fontsize=7.5, y=.98)
    for ax, hemi, hemi_title, side in zip(axes, ('lh', 'rh'), ('Left insula', 'Right insula'),
                                         (selected.x.lt(0), selected.x.gt(0))):
        frame = selected.loc[side].copy()
        xyz = frame[['x', 'y', 'z']].to_numpy(float)
        colors = np.asarray([FUNCTION_COLORS[c] for c in frame.functional_cluster], dtype=object)
        sizes = np.full(len(frame), 24.)
        assert not any(color == renderer['NSIG_COLOR'] for color in colors)
        brain = None
        try:
            image, project, brain = renderer['_render_insula_hemi'](hemi, xyz, colors, sizes)
            assert image.shape[:2] == (800, 800), 'Preserve the original 800-pixel KDE scale'
            if len(frame):
                camera = renderer['_camera_xyz'](brain)
                surface, pial = renderer[f'{hemi}_surf'], renderer[f'{hemi}_pial_coords']
                displayed = renderer['_project_to_insula'](xyz, surface, pial, camera)
                toward_camera = camera - displayed
                toward_camera /= np.linalg.norm(toward_camera, axis=1, keepdims=True)
                snapped = displayed - renderer['ELECTRODE_OFFSET_MM'] * toward_camera
                residual, vertex = renderer['cKDTree'](pial[surface['verts']]).query(snapped)
                assert np.max(residual) < 1e-8
                np.testing.assert_allclose(np.linalg.norm(displayed - snapped, axis=1), 4., rtol=0, atol=1e-10)
                pixels = project(displayed)
                for i, row in enumerate(frame.itertuples()):
                    audit_rows.append(dict(electrode=row.electrode, subject=row.subject, channel=row.channel,
                        hemisphere=hemi, selected=True, functional_cluster=row.functional_cluster,
                        x_original=float(xyz[i, 0]), y_original=float(xyz[i, 1]), z_original=float(xyz[i, 2]),
                        x_surface=float(snapped[i, 0]), y_surface=float(snapped[i, 1]), z_surface=float(snapped[i, 2]),
                        x_display=float(displayed[i, 0]), y_display=float(displayed[i, 1]), z_display=float(displayed[i, 2]),
                        insula_vertex=int(surface['verts'][vertex[i]]), surface_residual_mm=float(residual[i]),
                        display_lift_mm=4., pixel_x=float(pixels[i, 0]), pixel_y=float(pixels[i, 1])))
            else:
                pixels = np.empty((0, 2))
        finally:
            if brain is not None:
                brain.close()
        image, pixels = contours['crop_to_content'](image, pixels)
        ax.imshow(image)
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.axis('off')
        ax.set_title(f'{hemi_title} · n={len(frame)}', fontsize=7, pad=2)
        in_frame = ((pixels[:, 0] >= 0) & (pixels[:, 0] < image.shape[1])
                    & (pixels[:, 1] >= 0) & (pixels[:, 1] < image.shape[0]))
        assert in_frame.all(), 'Every selected point must remain in frame'
        for cluster in CLUSTERS:
            group = pixels[in_frame & frame.functional_cluster.eq(cluster).to_numpy()]
            rings = contours['kde_contour_rings'](group, image.shape[:2])
            contours['draw_rings'](ax, rings, FUNCTION_COLORS[cluster])
            contour_rows.append(dict(hemisphere=hemi, functional_cluster=cluster,
                n_selected=len(group), n_rings=len(rings),
                sparse_group=len(group) < 3, sparse_group_rule='Original Figure 2: retain points; no contour below 3 sites'))
            for ri, ring in enumerate(rings):
                ring_rows.extend(dict(hemisphere=hemi, functional_cluster=cluster, ring=ri,
                                      vertex=vi, pixel_x=float(x), pixel_y=float(y))
                                 for vi, (x, y) in enumerate(ring))
    # Bring both hemispheres inward without changing their physical scale.
    fig.canvas.draw()
    for ax, shift in zip(axes, (.025, -.025)):
        position = ax.get_position()
        ax.set_position([position.x0 + shift, position.y0, position.width, position.height])
    handles = [Line2D([0], [0], color=FUNCTION_COLORS[c], linewidth=.9,
                     label=f'{LABELS[c]} ({int(selected.functional_cluster.eq(c).sum())})')
               for c in CLUSTERS]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .095), fontsize=6,
               frameon=False, ncol=3, handletextpad=.25, columnspacing=.65)
    fig.text(.5, .018, 'Selected sites only · Contours show spatial density',
             ha='center', va='bottom', fontsize=5.9)
    assert set(r['electrode'] for r in audit_rows) == set(selected.electrode)
    assert len(audit_rows) == len(selected)
    if display_output is not None:
        destination = Path(display_output)
        destination.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(audit_rows).to_csv(destination / 'display_projection_audit.csv', index=False)
        pd.DataFrame(contour_rows).to_csv(destination / 'contour_component_counts.csv', index=False)
        pd.DataFrame(ring_rows, columns=['hemisphere', 'functional_cluster', 'ring', 'vertex', 'pixel_x', 'pixel_y']).to_csv(
            destination / 'contour_vertices.csv', index=False)
        proof.update(n_sites=len(points), n_selected=len(selected), n_rendered_points=len(selected),
                     n_rendered_nonsignificant_points=0, all_base_points_on_insula_surface=True,
                     display_marker_sizes_px=dict(selected=24.),
                     maximum_surface_residual_mm=max(r['surface_residual_mm'] for r in audit_rows),
                     contour_method=contour_proof, selection_changed=False, statistical_results_changed=False,
                     hemisphere_inward_shift_figure_fraction=.025,
                     total_interhemisphere_gap_reduction_cm=.43,
                     hemisphere_image_sizes_preserved=True)
        (destination / 'figure1_renderer_provenance.json').write_text(json.dumps(proof, indent=2) + '\n')
    return fig
