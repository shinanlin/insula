"""Three independent PN Gaussian maps using only each site's largest frozen W."""
import os

from src.supplementary.picture_region_point_panels import (
    ROOT, BASE, IMG, SOURCE, GROUPS, DISPLAY, KEYS, COMPONENTS, COLORS, LABELS,
)

OUTPUT = BASE / 'top_weight_gaussian_2026-10-27'
AUDIT = ROOT / 'results/audit/picture_top_weight_gaussian_2026-10-27'
DISPLAY_OUTPUT = BASE / 'top_weight_gaussian_local_2026-10-27'
DISPLAY_AUDIT = ROOT / 'results/audit/picture_top_weight_gaussian_local_2026-10-27'
EXPORTS = {KEYS[g]: f'fig3_picture_{KEYS[g]}_top_weight_gaussian.svg' for g in GROUPS}


def notebook_cells():
    import nbformat as nbf
    return [nbf.v4.new_markdown_cell('''## 8. Three separate maps: each electrode's top original NMF weight

Author-confirmed display: **Sound-only, Both and Image-only each have one independent bilateral Gaussian map**, with no assembly. The native Stimulus [0,1] s memberships remain **25 / 13 / 11** sites. Original frozen NMF W is available for **23 / 12 / 9** sites, respectively; the remaining **2 / 1 / 2** stay explicitly Unassigned and make no invented weight contribution.

For each available electrode, read its original **loading_sensory / loading_sustain / loading_motor** from the unchanged NMF assignments. Retain **only the largest raw value in its original component column**; set its two other display contributions to zero. Example: (0.2, 0.7, 0.1) contributes (0, 0.7, 0), so its Gaussian contribution is gold and scaled by 0.7. Do not divide weights by their sum, use PN condition-specific projections, apply HGA multiplication, refit NMF or overwrite labels. The largest-loading component is a **display encoding**, distinct from the retained functional_cluster column; any discrepancies are exported explicitly.

Gaussian fields use the exact existing reference function: **D_k(v) = sum_i exp(-distance(v,i)^2 / (2 × 5^2)) × Wtop_i,k**, with support restricted to vertices within **6 mm of the nearest available electrode**. At each supported vertex, use the pure Sensory blue / Sustained gold / Motor red of its **largest local field**, without the old proportional RGB mixture. Intensity remains continuous through the original opacity transform: total top-weight field / its **per-hemisphere-map P95**, clipped at one. Thus the maximum loading scales the input contribution, but the displayed opacity is locally normalized and is **not an absolute cross-map weight or HGA amplitude scale**. Different component colors meet at display boundaries; these are not statistical or anatomical parcels.

**Background-only correction:** load the previously saved Gaussian density and RGBA arrays without recomputing or renormalizing them. Use Figure 1's exact local-insula mesh, white background, hidden whole-hemisphere actors and original bilateral cameras. Display the saved RGBA values only on those existing insula faces, indexed by their unchanged original pial vertex IDs; do not project, interpolate, move electrodes or rebuild Gaussian fields. In particular, P95 is **not recomputed after restricting the visible surface**. Peripheral translucent brain structures are removed. The previous numerical data files remain byte-for-byte unchanged; only these three SVGs and this section's display output are refreshed, with the preceding images archived.

The three maps include available-weight/total-site counts and separately state missing weights. All 49 memberships, original coordinates, prior notebook cells/outputs and numerical/statistical results are preserved. No new response test, correction, NMF/NNLS fit, grouping or label assignment is performed. The absolute-count bars from section 7 remain unchanged.
''', id='picture-top-weight-gaussian-note'),
        nbf.v4.new_code_cell('''from pathlib import Path
import sys
from IPython.display import SVG, display
_pn_top_root = Path('/cwork/ns458/insula-paper')
if str(_pn_top_root) not in sys.path:
    sys.path.insert(0, str(_pn_top_root))
from src.supplementary.picture_top_weight_gaussian import render, OUTPUT as top_weight_data
top_weight_paths = render()
for _region in ['sound_only', 'both', 'image_only']:
    display(SVG(filename=str(top_weight_paths[_region])))
print('Three independent bilateral SVGs:', {k: str(v) for k, v in top_weight_paths.items()})
print('Per-electrode top values:', top_weight_data / 'electrode_top_weights.csv')
print('Unassigned sites:', top_weight_data / 'unassigned_sites.csv')
''', id='picture-top-weight-gaussian-display')]


def _prepare():
    import numpy as np
    import pandas as pd
    from scripts.refresh_lexical_condition_sets import sha
    files = [SOURCE / 'electrode_weights.csv', ROOT / 'results/nmf/channel_assignments.csv', BASE / 'electrodes.csv']
    hashes = {str(p): sha(p) for p in files}
    sites = pd.read_csv(files[0], float_precision='round_trip')
    assignments = pd.read_csv(files[1], float_precision='round_trip')
    native = pd.read_csv(files[2])
    assert len(sites) == 49 and sites.electrode.is_unique and sites.has_original_nmf_weight.sum() == 44
    assert set(sites.electrode) == set(native.loc[native.sound_responsive | native.image_responsive, 'electrode'])
    original = assignments.set_index(['subject', 'channel'])
    columns = [f'loading_{c}' for c in COMPONENTS]
    for c in COMPONENTS:
        sites[f'display_top_{c}'] = np.nan
    sites['display_top_component'] = 'Unassigned'
    sites['display_top_weight'] = np.nan
    sites['top_differs_from_frozen_label'] = False
    available = sites.has_original_nmf_weight
    W = sites.loc[available, columns].to_numpy(float)
    assert np.isfinite(W).all() and (W >= 0).all() and (W.sum(axis=1) > 0).all()
    expected = original.loc[list(zip(sites.loc[available, 'subject'], sites.loc[available, 'channel'])), columns].to_numpy(float)
    np.testing.assert_array_equal(W, expected)
    dominant = W.argmax(axis=1)
    max_values = W.max(axis=1)
    assert (W == max_values[:, None]).sum(axis=1).max() == 1, 'Tied largest weights need an explicit display decision'
    top = np.zeros_like(W)
    top[np.arange(len(W)), dominant] = max_values
    assert np.count_nonzero(top, axis=1).min() == 1 and np.count_nonzero(top, axis=1).max() == 1
    for k, c in enumerate(COMPONENTS):
        sites.loc[available, f'display_top_{c}'] = top[:, k]
    sites.loc[available, 'display_top_component'] = np.array(COMPONENTS)[dominant]
    sites.loc[available, 'display_top_weight'] = max_values
    sites.loc[available, 'top_differs_from_frozen_label'] = (
        sites.loc[available, 'display_top_component'] != sites.loc[available, 'functional_cluster'])
    assert sites.loc[~available, 'functional_cluster'].eq('Unassigned').all()
    assert sites.loc[~available, columns].isna().all().all()
    counts = []
    for group, expected_counts in zip(GROUPS, [(25, 23, 2), (13, 12, 1), (11, 9, 2)]):
        selected = sites.loc[sites.response_set.eq(group)]
        known = selected.loc[selected.has_original_nmf_weight]
        assert (len(selected), len(known), len(selected) - len(known)) == expected_counts
        counts.append(dict(response_region=DISPLAY[group], n_sites=len(selected), n_original_weights=len(known),
            n_unassigned=len(selected) - len(known), n_participants=selected.subject.nunique(),
            **{f'top_{c}': int(known.display_top_component.eq(c).sum()) for c in COMPONENTS},
            n_display_top_vs_frozen_label_discrepancies=int(known.top_differs_from_frozen_label.sum())))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sites.to_csv(OUTPUT / 'electrode_top_weights.csv', index=False)
    sites.loc[~available].to_csv(OUTPUT / 'unassigned_sites.csv', index=False)
    sites.loc[sites.top_differs_from_frozen_label].to_csv(OUTPUT / 'top_vs_frozen_label_discrepancies.csv', index=False)
    pd.DataFrame(counts).to_csv(OUTPUT / 'region_counts.csv', index=False)
    return sites, hashes, counts


def _top_rgba(density, colors_rgb):
    """Pure dominant hue; preserve the reference's total/P95 alpha transform."""
    import numpy as np
    total = density.sum(axis=1)
    positive = total > 0
    dominant = density.argmax(axis=1)
    rgba = np.zeros((len(density), 4))
    rgba[positive, :3] = colors_rgb[dominant[positive]]
    scale = float(np.percentile(total[positive], 95)) if positive.any() else 0.
    rgba[:, 3] = np.clip(total / max(scale, 1e-12), 0, 1)
    assert np.isfinite(rgba).all()
    if positive.any():
        assert all(tuple(rgb) in {tuple(c) for c in colors_rgb} for rgb in np.unique(rgba[positive, :3], axis=0))
    return rgba, scale


def _save_panel(group, sites, images, crop):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    selected = sites.loc[sites.response_set.eq(group)]
    known = selected.loc[selected.has_original_nmf_weight]
    cm = 1 / 2.54
    fig, axes = plt.subplots(1, 2, figsize=(10.8 * cm, 6.6 * cm))
    fig.subplots_adjust(left=.025, right=.975, top=.79, bottom=.29, wspace=.025)
    fig.suptitle(f'{DISPLAY[group]} · top NMF weight\n{len(known)} mapped / {len(selected)} responsive sites', fontsize=8.5, y=.99)
    for ax, hemi, label in zip(axes, ['lh', 'rh'], ['Left insula', 'Right insula']):
        image, _ = crop(images[hemi], np.empty((0, 2)))
        # Keep the original 800-pixel render in the SVG instead of downsampling
        # it to the notebook's small on-screen axes dimensions.
        ax.imshow(image, interpolation='none')
        ax.axis('off')
        take = selected.x.lt(0) if hemi == 'lh' else selected.x.gt(0)
        available = selected.loc[take, 'has_original_nmf_weight']
        ax.set_title(f'{label} · {int(available.sum())}/{int(take.sum())} mapped', fontsize=7.2, pad=2)
    fig.canvas.draw()
    for ax, shift in zip(axes, (.025, -.025)):
        box = ax.get_position()
        ax.set_position([box.x0 + shift, box.y0, box.width, box.height])
    handles = [Patch(facecolor=COLORS[c], label=LABELS[c]) for c in COMPONENTS]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .17), frameon=False,
        ncol=3, fontsize=7, columnspacing=1, handlelength=1.2, handletextpad=.4)
    fig.text(.5, .105, f'Unassigned (no original W): {len(selected)-len(known)} · Stimulus 0–1 s', ha='center', fontsize=6.5)
    fig.text(.5, .035, 'Top raw loading only · Gaussian 5 mm · opacity normalized within each map', ha='center', fontsize=6)
    path = IMG / EXPORTS[KEYS[group]]
    fig.savefig(path, format='svg', bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return path


def render():
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm'
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import json
    from scripts.refresh_lexical_condition_sets import sha, write
    from src.supplementary.decision_repeat_maps import _figure1_local_insula_renderer
    from src.supplementary.selected_brain_contours import _figure2_contour_helpers
    data_paths = [p for p in OUTPUT.rglob('*') if p.is_file()]
    assert data_paths, 'Saved Gaussian fields are required; do not silently recompute'
    hashes = {str(p): sha(p) for p in data_paths}
    previous = json.loads((OUTPUT / 'provenance.json').read_text())
    assert all(sha(p) == h for p, h in previous['inputs'].items())
    sites = pd.read_csv(OUTPUT / 'electrode_top_weights.csv', float_precision='round_trip')
    assert len(sites) == 49 and sites.electrode.is_unique and sites.has_original_nmf_weight.sum() == 44
    fields = np.load(OUTPUT / 'top_surface_fields.npz', allow_pickle=False)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'svg.fonttype': 'none'})
    renderer, renderer_proof = _figure1_local_insula_renderer(ROOT)
    helpers, crop_proof = _figure2_contour_helpers(ROOT)
    # Figure 1 and the cached fields must use the exact same pial vertices.
    assert all(sha(p) == h for p, h in previous['renderer_reference']['surfaces'].items())
    paths, mesh_audit = {}, []
    for group in GROUPS:
        print('Rendering saved top-weight fields on Figure 1 local insula:', DISPLAY[group], flush=True)
        images = {}
        for hemi in ['lh', 'rh']:
            brain = None
            try:
                surface, pial = renderer[f'{hemi}_surf'], renderer[f'{hemi}_pial_coords']
                density = fields[f'{KEYS[group]}_{hemi}_top_density']
                rgba = fields[f'{KEYS[group]}_{hemi}_rgba']
                assert rgba.shape == (len(pial), 4) and density.shape == (len(pial), 3)
                assert np.isin(surface['faces'], surface['verts']).all()
                brain = renderer['_make_insula_brain'](hemi)
                mesh = surface['mesh'].copy(deep=True)
                np.testing.assert_array_equal(np.asarray(mesh.points), pial)
                np.testing.assert_array_equal(np.asarray(mesh.faces).reshape(-1, 4)[:, 1:], surface['faces'])
                rgba_bytes = (np.clip(rgba, 0., 1.) * 255.).astype(np.uint8)
                mesh.point_data['rgba'] = rgba_bytes
                # Identical saved RGBA conversion and mesh settings; only the
                # visible faces change from whole hemisphere to Figure 1 insula.
                brain._renderer.plotter.add_mesh(mesh, scalars='rgba', rgba=True,
                    lighting=False, smooth_shading=True)
                brain.show_view(azimuth=180 if hemi == 'lh' else 0, elevation=90, distance=180,
                    focalpoint=surface['center'])
                for _ in range(2):
                    brain._renderer.plotter.render()
                images[hemi] = np.asarray(brain.screenshot(mode='rgb'))
                assert images[hemi].shape[:2] == (800, 800)
                mesh_audit.append(dict(response_region=DISPLAY[group], hemisphere=hemi,
                    n_visible_insula_faces=int(len(surface['faces'])), n_saved_pial_vertices=len(pial),
                    original_vertex_indices_preserved=True, saved_RGBA_used_without_renormalization=True,
                    whole_hemisphere_hidden=True, original_camera_preserved=True))
            finally:
                if brain is not None:
                    brain.close()
        paths[KEYS[group]] = _save_panel(group, sites, images, helpers['crop_to_content'])
    fields.close()
    assert all(sha(p) == h for p, h in hashes.items()), 'A saved Gaussian field or numerical result changed'
    assert all(sha(p) == h for p, h in previous['inputs'].items())
    DISPLAY_OUTPUT.mkdir(parents=True, exist_ok=True)
    write(DISPLAY_OUTPUT / 'provenance.json', dict(n_bilateral_maps=3, n_sites=49, n_original_weights=44, n_unassigned=5,
        grouping_window=[0, 1], counts=previous['counts'], inputs=hashes,
        renderer_reference=renderer_proof, crop_reference=crop_proof, visible_meshes=mesh_audit,
        change='Figure 1 local-insula display only; peripheral whole-hemisphere structures hidden',
        previous_numerical_data_files_byte_identical=True, gaussian_fields_recomputed=False,
        weight_values_recomputed=False, saved_RGBA_unchanged=True, alpha_renormalized=False,
        no_new_model_or_inference=True, original_labels_and_coordinates_unchanged=True,
        no_composite_created=True,
        exports={str(p): sha(p) for p in paths.values()}))
    return paths
