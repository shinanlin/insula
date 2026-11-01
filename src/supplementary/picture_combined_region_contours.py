"""Absolute site counts and one bilateral PN response-set/functional-label map."""
import os

from src.supplementary.picture_region_point_panels import (
    ROOT, BASE, IMG, SOURCE, GROUPS, DISPLAY, COMPONENTS, COLORS, LABELS,
    POINT_SIZE, UNASSIGNED_COLOR, _orientation_helper,
)

OUTPUT = BASE / 'combined_region_contours_2026-10-27'
AUDIT = ROOT / 'results/audit/picture_combined_region_contours_2026-10-27'
EXPORTS = {
    'counts': 'fig3_picture_response_regions_absolute_counts.svg',
    'brain': 'fig3_picture_response_regions_combined_contours.svg',
}
MARKERS = {'Sound-only': 'o', 'Both': 'D', 'Picture-only': '^'}
LINESTYLES = {'Sound-only': '-', 'Both': '--', 'Picture-only': ':'}
CONTOUR_COLOR = '#666666'


def notebook_cells():
    import nbformat as nbf
    return [nbf.v4.new_markdown_cell('''## 7. Absolute electrode counts and a single combined bilateral map

Author-approved display draft: **two independent SVG subpanels, without assembly**. Response groups retain native Picture Naming Stimulus **[0,1] s** baseline masks: Sound-only **25**, Both **13**, Image-only **11**. The eligible denominator is unchanged at **128 sites / 27 participants**; the 49-site response union contains 22 participants.

**Counts:** three stacked horizontal bars show absolute electrodes on a common 0–25 count axis. Colors represent the existing frozen Figure 2 functional labels, including the five Unassigned sites. They do not show relative NMF weights or percentages. Sound-only = 8 Sensory / 6 Sustained / 9 Motor / 2 Unassigned; Both = 2 / 7 / 3 / 1; Image-only = 1 / 2 / 6 / 2.

**Combined anatomy:** all 49 responsive electrodes appear once on a single left/right local-insula pair. Color represents the frozen functional label (Sensory blue, Sustained gold, Motor red). Symbol and neutral-gray contour line style represent response-set membership: **Sound-only circle / solid; Both diamond / dashed; Image-only triangle / dotted**. The five sites without an original NMF assignment retain their set's symbol with an open gray outline. No nonsignificant background sites are added.

The original Figure 1 insula surface, camera, surface projection and 4-mm display lift are unchanged. Symbols are vector overlays at the exact previously accepted projected electrode centers, with nominal 24-source-pixel size and thin white edges on classified sites. No coordinates are jittered or moved. Contours reuse the exact original Figure 2 KDE helper (25-pixel bandwidth, 0.18 peak fraction, original minimum of 3 sites per hemisphere/island) for each **response set**, across its functional labels. Contours have **no fill** and indicate descriptive spatial density, not a significant anatomical boundary. The single Both site in the right hemisphere remains visible without a forced contour.

All prior notebook cells, outputs, scientific results and figures are retained. No NMF fitting, relabeling, response testing, statistical correction, weight projection or new inferential analysis is performed.
''', id='picture-combined-region-contours-note'),
        nbf.v4.new_code_cell('''from pathlib import Path
import sys
from IPython.display import SVG, display
_pn_combined_root = Path('/cwork/ns458/insula-paper')
if str(_pn_combined_root) not in sys.path:
    sys.path.insert(0, str(_pn_combined_root))
from src.supplementary.picture_combined_region_contours import render, OUTPUT as combined_region_data
combined_region_paths = render()
display(SVG(filename=str(combined_region_paths['counts'])))
display(SVG(filename=str(combined_region_paths['brain'])))
print('Independent SVGs:', {k: str(v) for k, v in combined_region_paths.items()})
print('Counts:', combined_region_data / 'absolute_component_counts.csv')
print('Plotted electrodes:', combined_region_data / 'displayed_sites.csv')
''', id='picture-combined-region-contours-display')]


def _save(fig, key):
    import matplotlib.pyplot as plt
    path = IMG / EXPORTS[key]
    fig.savefig(path, format='svg', bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return path


def _counts_panel(counts):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(11.5 * cm, 5.8 * cm))
    fig.subplots_adjust(left=.23, right=.91, top=.73, bottom=.23)
    for y, group in enumerate(GROUPS):
        row = counts.loc[counts.response_set.eq(group)].iloc[0]
        left = 0
        for name, color in [(LABELS[c], COLORS[c]) for c in COMPONENTS] + [('Unassigned', UNASSIGNED_COLOR)]:
            n = int(row[name])
            ax.barh(y, n, left=left, height=.62, color=color,
                    edgecolor='white', linewidth=.65)
            if n:
                ax.text(left + n / 2, y, str(n), ha='center', va='center', color='black', fontsize=7)
            left += n
        assert left == int(row.n_electrodes)
        ax.text(left + .35, y, str(left), ha='left', va='center', fontsize=7.5,
                fontweight='bold', clip_on=False)
    ax.set_yticks(range(3), [DISPLAY[g] for g in GROUPS])
    ax.set_ylim(2.65, -.65)
    ax.set_xlim(0, 25)
    ax.set_xticks(range(0, 26, 5))
    ax.set_xlabel('Electrodes', labelpad=3)
    ax.tick_params(axis='y', length=0, pad=6)
    ax.tick_params(axis='x', direction='out', length=3)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    handles = [Patch(facecolor=COLORS[c], label=LABELS[c]) for c in COMPONENTS]
    handles.append(Patch(facecolor=UNASSIGNED_COLOR, label='Unassigned'))
    fig.legend(handles=handles, ncol=4, loc='upper center', bbox_to_anchor=(.5, .89),
               frameon=False, fontsize=6.5, handlelength=1, columnspacing=.85, handletextpad=.4)
    fig.suptitle('Picture Naming · Stimulus 0–1 s', fontsize=8.5, y=.99)
    fig.text(.5, .025, '49 responsive electrodes · absolute counts', ha='center', fontsize=6.5)
    return _save(fig, 'counts')


def _combined_map(sites, reference_positions, renderer, helpers, draw_orientation):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    cm = 1 / 2.54
    fig, axes = plt.subplots(1, 2, figsize=(12 * cm, 7.4 * cm))
    fig.subplots_adjust(left=.025, right=.975, bottom=.32, top=.84, wspace=.025)
    fig.suptitle('Picture Naming · Stimulus 0–1 s\n49 responsive sites / 22 participants', fontsize=8, y=.99)
    audit, contour_rows, ring_rows, orientation, pending = [], [], [], [], []
    for ax, hemi, label in zip(axes, ['lh', 'rh'], ['Left insula', 'Right insula']):
        frame = sites.loc[sites.x.lt(0) if hemi == 'lh' else sites.x.gt(0)].copy()
        xyz = frame[['x', 'y', 'z']].to_numpy(float)
        brain = None
        try:
            # Anatomy-only image; every electrode is drawn once as a vector symbol.
            image, project, brain = renderer['_render_insula_hemi'](
                hemi, np.empty((0, 3)), np.array([], dtype=object), np.array([]))
            assert image.shape[:2] == (800, 800)
            camera = renderer['_camera_xyz'](brain)
            surface, pial = renderer[f'{hemi}_surf'], renderer[f'{hemi}_pial_coords']
            displayed = renderer['_project_to_insula'](xyz, surface, pial, camera)
            raw_pixels = project(displayed)
            toward = camera - displayed
            toward /= np.linalg.norm(toward, axis=1, keepdims=True)
            snapped = displayed - 4. * toward
            residual, _ = renderer['cKDTree'](pial[surface['verts']]).query(snapped)
            assert (residual < 1e-8).all()
            np.testing.assert_allclose(np.linalg.norm(displayed - snapped, axis=1), 4., rtol=0, atol=1e-10)
            center = surface['center']
            probes = project(np.stack([center + [0, 5, 0], center - [0, 5, 0],
                                       center + [0, 0, 5], center - [0, 0, 5]]))
            anterior_left = bool(probes[0, 0] < probes[1, 0])
            assert probes[2, 1] < probes[3, 1]
        finally:
            if brain is not None:
                brain.close()
        cropped, pixels = helpers['crop_to_content'](image, raw_pixels)
        offset = raw_pixels[0] - pixels[0]
        assert ((pixels >= 0) & (pixels < np.array([cropped.shape[1], cropped.shape[0]]))).all()
        ax.imshow(cropped)
        ax.set_xlim(0, cropped.shape[1])
        ax.set_ylim(cropped.shape[0], 0)
        ax.axis('off')
        ax.set_title(f'{label} · n={len(frame)}', fontsize=7.5, pad=2)
        draw_orientation(ax, anterior_is_left=anterior_left, corner='lower left' if hemi == 'lh' else 'lower right')
        orientation.append(dict(hemisphere=hemi, anterior_is_left=anterior_left))
        for group in GROUPS:
            chosen = frame.response_set.eq(group).to_numpy()
            # Raw 800-pixel scale retains the original KDE bandwidth exactly.
            rings = helpers['kde_contour_rings'](raw_pixels[chosen], image.shape[:2])
            contour_rows.append(dict(hemisphere=hemi, response_region=DISPLAY[group],
                n_sites=int(chosen.sum()), n_rings=len(rings), line_style=LINESTYLES[group],
                color=CONTOUR_COLOR, filled=False))
            for ri, ring in enumerate(rings):
                translated = ring - offset
                line, = ax.plot(translated[:, 0], translated[:, 1], color=CONTOUR_COLOR,
                                linestyle=LINESTYLES[group], linewidth=.85, zorder=3)
                line.set_gid(f'outline-only-{hemi}-{group}-{ri}')
                ring_rows.extend(dict(hemisphere=hemi, response_region=DISPLAY[group], ring=ri,
                    vertex=vi, pixel_x=float(x), pixel_y=float(y))
                    for vi, (x, y) in enumerate(ring))
        pending.append((ax, frame, pixels, cropped.shape[1]))
        for i, row in enumerate(frame.itertuples()):
            assigned = bool(row.has_original_nmf_weight)
            audit.append(dict(electrode=row.electrode, subject=row.subject, hemisphere=hemi,
                response_region=DISPLAY[row.response_set], functional_cluster=row.functional_cluster,
                has_original_nmf_weight=assigned, marker=MARKERS[row.response_set],
                marker_color=COLORS[row.functional_cluster] if assigned else UNASSIGNED_COLOR,
                marker_fill='functional color' if assigned else 'none',
                contour_line_style=LINESTYLES[row.response_set],
                x_original=float(xyz[i, 0]), y_original=float(xyz[i, 1]), z_original=float(xyz[i, 2]),
                pixel_x=float(raw_pixels[i, 0]), pixel_y=float(raw_pixels[i, 1]),
                cropped_pixel_x=float(pixels[i, 0]), cropped_pixel_y=float(pixels[i, 1]),
                surface_residual_mm=float(residual[i]), display_lift_mm=4.))
    fig.canvas.draw()
    for ax, shift in zip(axes, (.025, -.025)):
        box = ax.get_position()
        ax.set_position([box.x0 + shift, box.y0, box.width, box.height])
    fig.canvas.draw()
    for ax, frame, pixels, image_width in pending:
        diameter_pt = POINT_SIZE * ax.get_window_extent().width / image_width * 72 / fig.dpi
        for group in GROUPS:
            for component in (*COMPONENTS, 'Unassigned'):
                chosen = (frame.response_set.eq(group) & frame.functional_cluster.eq(component)).to_numpy()
                if not chosen.any():
                    continue
                unknown = component == 'Unassigned'
                points = ax.scatter(pixels[chosen, 0], pixels[chosen, 1], s=diameter_pt ** 2,
                    marker=MARKERS[group], facecolors='none' if unknown else COLORS[component],
                    edgecolors=UNASSIGNED_COLOR if unknown else 'white',
                    linewidths=.7 if unknown else .35, zorder=6)
                points.set_gid(f'sites-{ax.get_title().split()[0]}-{group}-{component}')
    component_handles = [Line2D([], [], ls='none', marker='o', ms=4.5, mfc=COLORS[c], mec='white', mew=.35,
        label=f'{LABELS[c]} ({int(sites.functional_cluster.eq(c).sum())})') for c in COMPONENTS]
    component_handles.append(Line2D([], [], ls='none', marker='o', ms=4.5, mfc='none', mec=UNASSIGNED_COLOR,
        label='Unassigned (5)'))
    fig.legend(handles=component_handles, ncol=4, loc='lower center', bbox_to_anchor=(.5, .205),
               frameon=False, fontsize=6.3, handletextpad=.25, columnspacing=.65)
    set_handles = [Line2D([], [], color=CONTOUR_COLOR, ls=LINESTYLES[g], lw=.85,
        marker=MARKERS[g], mfc='white', mec=CONTOUR_COLOR, ms=4,
        label=f'{DISPLAY[g]} ({int(sites.response_set.eq(g).sum())})') for g in GROUPS]
    fig.legend(handles=set_handles, ncol=3, loc='lower center', bbox_to_anchor=(.5, .125),
               frameon=False, fontsize=6.5, handlelength=2.6, columnspacing=1, handletextpad=.45)
    fig.text(.5, .067, 'Contours: spatial density, no fill · Color: functional component', ha='center', fontsize=6)
    fig.text(.5, .018, 'Open gray symbols: responsive sites without an original NMF assignment', ha='center', fontsize=6)
    audit = pd.DataFrame(audit)
    assert len(audit) == 49 and audit.electrode.is_unique
    assert audit.has_original_nmf_weight.sum() == 44 and audit.marker_fill.eq('none').sum() == 5
    compared = audit.merge(reference_positions[['electrode', 'pixel_x', 'pixel_y']],
        on='electrode', validate='one_to_one', suffixes=('_new', '_old'))
    assert len(compared) == 49
    for key in ['pixel_x', 'pixel_y']:
        np.testing.assert_allclose(compared[key + '_new'], compared[key + '_old'], atol=1e-8, rtol=0)
    audit.to_csv(OUTPUT / 'displayed_sites.csv', index=False)
    pd.DataFrame(contour_rows).to_csv(OUTPUT / 'contour_set_counts.csv', index=False)
    pd.DataFrame(ring_rows).to_csv(OUTPUT / 'contour_vertices.csv', index=False)
    pd.DataFrame(orientation).to_csv(OUTPUT / 'anatomical_orientation.csv', index=False)
    return _save(fig, 'brain'), contour_rows


def render():
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm'
    import pandas as pd
    import matplotlib.pyplot as plt
    from scripts.refresh_lexical_condition_sets import sha, write
    from src.supplementary.decision_repeat_maps import _figure1_local_insula_renderer
    from src.supplementary.selected_brain_contours import _figure2_contour_helpers
    sources = [SOURCE / 'electrode_weights.csv', BASE / 'electrodes.csv',
        BASE / 'functional_set_contours_2026-10-27/counts.csv',
        BASE / 'baseline_classes_display_coordinates.csv', ROOT / 'results/nmf/channel_assignments.csv']
    hashes = {str(p): sha(p) for p in sources}
    sites = pd.read_csv(SOURCE / 'electrode_weights.csv', float_precision='round_trip')
    table = pd.read_csv(BASE / 'electrodes.csv')
    counts = pd.read_csv(BASE / 'functional_set_contours_2026-10-27/counts.csv')
    reference = pd.read_csv(BASE / 'baseline_classes_display_coordinates.csv', float_precision='round_trip')
    assignments = pd.read_csv(ROOT / 'results/nmf/channel_assignments.csv')
    labels = assignments.set_index(['subject', 'channel']).functional_cluster
    assert len(table) == 128 and table.subject.nunique() == 27
    assert len(sites) == 49 and sites.electrode.is_unique and sites.subject.nunique() == 22
    assert set(sites.electrode) == set(table.loc[table.sound_responsive | table.image_responsive, 'electrode'])
    expected = {'Sound-only': [8, 6, 9, 2], 'Both': [2, 7, 3, 1], 'Picture-only': [1, 2, 6, 2]}
    for row in sites.itertuples():
        if row.has_original_nmf_weight:
            assert row.functional_cluster == labels.loc[(row.subject, row.channel)]
        else:
            assert row.functional_cluster == 'Unassigned'
    for group in GROUPS:
        frame = sites.loc[sites.response_set.eq(group)]
        observed = [int(frame.functional_cluster.eq(c).sum()) for c in (*COMPONENTS, 'Unassigned')]
        assert observed == expected[group]
        row = counts.loc[counts.response_set.eq(group)].iloc[0]
        assert [int(row[LABELS[c]]) for c in COMPONENTS] + [int(row.Unassigned)] == observed
        assert int(row.n_electrodes) == len(frame) and int(row.n_participants) == frame.subject.nunique()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'svg.fonttype': 'none'})
    paths = {'counts': _counts_panel(counts)}
    renderer, renderer_proof = _figure1_local_insula_renderer(ROOT)
    helpers, contour_proof = _figure2_contour_helpers(ROOT)
    draw_orientation, orientation_proof = _orientation_helper()
    paths['brain'], contours = _combined_map(sites, reference, renderer, helpers, draw_orientation)
    counts.assign(response_region=counts.response_set.map(DISPLAY)).to_csv(OUTPUT / 'absolute_component_counts.csv', index=False)
    assert all(sha(p) == h for p, h in hashes.items())
    write(OUTPUT / 'provenance.json', dict(n_response_regions=3, n_bilateral_brain_panels=1, n_bar_panels=1,
        no_composite_created=True, n_displayed_sites=49, n_existing_assignments=44, n_open_gray_unassigned=5,
        grouping_window=[0, 1], bar_metric='Absolute counts of frozen functional_cluster labels',
        counts={DISPLAY[g]: expected[g] for g in GROUPS}, colors=COLORS,
        markers={DISPLAY[g]: MARKERS[g] for g in GROUPS},
        contour_line_styles={DISPLAY[g]: LINESTYLES[g] for g in GROUPS},
        contour_color=CONTOUR_COLOR, contour_fill=False, contour_groups='Response sets, including Unassigned sites',
        contour_counts=contours, no_Gaussian_RGB_overlay=True, no_model_fit_or_new_statistics=True,
        all_49_pixel_centers_match_previous_accepted_export=True, analytic_coordinates_unchanged=True,
        point_nominal_pixels=POINT_SIZE, renderer=renderer_proof, contour_method=contour_proof,
        orientation=orientation_proof, inputs=hashes, exports={str(p): sha(p) for p in paths.values()}))
    return paths
