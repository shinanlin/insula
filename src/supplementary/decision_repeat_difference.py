"""Display paired Decision-minus-Repeat HGA without changing saved inference."""
import os
from pathlib import Path

ROOT = Path('/cwork/ns458/insula-paper')
BASE = ROOT / 'results/supplementary/decision_repeat_test'
OUTPUT = BASE / 'difference_heatmap_2026-10-27'
AUDIT = ROOT / 'results/audit/decision_repeat_difference_heatmap_2026-10-27'
NOTEBOOK = ROOT / 'vizpub/supplementary/decision_repeat_test.ipynb'
IMAGE = ROOT / 'img/supplementary/decision_repeat_test/fig3_decision_repeat_test.svg'
CLUSTERS = ('sensory', 'sustain', 'motor')
LABELS = {'sensory': 'Sensory', 'sustain': 'Sustained', 'motor': 'Motor'}
NEGATIVE_COLOR = '#737373'


def notebook_cells():
    import nbformat as nbf
    title = '''# Figure 3 — Decision / Repeat traces and HGA difference heatmaps

Top curves and their saved purple significance bars remain unchanged: **123 sites / 34 participants**, electrode mean ± SEM, Decision purple and Repeat black. The lower heatmaps retain the reviewed **29 Delay-selected sites / 18 participants** (Sensory 3 / Sustained 14 / Motor 12), existing shared row order, 1-mm row height and original phase widths/windows.

**Heatmap values are now paired HGA differences: Decision − Repeat**, in the original HGA z-score units. Match the same electrode and time sample using the unchanged native-QC-kept trial means. Subtract the two unmasked condition means first; do not subtract independently masked rasters or pair unrelated trials. Retain the preceding Decision baseline-significance display mask; missing values in either condition stay blank. The blanking mask is not a new test of the difference, and no per-time difference significance is claimed by the color alone.

As requested, each component retains its original color family: positive differences use Sensory blue / Sustained gold / Motor red; negative differences use gray; zero is white. All nine rasters share a symmetric range based on the P95 absolute displayed difference (minimum 0.15 z), with a separate signed color key per component. Values beyond the color endpoints saturate without changing the saved numbers. No smoothing, inference, site reselection, reordering, NMF change or main-figure change is performed.

The original Decision-only matrices, trace/SEM tables, temporal significance masks, electrode lists and statistical results remain unchanged. Their prior plot is archived. The new difference values, finite-pair audit and display specification are saved separately under `difference_heatmap_2026-10-27/`. The earlier preparation/validation cells reproduce the original single-condition rasters as the reference for the retained display mask; the plotting cell below loads the frozen results and displays the new metric.
'''
    plot = '''# Display-only refresh: original upper curves, paired HGA-difference heatmaps below.
from pathlib import Path
import sys
from IPython.display import SVG, display
_dr_difference_root = Path('/cwork/ns458/insula-paper')
if str(_dr_difference_root) not in sys.path:
    sys.path.insert(0, str(_dr_difference_root))
from src.supplementary.decision_repeat_difference import render, OUTPUT as difference_output
difference_svg = render()
display(SVG(filename=str(difference_svg)))
print('Difference matrices and audit:', difference_output)
'''
    preview = '''# The canonical SVG and notebook output above show Decision − Repeat.
from src.supplementary.decision_repeat_difference import IMAGE, OUTPUT
print('Updated SVG:', IMAGE)
print('Display validation:', OUTPUT / 'plot_validation.json')
print('Original trace values, statistics, selected electrodes and row order are unchanged.')
'''
    return [nbf.v4.new_markdown_cell(title, id='test-title'),
            nbf.v4.new_code_cell(plot, id='test-plot'),
            nbf.v4.new_code_cell(preview, id='test-preview')]


def _prepare():
    import numpy as np
    import pandas as pd
    from scripts.refresh_lexical_condition_sets import sha, write
    source = ROOT / 'results/supplementary/context_comparisons/waveforms.pkl'
    spec = __import__('json').loads((BASE / 'plot_validation.json').read_text())
    assert sha(source) == spec['source_sha256']
    assert sha(ROOT / 'results/nmf/channel_assignments.csv') == spec['assignments_sha256']
    assert spec['component_site_counts'] == {'sensory': 3, 'sustain': 14, 'motor': 12}
    assert spec['n_sites'] == 123 and spec['n_heatmap_sites'] == 29
    members = pd.read_csv(BASE / 'decision_heatmap_row_membership.csv')
    traces = pd.read_csv(BASE / 'pooled_electrode_mean_sem.csv', float_precision='round_trip')
    intervals = pd.read_csv(BASE / 'decision_gt_repeat_significant_intervals.csv', float_precision='round_trip')
    cache = pd.read_pickle(source)
    frame = cache.loc[cache.contrast.eq('decision_repeat')].copy()
    frame['electrode'] = frame.subject.astype(str) + '|' + frame.channel.astype(str)
    assert frame.task.eq('LexicalDelay').all() and frame.modality.eq('sound').all()
    assert frame.electrode.nunique() == 123 and frame.subject.nunique() == 34
    assert set(frame.level) == {'Decision', 'Repeat'}
    phases = tuple(spec['xlims'])
    assert phases == ('stimulus', 'delay', 'response')
    matrices, unmasked, orders, payload, audits = {}, {}, {}, {}, []
    with np.load(BASE / 'decision_heatmap_matrices.npz', allow_pickle=False) as old:
        for phase in phases:
            lower, upper = spec['xlims'][phase]
            sub = frame.loc[frame.phase.eq(phase) & frame.time.between(lower, upper)]
            times = np.sort(sub.time.unique())
            payload[f'{phase}_time'] = times
            for cluster in CLUSTERS:
                membership = members.loc[members.phase.eq(phase) & members.functional_cluster.eq(cluster)].sort_values('row')
                order = membership.electrode.tolist()
                assert len(order) == spec['component_site_counts'][cluster] and len(set(order)) == len(order)
                if cluster in orders:
                    assert order == orders[cluster]
                else:
                    orders[cluster] = order
                pair = {}
                for condition in ('Decision', 'Repeat'):
                    take = sub.loc[sub.level.eq(condition) & sub.electrode.isin(order)]
                    assert set(take.electrode) == set(order)
                    pair[condition] = take.pivot_table(index='electrode', columns='time', values='value', aggfunc='mean').reindex(
                        index=order, columns=times).to_numpy(float)
                key = f'{phase}_{cluster}'
                reference = old[key]
                assert reference.shape == pair['Decision'].shape == pair['Repeat'].shape
                mask = np.isfinite(reference)
                np.testing.assert_allclose(pair['Decision'][mask], reference[mask], rtol=0, atol=1e-14)
                paired_finite = np.isfinite(pair['Decision']) & np.isfinite(pair['Repeat'])
                difference = pair['Decision'] - pair['Repeat']
                displayed = np.where(mask & paired_finite, difference, np.nan)
                np.testing.assert_array_equal(displayed[np.isfinite(displayed)], difference[np.isfinite(displayed)])
                matrices[(phase, cluster)] = displayed
                unmasked[(phase, cluster)] = difference
                payload[key] = displayed
                payload[key + '_unmasked_difference'] = difference
                payload[key + '_original_decision_mask'] = mask
                payload[key + '_paired_finite'] = paired_finite
                payload[key + '_electrodes'] = np.asarray(order, dtype=str)
                for row, electrode in enumerate(order):
                    audits.append(dict(phase=phase, functional_cluster=cluster, row=row, electrode=electrode,
                        n_samples=len(times), n_original_mask_bins=int(mask[row].sum()),
                        n_displayed_bins=int(np.isfinite(displayed[row]).sum()),
                        n_missing_pairs_in_original_mask=int((mask[row] & ~paired_finite[row]).sum()),
                        n_positive_bins=int((displayed[row] > 0).sum()), n_negative_bins=int((displayed[row] < 0).sum())))
    all_values = np.concatenate([m[np.isfinite(m)] for m in matrices.values()])
    assert all_values.size and np.isfinite(all_values).all()
    limit = max(.15, float(np.quantile(np.abs(all_values), .95)))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / 'decision_minus_repeat_heatmap_matrices.npz', **payload)
    pd.DataFrame(audits).to_csv(OUTPUT / 'paired_sample_audit.csv', index=False)
    write(OUTPUT / 'color_scale.json', dict(symmetric_limit=limit, units='Difference of HGA z-scored condition means',
        rule='P95 of absolute finite displayed differences across all nine panels; floor 0.15',
        n_finite_values=int(len(all_values)), minimum=float(all_values.min()), maximum=float(all_values.max()),
        n_saturated_values=int((np.abs(all_values) > limit).sum()), negative_color=NEGATIVE_COLOR, zero_color='white'))
    return spec, traces, intervals, matrices, orders, limit, audits


def render():
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm'
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap, to_hex
    from src.nmf.waveform_analysis import FUNCTION_COLORS
    from src.paths import save_svg
    from scripts.refresh_lexical_condition_sets import write
    import sys
    for font in Path(sys.prefix, 'fonts').glob('arial*.ttf'):
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'sans-serif', 'font.sans-serif': ['Arial'], 'font.size': 7})
    spec, traces, intervals, matrices, orders, limit, audits = _prepare()
    geometry, xlims = spec['reference_geometry'], spec['xlims']
    phases = tuple(xlims)
    cm, fontsize = 1 / 2.54, 7
    trace_height_cm = geometry['trace_height_cm']
    row_height_cm = spec['actual_cm_per_electrode_row']
    row_heights_cm = [trace_height_cm] + [len(orders[c]) * row_height_cm for c in CLUSTERS]
    widths_cm = geometry['phase_widths_cm']
    left_cm, right_cm = geometry['left_cm'], geometry['right_cm']
    top_cm, bottom_cm = geometry['top_cm'], geometry['bottom_cm']
    gap_x_cm, gap_y_cm = geometry['gap_x_cm'], geometry['gap_y_cm']
    fig_w_cm = left_cm + sum(widths_cm) + 2 * gap_x_cm + right_cm
    fig_h_cm = top_cm + sum(row_heights_cm) + 3 * gap_y_cm + bottom_cm
    fig = plt.figure(figsize=(fig_w_cm * cm, fig_h_cm * cm))
    gs = fig.add_gridspec(4, 3, height_ratios=row_heights_cm, width_ratios=widths_cm,
        left=left_cm / fig_w_cm, right=1 - right_cm / fig_w_cm, top=1 - top_cm / fig_h_cm,
        bottom=bottom_cm / fig_h_cm, wspace=gap_x_cm / np.mean(widths_cm), hspace=gap_y_cm / np.mean(row_heights_cm))
    axes = np.empty((4, 3), dtype=object)
    condition_colors = spec['trace_condition_colors']
    ticks = {'stimulus': [-.5, 0, .6], 'delay': [0, 1], 'response': [-.5, 0, 1]}
    images = {}
    for j, phase in enumerate(phases):
        for row in range(4):
            axes[row, j] = fig.add_subplot(gs[row, j], sharex=axes[0, j] if row else None)
        ax = axes[0, j]
        for condition in ('Decision', 'Repeat'):
            sub = traces.loc[traces.phase.eq(phase) & traces.level.eq(condition)]
            time, mean, sem = sub.time.to_numpy(), sub['mean'].to_numpy(), sub['sem'].to_numpy()
            ax.plot(time, mean, color=condition_colors[condition], lw=1, label=condition)
            ax.fill_between(time, mean - sem, mean + sem, color=condition_colors[condition], alpha=.18, linewidth=0)
        ax.axhline(0, color='0.2', ls='--', lw=.7)
        ax.axvline(0, color='0.2', ls='--', lw=.7)
        ax.set_xlim(*xlims[phase]); ax.set_ylim(*spec['trace_ylim'])
        for span in intervals.loc[intervals.phase.eq(phase)].itertuples():
            ax.plot([span.start, span.end], [1.03, 1.03], transform=ax.get_xaxis_transform(),
                color=condition_colors['Decision'], lw=1.5, solid_capstyle='butt', clip_on=False)
        ax.set_title(phase.capitalize(), fontsize=fontsize)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelbottom=False, labelleft=False, left=False, labelsize=fontsize, width=.75, length=2)
        ax.spines['left'].set_visible(False)
        if j == 0:
            ax.text(.02, .96, f"n={spec['n_sites']}", transform=ax.transAxes, va='top', fontsize=fontsize)
        for i, cluster in enumerate(CLUSTERS):
            ax = axes[i + 1, j]
            n = len(orders[cluster])
            matrix = matrices[(phase, cluster)]
            cmap = LinearSegmentedColormap.from_list('signed_' + cluster,
                [(0., NEGATIVE_COLOR), (.5, '#FFFFFF'), (1., FUNCTION_COLORS[cluster])], N=257)
            cmap.set_bad('white')
            assert to_hex(cmap(1.)) == FUNCTION_COLORS[cluster].lower() and to_hex(cmap(.5)) == '#ffffff'
            image = ax.imshow(matrix, extent=(*xlims[phase], 0, n), origin='lower', aspect='auto',
                cmap=cmap, vmin=-limit, vmax=limit, interpolation='nearest', rasterized=True)
            images[(phase, cluster)] = image
            np.testing.assert_array_equal(np.asarray(image.get_array()), matrix)
            ax.axvline(0, color='0.2', ls='--', lw=.8)
            ax.set_xlim(*xlims[phase]); ax.set_ylim(0, n)
            ax.set_yticks([]); ax.set_xticks(ticks[phase], labels=[f'{t:g}' for t in ticks[phase]])
            ax.get_xticklabels()[0].set_ha('left'); ax.get_xticklabels()[-1].set_ha('right')
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.tick_params(labelbottom=i == 2, bottom=i == 2, labelsize=fontsize, width=.75, length=2)
            if j == 0:
                ax.text(-.07, .5, f'{LABELS[cluster]} (n={n})', transform=ax.transAxes,
                    ha='right', va='center', fontsize=fontsize, color=FUNCTION_COLORS[cluster])
    axes[0, -1].legend(frameon=False, fontsize=fontsize, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    axes[-1, 1].set_xlabel('Time (s)', fontsize=fontsize)
    fig.text(left_cm / fig_w_cm, 1 - .09 / fig_h_cm, 'Lexical Delay · sound', ha='left', va='top', fontsize=fontsize)
    fig.text(.01, .025, 'Heatmaps: Decision − Repeat (gray < 0)', ha='left', va='bottom', fontsize=fontsize)
    fig.canvas.draw()
    for row in range(4):
        scales = [ax.get_position().width / np.diff(ax.get_xlim())[0] for ax in axes[row]]
        np.testing.assert_allclose(scales, scales[0], rtol=1e-12)
    for j in range(3):
        heights = [axes[i + 1, j].get_position().height * fig_h_cm / len(orders[c]) for i, c in enumerate(CLUSTERS)]
        np.testing.assert_allclose(heights, row_height_cm, rtol=1e-12)
    np.testing.assert_allclose([axes[0, j].get_position().width * fig_w_cm for j in range(3)], widths_cm, rtol=1e-12)
    np.testing.assert_allclose(axes[0, 0].get_position().height * fig_h_cm, trace_height_cm, rtol=1e-12)
    # Small signed keys occupy only the existing right margin; plot geometry stays fixed.
    for i, cluster in enumerate(CLUSTERS):
        box = axes[i + 1, -1].get_position()
        cax = fig.add_axes([box.x1 + .15 / fig_w_cm, (box.y0 + box.y1) / 2,
                            .75 / fig_w_cm, .10 / fig_h_cm])
        bar = fig.colorbar(images[(phases[-1], cluster)], cax=cax, orientation='horizontal', ticks=[-limit, limit])
        bar.ax.set_xticklabels([f'{-limit:.2g}', f'{limit:.2g}'])
        bar.ax.tick_params(labelsize=4.5, length=1.3, pad=1)
        bar.outline.set_visible(False)
        bar.ax.set_title('ΔHGA (z)', fontsize=5, pad=2)
    path = save_svg(fig, IMAGE, close=True)
    write(OUTPUT / 'plot_validation.json', dict(passed=True, metric='Decision minus Repeat HGA',
        n_trace_sites=123, n_trace_subjects=34, n_heatmap_sites=29, n_heatmap_subjects=18,
        component_counts={c: len(orders[c]) for c in CLUSTERS}, original_rows_and_order_preserved=True,
        original_decision_baseline_display_mask_retained=True, subtract_unmasked_condition_means_first=True,
        missing_pairs_left_blank=int(sum(r['n_missing_pairs_in_original_mask'] for r in audits)),
        original_trace_geometry_and_phase_widths_preserved=True, trace_ylim=spec['trace_ylim'],
        phase_windows=xlims, cm_per_electrode_row=row_height_cm, symmetric_color_limit=limit,
        component_positive_colors=FUNCTION_COLORS, negative_color=NEGATIVE_COLOR, zero_color='white',
        statistics_recomputed=False, site_selection_recomputed=False, smoothing_added=False,
        upper_trace_and_inference_files_unchanged=True, svg=str(path)))
    return path
