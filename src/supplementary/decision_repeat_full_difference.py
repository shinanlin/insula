"""Canonical 187/31 Decision–Repeat curves and positive-range HGA differences."""
import json
import os
from pathlib import Path

ROOT = Path('/cwork/ns458/insula-paper')
BASE = ROOT / 'results/supplementary/decision_repeat_test'
FULL = ROOT / 'results/supplementary/fig3_lexical_delay_condition_sets/full_cohort_2026-10-27'
OUTPUT = BASE / 'difference_heatmap_full_cohort_2026-10-27'
AUDIT = ROOT / 'results/audit/decision_repeat_difference_full_cohort_2026-10-27'
NOTEBOOK = ROOT / 'vizpub/supplementary/decision_repeat_test.ipynb'
IMAGE = ROOT / 'img/supplementary/decision_repeat_test/fig3_decision_repeat_test.svg'
PHASES = ('stimulus', 'delay', 'response')
XLIMS = {'stimulus': (-.5, .6), 'delay': (0., 1.), 'response': (-.5, 1.)}
CLUSTERS = ('sensory', 'sustain', 'motor', 'Unassigned')
LABELS = {'sensory': 'Sensory', 'sustain': 'Sustained', 'motor': 'Motor', 'Unassigned': 'Unassigned'}
COLORS = {'sensory': '#2369BD', 'sustain': '#C4A35A', 'motor': '#A9373B', 'Unassigned': '#858585'}
PARAMETERS = dict(p_thresh=.05, p_cluster=.05, tails=1, n_perm=5000,
                  axis=0, permutation_type='samples', seed=42, n_jobs=1)


def notebook_cells():
    import nbformat as nbf
    return [nbf.v4.new_markdown_cell('''# Decision / Repeat: complete-cohort traces and HGA differences

**Current sample: 187 eligible paired insula sites / 35 participants.** This notebook now uses the same authoritative full-cohort inputs as [the complete Lexical characterization](fig3_lexical_delay_condition_sets.ipynb). The preceding 123-site trace / 29-site heatmap notebook and SVG were archived before replacement. No active cell in this version loads that restricted waveform cache.

**Upper curves:** native electrode mean ± SEM for all 187 sites, Decision purple / Repeat black. The pooled Decision > Repeat bars use the previously approved subject-block condition swaps (S/N-weighted subject contributions preserve the plotted electrode mean), original iEEG time_perm_cluster, 5000 permutations, mean_diff, one-sided, p_thresh=p_cluster=.05, seed 42, separate temporal correction per displayed phase. The 123-site bars are not reused. Display windows match the complete-cohort version: Stimulus [−0.5,0.6], Delay [0,1], Response [−0.5,1] s. No trial/QC, classifier, NMF or per-electrode statistical method changes are introduced.

**Lower heatmaps:** the existing full-cohort Delay [0,1] s **Decision > Repeat set of 31 sites / 18 participants**. Its frozen categories are Sensory **3**, Sustained **14**, Motor **12**, and **2 Unassigned** sites without original NMF labels. All 31 are retained; no new assignment is invented. The original row order is preserved for the 29 previously displayed classified sites; the two Unassigned sites are shown in a separate gray row block, ordered by their first existing native baseline response across the plotted phases.

Values are **Decision − Repeat HGA z-score means**, paired by electrode and time. Subtract unmasked condition means before retaining the original Decision-baseline display-blanking rule. Missing pairs remain blank; colors do not denote newly inferred per-time significance. All components use the fixed **0–0.5** mapping: white at zero, original blue/gold/red (gray for Unassigned) at 0.5. Negative differences display as white; values above 0.5 saturate. Signed raw differences remain unchanged in the exported matrices. No positive-only electrode selection, normalization, smoothing or new per-electrode test is performed.

The same physical time scale and 1-mm electrode rows are retained. Original single-condition matrices, 123-site inference, complete-cohort per-electrode tests, source waveforms, Figure 1/2, all NMF files and the separate condition-set/brain-map notebooks stay unchanged. New complete-cohort summaries, paired values, pooled inference and provenance are saved separately under `difference_heatmap_full_cohort_2026-10-27/`.
''', id='test-title'), nbf.v4.new_code_cell('''from pathlib import Path
import sys
from IPython.display import SVG, display
_dr_full_root = Path('/cwork/ns458/insula-paper')
if str(_dr_full_root) not in sys.path:
    sys.path.insert(0, str(_dr_full_root))
from src.supplementary.decision_repeat_full_difference import render, OUTPUT, FULL
full_difference_svg = render()
display(SVG(filename=str(full_difference_svg)))
print('Authoritative complete-cohort inputs:', FULL)
print('Current 187/31 display and inference:', OUTPUT)
''', id='test-plot'), nbf.v4.new_code_cell('''import json
import pandas as pd
display(pd.read_csv(OUTPUT / 'cohort_flow.csv'))
display(pd.read_csv(OUTPUT / 'selected_electrodes.csv')[['subject','channel','functional_cluster','decision_gt_repeat_intervals']])
display(pd.read_csv(OUTPUT / 'pooled_significant_intervals.csv'))
print(json.dumps(json.loads((OUTPUT / 'cohort_audit.json').read_text()), indent=2))
''', id='test-current-cohort-audit')]


def prepare():
    import numpy as np
    import pandas as pd
    from scripts.refresh_lexical_condition_sets import sha, write
    from src.supplementary.lexical_condition_figures import summarize_traces
    paths = [FULL / n for n in ['electrodes.csv','waveforms.csv.gz','decision_gt_repeat_electrodes.csv','summary.json']]
    paths += [BASE / 'decision_heatmap_row_membership.csv', BASE / 'delay_electrode_selection/selected_electrodes.csv']
    hashes = {str(p): sha(p) for p in paths}
    table = pd.read_csv(FULL / 'electrodes.csv')
    waves = pd.read_csv(FULL / 'waveforms.csv.gz', float_precision='round_trip')
    selected = pd.read_csv(FULL / 'decision_gt_repeat_electrodes.csv')
    original = pd.read_csv(BASE / 'delay_electrode_selection/selected_electrodes.csv')
    old_rows = pd.read_csv(BASE / 'decision_heatmap_row_membership.csv')
    summary = json.loads((FULL / 'summary.json').read_text())
    assert len(table) == summary['n_sites'] == 187 and table.subject.nunique() == 35
    assert table.electrode.is_unique and table.eligible.all() and table.paired_statistics_complete.all()
    assert len(selected) == 31 and selected.subject.nunique() == 18 and selected.electrode.is_unique
    assert set(selected.electrode) == set(table.loc[table.decision_gt_repeat, 'electrode'])
    counts = selected.functional_cluster.value_counts().to_dict()
    assert counts == {'sustain': 14, 'motor': 12, 'sensory': 3, 'Unassigned': 2}
    assert set(original.electrode) <= set(selected.electrode) and len(original) == 29
    added = selected.loc[~selected.electrode.isin(original.electrode)]
    assert len(added) == 2 and added.functional_cluster.eq('Unassigned').all()
    assert set(waves.electrode) == set(table.electrode)
    assert not waves.duplicated(['electrode','phase','condition','time']).any()
    assert set(waves.condition) == {'Decision','Repeat'} and set(waves.phase) == set(PHASES)
    assert np.isfinite(waves.value).all(), 'Missing values need an explicit reviewed handling rule'
    assert waves.n_finite_trials.gt(0).all()
    traces = summarize_traces(table, waves, [('All eligible', table)]).rename(columns={'condition':'level'})
    assert traces.n_finite_electrodes.eq(187).all() and traces.n_selected_electrodes.eq(187).all()
    orders = {}
    for cluster in CLUSTERS:
        group = selected.loc[selected.functional_cluster.eq(cluster)]
        if cluster != 'Unassigned':
            order = old_rows.loc[old_rows.phase.eq('stimulus') & old_rows.functional_cluster.eq(cluster)].sort_values('row').electrode.tolist()
        else:
            first = waves.loc[waves.electrode.isin(group.electrode) & waves.baseline_mask].groupby('electrode').time.min()
            order = sorted(group.electrode, key=lambda site: (first.get(site, float('inf')), site))
        assert len(order) == len(group) and set(order) == set(group.electrode)
        orders[cluster] = order
    matrices, payload, audits, pooled = {}, {}, [], {}
    identities = sorted(table.electrode)
    subject_by_site = table.set_index('electrode').loc[identities, 'subject'].to_numpy()
    subjects = sorted(table.subject.unique())
    for phase in PHASES:
        sub = waves.loc[waves.phase.eq(phase) & waves.time.between(*XLIMS[phase])]
        times = np.sort(sub.time.unique())
        assert len(times) and times.min() >= XLIMS[phase][0] and times.max() <= XLIMS[phase][1]
        paired = {}
        for condition in ['Decision', 'Repeat']:
            pivot = sub.loc[sub.condition.eq(condition)].pivot(index='electrode', columns='time', values='value').reindex(
                index=identities, columns=times)
            values = pivot.to_numpy(float)
            assert values.shape == (187, len(times)) and np.isfinite(values).all()
            contributions = np.stack([values[subject_by_site == subject].sum(axis=0) * 35 / 187 for subject in subjects])
            trace = traces.loc[traces.phase.eq(phase) & traces.level.eq(condition)].sort_values('time')
            np.testing.assert_array_equal(times, trace.time.to_numpy())
            np.testing.assert_allclose(values.mean(axis=0), trace['mean'], rtol=0, atol=1e-14)
            np.testing.assert_allclose(contributions.mean(axis=0), trace['mean'], rtol=0, atol=1e-14)
            pooled[f'{phase}_{condition}'] = contributions
            paired[condition] = pivot
        pooled[f'{phase}_time'] = times
        payload[f'{phase}_time'] = times
        masks = sub.loc[sub.condition.eq('Decision')].pivot(index='electrode', columns='time', values='baseline_mask')
        for cluster in CLUSTERS:
            order = orders[cluster]
            decision = paired['Decision'].reindex(order).to_numpy(float)
            repeat = paired['Repeat'].reindex(order).to_numpy(float)
            mask = masks.reindex(index=order, columns=times).to_numpy(bool)
            difference = decision - repeat
            shown = np.where(mask, difference, np.nan)
            key = f'{phase}_{cluster}'
            matrices[(phase, cluster)] = shown
            payload[key] = shown
            payload[key + '_unmasked_difference'] = difference
            payload[key + '_decision_baseline_mask'] = mask
            payload[key + '_electrodes'] = np.asarray(order, dtype=str)
            for row, site in enumerate(order):
                audits.append(dict(phase=phase, functional_cluster=cluster, row=row, electrode=site,
                    n_paired_samples=len(times), n_displayed_bins=int(mask[row].sum()),
                    n_negative_numeric_bins=int((shown[row] < 0).sum()), n_above_point5=int((shown[row] > .5).sum())))
    assert sum(len(order) for order in orders.values()) == 31
    OUTPUT.mkdir(parents=True, exist_ok=True)
    traces.to_csv(OUTPUT / 'pooled_electrode_mean_sem.csv', index=False)
    selected.to_csv(OUTPUT / 'selected_electrodes.csv', index=False)
    added.to_csv(OUTPUT / 'new_vs_29_electrodes.csv', index=False)
    pd.DataFrame(audits).to_csv(OUTPUT / 'paired_sample_audit.csv', index=False)
    np.savez_compressed(OUTPUT / 'decision_minus_repeat_heatmap_matrices.npz', **payload)
    pooled['electrode'] = np.asarray(identities, dtype=str)
    pooled['subject'] = np.asarray(subjects, dtype=str)
    np.savez_compressed(OUTPUT / 'pooled_paired_subject_contributions.npz', **pooled)
    flow = [dict(stage='Eligible paired sample', n_sites=187, n_participants=35),
            dict(stage='Existing Decision > Repeat selection', n_sites=31, n_participants=18)]
    flow += [dict(stage=LABELS[c], n_sites=len(orders[c]),
                  n_participants=int(selected.loc[selected.functional_cluster.eq(c)].subject.nunique())) for c in CLUSTERS]
    pd.DataFrame(flow).to_csv(OUTPUT / 'cohort_flow.csv', index=False)
    audit = dict(passed=True, old_figure_used_legacy_cache=True, previous_upper_sites=123, previous_heatmap_sites=29,
        upper_sites=187, upper_participants=35, selected_sites=31, selected_participants=18,
        component_counts={c: len(orders[c]) for c in CLUSTERS}, added_sites=added[['subject','channel','functional_cluster']].to_dict('records'),
        old_29_retained=True, original_classified_row_order_preserved=True, native_waveforms_complete=True,
        upper_baseline_scope='All anatomically eligible native-good paired sites, without NMF or response gating',
        lower_selection_source=str(FULL / 'decision_gt_repeat_electrodes.csv'), per_electrode_tests_recomputed=False,
        raw_signed_differences_preserved=True, color_mapping=[0., .5], negatives_displayed_white=True,
        phase_windows=XLIMS, source_files=hashes)
    write(OUTPUT / 'cohort_audit.json', audit)
    print('COHORT AUDIT:', json.dumps(audit, indent=2), flush=True)
    return traces, matrices, orders, audits, hashes


def pooled_inference(traces):
    import numpy as np
    import pandas as pd
    from ieeg.calc import stats
    from ieeg.calc.fast import mean_diff
    from scripts.run_sound_image_selection import install_reviewed_runtime_compatibility
    from scripts.refresh_lexical_condition_sets import sha, write
    expected = sha(OUTPUT / 'pooled_paired_subject_contributions.npz')
    compatibility = install_reviewed_runtime_compatibility()
    time_tables, spans = [], []
    with np.load(OUTPUT / 'pooled_paired_subject_contributions.npz', allow_pickle=False) as data:
        for phase in PHASES:
            time = data[f'{phase}_time']
            decision, repeat = data[f'{phase}_Decision'], data[f'{phase}_Repeat']
            assert decision.shape == repeat.shape == (35, len(time))
            observed = mean_diff(decision, repeat, axis=0)
            paired_trace = traces.loc[traces.phase.eq(phase)].pivot(index='time', columns='level', values='mean').sort_index()
            np.testing.assert_allclose(observed, paired_trace.Decision - paired_trace.Repeat, rtol=0, atol=1e-14)
            print('187-site pooled time_perm_cluster:', phase, flush=True)
            mask, p_raw = stats.time_perm_cluster(decision, repeat, stat_func=mean_diff, **PARAMETERS)
            mask, p_raw = np.asarray(mask, bool), np.asarray(p_raw, float)
            assert not np.any(mask & (observed <= 0))
            assert mask.shape == p_raw.shape == observed.shape and np.isfinite(p_raw).all()
            time_tables.append(pd.DataFrame(dict(phase=phase, time=time, mask=mask, p_value=p_raw,
                mean_difference=observed, n_sites=187, n_subjects=35)))
            edges = np.r_[time[0] - (time[1] - time[0]) / 2, (time[:-1] + time[1:]) / 2,
                          time[-1] + (time[-1] - time[-2]) / 2]
            changes = np.diff(np.r_[False, mask, False].astype(int))
            for first, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
                spans.append(dict(phase=phase, start=max(float(edges[first]), XLIMS[phase][0]),
                    end=min(float(edges[end]), XLIMS[phase][1]), first_significant_time=float(time[first]),
                    last_significant_time=float(time[end - 1]), n_bins=int(end - first)))
    pd.concat(time_tables, ignore_index=True).to_csv(OUTPUT / 'pooled_timeperm.csv', index=False)
    intervals = pd.DataFrame(spans, columns=['phase','start','end','first_significant_time','last_significant_time','n_bins'])
    intervals.to_csv(OUTPUT / 'pooled_significant_intervals.csv', index=False)
    write(OUTPUT / 'pooled_inference_manifest.json', dict(parameters=PARAMETERS, stat_func='ieeg.calc.fast.mean_diff',
        n_sites=187, n_subjects=35, input_sha256=expected, runtime_compatibility=compatibility,
        exchangeability='Swap conditions coherently within subject; S/N-weighted subject electrode sums',
        correction_scope='Temporal clusters separately within each full displayed phase',
        legacy_123_site_significance_not_reused=True, installed_library_unchanged=True, intervals=spans))
    return intervals


def render():
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm'
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap
    from src.paths import save_svg
    from scripts.refresh_lexical_condition_sets import sha, write
    import sys
    for font in Path(sys.prefix, 'fonts').glob('arial*.ttf'):
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'svg.fonttype':'none','font.family':'sans-serif','font.sans-serif':['Arial'],'font.size':7})
    traces, matrices, orders, audits, hashes = prepare()
    intervals = pooled_inference(traces)
    legacy = json.loads((BASE / 'plot_validation.json').read_text())
    geometry = legacy['reference_geometry']
    cm, fontsize = 1 / 2.54, 7
    row_height_cm = legacy['actual_cm_per_electrode_row']
    row_heights = [geometry['trace_height_cm']] + [len(orders[c]) * row_height_cm for c in CLUSTERS]
    time_scale = geometry['phase_widths_cm'][0] / 1.1
    widths = [time_scale * (XLIMS[p][1] - XLIMS[p][0]) for p in PHASES]
    left, right, top, bottom = [geometry[k] for k in ['left_cm','right_cm','top_cm','bottom_cm']]
    gap_x, gap_y = geometry['gap_x_cm'], geometry['gap_y_cm']
    nrows=len(row_heights)
    n_upper=int(traces.n_finite_electrodes.iloc[0])
    n_subjects=int(traces.n_participants.iloc[0])
    n_lower=sum(map(len,orders.values()))
    w, h = left + sum(widths) + 2 * gap_x + right, top + sum(row_heights) + (nrows-1) * gap_y + bottom
    fig = plt.figure(figsize=(w * cm, h * cm))
    gs = fig.add_gridspec(nrows, 3, height_ratios=row_heights, width_ratios=widths,
        left=left/w, right=1-right/w, top=1-top/h, bottom=bottom/h,
        wspace=gap_x/np.mean(widths), hspace=gap_y/np.mean(row_heights))
    axes = np.empty((nrows, 3), dtype=object)
    condition_colors = {'Decision':'#6A5ACD','Repeat':'#000000'}
    ticks = {'stimulus':[-.5,0,.6], 'delay':[0,1], 'response':[-.5,0,1]}
    images = {}
    for j, phase in enumerate(PHASES):
        for row in range(nrows):
            axes[row,j] = fig.add_subplot(gs[row,j], sharex=axes[0,j] if row else None)
        ax = axes[0,j]
        for level in ['Decision','Repeat']:
            data = traces.loc[traces.phase.eq(phase) & traces.level.eq(level)].sort_values('time')
            t, m, se = data.time.to_numpy(), data['mean'].to_numpy(), data['sem'].to_numpy()
            ax.plot(t, m, color=condition_colors[level], lw=1, label=level)
            ax.fill_between(t, m-se, m+se, color=condition_colors[level], alpha=.18, linewidth=0)
        ax.axhline(0, color='0.2', ls='--', lw=.7)
        ax.axvline(0, color='0.2', ls='--', lw=.7)
        ax.set_xlim(*XLIMS[phase]); ax.set_ylim(*legacy['trace_ylim'])
        for span in intervals.loc[intervals.phase.eq(phase)].itertuples():
            ax.plot([span.start,span.end], [1.03,1.03], transform=ax.get_xaxis_transform(),
                color=condition_colors['Decision'], lw=1.5, solid_capstyle='butt', clip_on=False)
        ax.set_title(phase.capitalize(), fontsize=fontsize)
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(labelbottom=False,labelleft=False,left=False,labelsize=fontsize,width=.75,length=2)
        if j == 0:
            ax.text(.02,.96,f'n={n_upper}',transform=ax.transAxes,va='top',fontsize=fontsize)
        for i, cluster in enumerate(CLUSTERS):
            ax = axes[i+1,j]
            n, matrix = len(orders[cluster]), matrices[(phase,cluster)]
            cmap = LinearSegmentedColormap.from_list('positive_' + cluster, ['#FFFFFF',COLORS[cluster]])
            cmap.set_bad('white'); cmap.set_under('white')
            image = ax.imshow(matrix, extent=(*XLIMS[phase],0,n), origin='lower', aspect='auto',
                cmap=cmap, vmin=0., vmax=.5, interpolation='nearest', rasterized=True)
            images[(phase,cluster)] = image
            np.testing.assert_array_equal(np.asarray(image.get_array()), matrix)
            assert image.norm.vmin == 0 and image.norm.vmax == .5
            ax.axvline(0,color='0.2',ls='--',lw=.8)
            ax.set_xlim(*XLIMS[phase]); ax.set_ylim(0,n)
            ax.set_yticks([]); ax.set_xticks(ticks[phase], labels=[f'{t:g}' for t in ticks[phase]])
            ax.get_xticklabels()[0].set_ha('left'); ax.get_xticklabels()[-1].set_ha('right')
            ax.spines[['top','right','left']].set_visible(False)
            ax.tick_params(labelbottom=i==len(CLUSTERS)-1,bottom=i==len(CLUSTERS)-1,labelsize=fontsize,width=.75,length=2)
            if j == 0:
                ax.text(-.07,.5,f'{LABELS[cluster]} (n={n})',transform=ax.transAxes,
                    ha='right',va='center',fontsize=fontsize,color=COLORS[cluster])
    axes[0,-1].legend(frameon=False,fontsize=fontsize,loc='upper left',bbox_to_anchor=(1.02,1),borderaxespad=0)
    axes[-1,1].set_xlabel('Time (s)',fontsize=fontsize)
    fig.text(left/w,1-.09/h,f'Lexical Delay · sound · {n_subjects} participants',ha='left',va='top',fontsize=fontsize)
    fig.text(.01,.025,f'Heatmaps: Decision − Repeat · {n_lower} selected sites',ha='left',va='bottom',fontsize=6.5)
    fig.canvas.draw()
    for row in range(nrows):
        scales = [ax.get_position().width/np.diff(ax.get_xlim())[0] for ax in axes[row]]
        np.testing.assert_allclose(scales,scales[0],rtol=1e-12)
    for j in range(3):
        heights = [axes[i+1,j].get_position().height*h/len(orders[c]) for i,c in enumerate(CLUSTERS)]
        np.testing.assert_allclose(heights,row_height_cm,rtol=1e-12)
    for i, cluster in enumerate(CLUSTERS):
        box = axes[i+1,-1].get_position()
        cax = fig.add_axes([box.x1+.15/w,(box.y0+box.y1)/2,.75/w,.10/h])
        bar = fig.colorbar(images[(PHASES[-1],cluster)],cax=cax,orientation='horizontal',ticks=[0,.5])
        bar.ax.set_xticklabels(['0','0.5']); bar.ax.tick_params(labelsize=4.5,length=1.3,pad=1)
        bar.outline.set_visible(False); bar.ax.set_title('ΔHGA (z)',fontsize=5,pad=2)
    path = save_svg(fig,IMAGE,close=True)
    assert all(sha(p) == expected for p, expected in hashes.items())
    write(OUTPUT / 'plot_validation.json',dict(passed=True,n_trace_sites=n_upper,n_trace_subjects=n_subjects,
        n_heatmap_sites=n_lower,component_counts={c:len(orders[c]) for c in CLUSTERS},
        authoritative_source=str(FULL),new_per_electrode_tests=False,pooled_test_parameters=PARAMETERS,
        mean_and_sem_unit='bipolar electrode',phase_windows=XLIMS,trace_ylim=legacy['trace_ylim'],
        color_mapping=[0,.5],negative_numeric_differences_preserved=True,negatives_displayed_white=True,
        original_decision_baseline_display_rule_retained=True,cm_per_electrode_row=row_height_cm,
        physical_time_scale_equal=True,source_files=hashes,svg=str(path)))
    return path
