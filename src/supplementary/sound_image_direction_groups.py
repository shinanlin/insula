"""Saved PN directional selections and their complementary no-difference set.

This display reads existing tests, curves and native condition means only.
It does not fit NMF or perform additional statistical inference.
"""
import json
import os
from pathlib import Path

from src.supplementary import sound_image_full_difference as reference

ROOT = reference.ROOT
BASE = reference.BASE
FULL = reference.FULL
OUTPUT = BASE / 'direction_groups_2026-10-28'
AUDIT = ROOT / 'results/audit/sound_image_direction_groups_2026-10-28'
NOTEBOOK = reference.NOTEBOOK
IMAGE_DIR = reference.IMAGE.parent
PHASES, XLIMS = reference.PHASES, reference.XLIMS
CLUSTERS, LABELS, COLORS = reference.CLUSTERS, reference.LABELS, reference.COLORS
SPECS = {
    'image_gt_sound': dict(title='Image > Sound', positive='image', negative='sound',
        selection='image_gt_sound', n=12, participants=7),
    'no_difference_sound_minus_image': dict(title='No detected difference', positive='sound', negative='image',
        selection='no_difference', n=93),
    'no_difference_image_minus_sound': dict(title='No detected difference', positive='image', negative='sound',
        selection='no_difference', n=93),
}


def notebook_cells():
    import nbformat as nbf
    intro = '''## Both directions and the tested no-difference complement

Use the **same complete 128 paired insula sites / 27 participants** and the **existing per-electrode Stimulus [0,1] s tests**. The preceding Sound > Image figure remains intact. Add Image > Sound and two directional views of the sites with neither detected direction. These are direct condition-comparison selections, not Sound-only/Image-only baseline-response sets.

- Sound > Image: 24 sites / 13 participants (preceding figure).
- Image > Sound: 12 sites / 7 participants.
- Neither direction detected: 93 sites; participant and component counts are shown below.
- One site has significant differences in **both directions at different times**, so the two directional lists overlap. The exclusive partition is 23 Sound-direction only + 11 Image-direction only + 1 bidirectional + 93 without a detected difference = 128.

**Upper curves and purple bars are unchanged in every panel:** all 128 sites, electrode-equal Sound/Image mean ± SEM, with the previously computed full-cohort **Sound > Image** pooled bars. They do not describe only the selected lower rows; no new subgroup or reverse-direction pooled test is performed.

**Lower heatmaps:** Image − Sound for the Image > Sound set; both Sound − Image and Image − Sound as separate views of the **same 93 no-difference sites**. Subtract the unmasked means first, then apply the corresponding positive-condition native baseline display mask (Image or Sound). Preserve component hues, explicit gray Unassigned, 1-mm rows, original row order, common y-axis limits and time-proportional widths. The color range remains **0–0.5**; negative values appear white and signed unmasked values are saved. These colors are descriptive HGA differences, not per-time difference-significance masks. As explicitly approved, place the repeated HGA (z) unit once in the caption to prevent overlap in the one-/two-electrode rows; every component retains its own colorbar and 0/0.5 ticks.

Selection uses **Stimulus 0–1 s**, whereas display windows remain **Stimulus −0.5–0.6, Delay 0–1, Response −0.5–1 s**. “No detected difference” means neither direction passed the existing test; it is **not evidence of equivalence**, and does not classify the other displayed phases. Main figures, NMF, existing test results, prior Sound > Image SVG, Venn/brain figures and Lexical outputs are preserved.
'''
    cells = [nbf.v4.new_markdown_cell(intro, id='sound-direction-groups-scope')]
    for key, spec in SPECS.items():
        delta = f"{spec['positive'].capitalize()} − {spec['negative'].capitalize()}"
        cells.append(nbf.v4.new_markdown_cell(
            f"### {spec['title']} · lower rows: {delta}\n\nUpper: unchanged full-cohort curves and Sound > Image bars.",
            id='sound-direction-heading-' + key.replace('_', '-')))
        cells.append(nbf.v4.new_code_cell(
            'from src.supplementary.sound_image_direction_groups import render, OUTPUT\n'
            f"display(SVG(filename=str(render('{key}'))))\n",
            id='sound-direction-plot-' + key.replace('_', '-')))
    cells.append(nbf.v4.new_code_cell(
        "display(pd.read_csv(OUTPUT/'group_counts.csv'))\n"
        "display(pd.read_csv(OUTPUT/'exclusive_partition.csv'))\n"
        "display(pd.read_csv(OUTPUT/'bidirectional_sites.csv')[['subject','channel','sound_gt_image_intervals','image_gt_sound_intervals']])\n"
        "print('Complete electrode lists and signed matrices:', OUTPUT)\n",
        id='sound-direction-group-audit'))
    return cells


def prepare():
    import numpy as np
    import pandas as pd
    from scripts.refresh_lexical_condition_sets import sha, write

    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm'
    done = json.loads((ROOT/'results/audit/sound_image_full_input_2026-10-28/completion.json').read_text())
    assert done['passed'] and done['complete_cells'] == 768
    assert all(sha(reference.INPUT/name) == h for name, h in done['outputs'].items())
    cohort_proof = json.loads((reference.OUTPUT/'cohort_audit.json').read_text())
    assert all(sha(path) == h for path, h in cohort_proof['input_files'].items())
    pooled_proof = json.loads((reference.OUTPUT/'pooled_inference_manifest.json').read_text())
    assert all(sha(reference.OUTPUT/name) == h for name, h in pooled_proof['outputs'].items())
    assert sha(reference.OUTPUT/'pooled_paired_subject_contributions.npz') == pooled_proof['input_sha256']
    table = pd.read_csv(FULL/'electrodes.csv')
    assert len(table) == 128 and table.subject.nunique() == 27
    assert table.eligible.all() and table.paired_statistics_complete.all() and table.electrode.is_unique
    sound = table.sound_gt_image.astype(bool)
    image = table.image_gt_sound.astype(bool)
    groups = {'sound_gt_image': table.loc[sound].copy(), 'image_gt_sound': table.loc[image].copy(),
              'no_difference': table.loc[~sound & ~image].copy()}
    assert [len(groups[k]) for k in groups] == [24, 12, 93]
    for key in ('sound_gt_image', 'image_gt_sound'):
        saved = pd.read_csv(FULL/f'{key}_electrodes.csv')
        assert set(saved.electrode) == set(groups[key].electrode)
    both = table.loc[sound & image].copy()
    assert len(both) == 1
    assert len(set(groups['sound_gt_image'].electrode) | set(groups['image_gt_sound'].electrode)) == 35
    assert set(groups['no_difference'].electrode).isdisjoint(
        set(groups['sound_gt_image'].electrode) | set(groups['image_gt_sound'].electrode))
    assert sum([int((sound & ~image).sum()), int((image & ~sound).sum()), len(both), len(groups['no_difference'])]) == 128
    waves = pd.read_csv(reference.INPUT/'waveforms.csv.gz', float_precision='round_trip')
    shown = waves.loc[np.logical_or.reduce([
        waves.phase.eq(p) & waves.time.between(*XLIMS[p]) for p in PHASES])].copy()
    assert not shown.duplicated(['electrode', 'phase', 'condition', 'time']).any()
    assert np.isfinite(shown.value).all()
    traces = pd.read_csv(reference.OUTPUT/'pooled_electrode_mean_sem.csv', float_precision='round_trip')
    assert traces.n_finite_electrodes.eq(128).all()
    intervals = pd.read_csv(reference.OUTPUT/'pooled_significant_intervals.csv', float_precision='round_trip')
    ylim = cohort_proof['original_shared_y_helper']['limits']
    old_order = pd.read_csv(BASE/'frozen_heatmap_row_membership.csv')
    orders = {}
    for key, selected in groups.items():
        orders[key] = {}
        for cluster in CLUSTERS:
            sites = set(selected.loc[selected.functional_cluster.eq(cluster), 'electrode'])
            if cluster != 'Unassigned':
                original = old_order.loc[old_order.phase.eq('stimulus') & old_order.functional_cluster.eq(cluster)].sort_values('row').electrode
                order = [site for site in original if site in sites]
            else:
                first = shown.loc[shown.electrode.isin(sites) & shown.baseline_mask].groupby('electrode').time.min()
                order = sorted(sites, key=lambda site: (first.get(site, float('inf')), site))
            assert len(order) == len(sites) and set(order) == sites
            orders[key][cluster] = order
    # Reproduce the already accepted lower Sound > Image data exactly, without
    # rewriting the original matrices, results, SVG or notebook outputs.
    with np.load(reference.OUTPUT/'sound_minus_image_heatmap_matrices.npz', allow_pickle=False) as saved:
        for phase in PHASES:
            sub = shown.loc[shown.phase.eq(phase)]
            times = np.sort(sub.time.unique())
            np.testing.assert_array_equal(times, saved[f'{phase}_time'])
            values = {level: sub.loc[sub.condition.eq(level)].pivot(index='electrode', columns='time', values='value')
                      for level in ('sound', 'image')}
            masks = sub.loc[sub.condition.eq('sound')].pivot(index='electrode', columns='time', values='baseline_mask')
            for cluster in CLUSTERS:
                key = f'{phase}_{cluster}'
                order = orders['sound_gt_image'][cluster]
                np.testing.assert_array_equal(np.asarray(order), saved[key+'_electrodes'])
                delta = (values['sound']-values['image']).reindex(index=order, columns=times).to_numpy(float)
                np.testing.assert_array_equal(delta, saved[key+'_unmasked_difference'])
                mask = masks.reindex(index=order, columns=times).to_numpy(bool)
                np.testing.assert_array_equal(np.where(mask, delta, np.nan), saved[key])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    counts = []
    for key, selected in groups.items():
        selected.to_csv(OUTPUT/f'{key}_electrodes.csv', index=False)
        counts.append(dict(group=key, component='All', n_sites=len(selected), n_participants=selected.subject.nunique()))
        counts.extend(dict(group=key, component=c, n_sites=len(orders[key][c]),
                           n_participants=selected.loc[selected.functional_cluster.eq(c)].subject.nunique()) for c in CLUSTERS)
    pd.DataFrame(counts).to_csv(OUTPUT/'group_counts.csv', index=False)
    both.to_csv(OUTPUT/'bidirectional_sites.csv', index=False)
    partition = [('Sound direction only', table.loc[sound & ~image]),
                 ('Image direction only', table.loc[image & ~sound]), ('Both directions', both),
                 ('Neither direction detected', groups['no_difference'])]
    pd.DataFrame([dict(group=k, n_sites=len(t), n_participants=t.subject.nunique()) for k,t in partition]).to_csv(
        OUTPUT/'exclusive_partition.csv', index=False)
    write(OUTPUT/'cohort_audit.json', dict(passed=True, n_tested=128, n_participants=27,
        groups={k:dict(n_sites=len(t), n_participants=int(t.subject.nunique()),
                       component_counts={c:len(orders[k][c]) for c in CLUSTERS}) for k,t in groups.items()},
        overlap_sites=both.electrode.tolist(), exclusive_partition_total=128, selection_window=[0,1],
        selection='Existing two-sided per-electrode Stimulus tests; neither direction is the tested complement',
        no_new_inference=True, no_new_NMF=True, original_sound_difference_matrices_reproduced_exactly=True,
        upper_curves_bars_and_ylim='Read original full 128-site outputs unchanged for all panels',
        no_difference_not_equivalence=True))
    return groups, shown, traces, intervals, orders, ylim


def render(key):
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm'
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap
    import sys
    from src.paths import save_svg
    from scripts.refresh_lexical_condition_sets import write

    spec = SPECS[key]
    groups, shown, traces, intervals, all_orders, ylim = prepare()
    group, orders = groups[spec['selection']], all_orders[spec['selection']]
    assert len(group) == spec['n']
    if 'participants' in spec:
        assert group.subject.nunique() == spec['participants']
    matrices, payload = {}, {}
    for phase in PHASES:
        sub = shown.loc[shown.phase.eq(phase)]
        times = np.sort(sub.time.unique())
        values = {level: sub.loc[sub.condition.eq(level)].pivot(index='electrode', columns='time', values='value')
                  for level in ('sound', 'image')}
        masks = sub.loc[sub.condition.eq(spec['positive'])].pivot(index='electrode', columns='time', values='baseline_mask')
        payload[f'{phase}_time'] = times
        for cluster in CLUSTERS:
            order = orders[cluster]
            delta = (values[spec['positive']]-values[spec['negative']]).reindex(index=order, columns=times).to_numpy(float)
            mask = masks.reindex(index=order, columns=times).to_numpy(bool)
            assert np.isfinite(delta).all()
            displayed = np.where(mask, delta, np.nan)
            matrices[(phase, cluster)] = displayed
            prefix = f'{phase}_{cluster}'
            payload[prefix] = displayed
            payload[prefix+'_unmasked_difference'] = delta
            payload[prefix+'_positive_condition_baseline_mask'] = mask
            payload[prefix+'_electrodes'] = np.asarray(order, dtype=str)
    matrix_path = OUTPUT/f'{key}_matrices.npz'
    if matrix_path.exists():
        with np.load(matrix_path,allow_pickle=False) as cached:
            assert set(cached.files)==set(payload)
            for name,value in payload.items():
                np.testing.assert_array_equal(cached[name],value)
    else:
        np.savez_compressed(matrix_path, **payload)
    for font in Path(sys.prefix, 'fonts').glob('arial*.ttf'):
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'svg.fonttype':'none', 'font.family':'sans-serif', 'font.sans-serif':['Arial'], 'font.size':7})
    legacy = json.loads((BASE/'plot_validation.json').read_text())
    geometry = legacy['reference_geometry']
    row_height = legacy['actual_cm_per_electrode_row']
    active = [c for c in CLUSTERS if len(orders[c])]
    row_heights = [geometry['trace_height_cm']] + [len(orders[c])*row_height for c in active]
    widths = geometry['phase_widths_cm']
    left,right,top,bottom = [geometry[k] for k in ['left_cm','right_cm','top_cm','bottom_cm']]
    gap_x,gap_y = geometry['gap_x_cm'],geometry['gap_y_cm']
    w,h = left+sum(widths)+2*gap_x+right, top+sum(row_heights)+len(active)*gap_y+bottom
    fig = plt.figure(figsize=(w/2.54, h/2.54))
    gs = fig.add_gridspec(len(row_heights),3,height_ratios=row_heights,width_ratios=widths,
        left=left/w,right=1-right/w,top=1-top/h,bottom=bottom/h,
        wspace=gap_x/np.mean(widths),hspace=gap_y/np.mean(row_heights))
    axes = np.empty((len(row_heights),3),dtype=object)
    trace_colors = {'sound':'#6A5ACD', 'image':'#000000'}
    ticks = {'stimulus':[-.5,0,.6], 'delay':[0,1], 'response':[-.5,0,1]}
    images = {}
    for j,phase in enumerate(PHASES):
        for row in range(len(row_heights)):
            axes[row,j] = fig.add_subplot(gs[row,j],sharex=axes[0,j] if row else None)
        ax = axes[0,j]
        for level in ['sound','image']:
            data = traces.loc[traces.phase.eq(phase)&traces.level.eq(level)].sort_values('time')
            t,m,se = data.time.to_numpy(),data['mean'].to_numpy(),data['sem'].to_numpy()
            ax.plot(t,m,color=trace_colors[level],lw=1,label=level.capitalize())
            ax.fill_between(t,m-se,m+se,color=trace_colors[level],alpha=.18,linewidth=0)
        ax.axhline(0,color='0.2',ls='--',lw=.7);ax.axvline(0,color='0.2',ls='--',lw=.7)
        ax.set_xlim(*XLIMS[phase]);ax.set_ylim(*ylim)
        for span in intervals.loc[intervals.phase.eq(phase)].itertuples():
            ax.plot([span.start,span.end],[1.03,1.03],transform=ax.get_xaxis_transform(),
                color=trace_colors['sound'],lw=1.5,solid_capstyle='butt',clip_on=False)
        ax.set_title(phase.capitalize(),fontsize=7)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelbottom=False,labelleft=j==0,left=j==0,labelsize=7,width=.75,length=2)
        if j==0:
            ax.set_ylabel('HGA (z)',fontsize=7)
            ax.text(.02,.96,'n=128',transform=ax.transAxes,va='top',fontsize=7)
        else:
            ax.spines['left'].set_visible(False)
        for i,cluster in enumerate(active):
            ax = axes[i+1,j]
            n,matrix = len(orders[cluster]),matrices[(phase,cluster)]
            cmap = LinearSegmentedColormap.from_list('positive_'+cluster,['#FFFFFF',COLORS[cluster]])
            cmap.set_bad('white');cmap.set_under('white')
            im = ax.imshow(matrix,extent=(*XLIMS[phase],0,n),origin='lower',aspect='auto',
                cmap=cmap,vmin=0,vmax=.5,interpolation='nearest',rasterized=True)
            images[(phase,cluster)] = im
            np.testing.assert_array_equal(np.asarray(im.get_array()),matrix)
            ax.axvline(0,color='0.2',ls='--',lw=.8)
            ax.set_xlim(*XLIMS[phase]);ax.set_ylim(0,n);ax.set_yticks([])
            ax.set_xticks(ticks[phase],labels=[f'{t:g}' for t in ticks[phase]])
            ax.get_xticklabels()[0].set_ha('left');ax.get_xticklabels()[-1].set_ha('right')
            ax.spines[['top','right','left']].set_visible(False)
            ax.tick_params(labelbottom=i==len(active)-1,bottom=i==len(active)-1,labelsize=7,width=.75,length=2)
            if j==0:
                ax.text(-.07,.5,f'{LABELS[cluster]} (n={n})',transform=ax.transAxes,
                    ha='right',va='center',fontsize=7,color=COLORS[cluster])
    axes[0,-1].legend(frameon=False,fontsize=7,loc='upper left',bbox_to_anchor=(1.02,1),borderaxespad=0)
    axes[-1,1].set_xlabel('Time (s)',fontsize=7)
    fig.text(left/w,1-.09/h,'Picture Naming · upper: full cohort',ha='left',va='top',fontsize=7)
    metric = f"{spec['positive'].capitalize()} − {spec['negative'].capitalize()}"
    footer = (f"Lower: {spec['title']} · {len(group)} sites / {group.subject.nunique()} participants\n"
              f"Heatmaps: {metric} HGA (z) · selection: Stimulus 0–1 s\n"
              'Upper bars: Sound > Image, full cohort')
    caption = fig.text(.01,.025,footer,ha='left',va='bottom',fontsize=6.5,linespacing=1.35)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    time_labels = [label for ax in axes[-1] for label in ax.get_xticklabels()] + [axes[-1,1].xaxis.label]
    lowest = min(label.get_window_extent(renderer).y0 for label in time_labels)
    caption.set_position((.01,fig.transFigure.inverted().transform((0,lowest-3*fig.dpi/72))[1]))
    caption.set_verticalalignment('top')
    for row in axes:
        scales = [ax.get_position().width/np.diff(ax.get_xlim())[0] for ax in row]
        np.testing.assert_allclose(scales,scales[0],rtol=1e-12)
    for j in range(3):
        heights = [axes[i+1,j].get_position().height*h/len(orders[c]) for i,c in enumerate(active)]
        np.testing.assert_allclose(heights,row_height,rtol=1e-12)
    for i,cluster in enumerate(active):
        box = axes[i+1,-1].get_position()
        cax = fig.add_axes([box.x1+.15/w,(box.y0+box.y1)/2,.75/w,.10/h])
        bar = fig.colorbar(images[(PHASES[-1],cluster)],cax=cax,orientation='horizontal',ticks=[0,.5])
        bar.ax.set_xticklabels(['0','0.5']);bar.ax.tick_params(labelsize=4.5,length=1.3,pad=1)
        bar.outline.set_visible(False)
    path = save_svg(fig,IMAGE_DIR/f'fig3_{key}_difference.svg',close=True)
    write(OUTPUT/f'{key}_plot_validation.json',dict(passed=True,svg=str(path),n_upper=128,n_upper_participants=27,
        n_lower=len(group),n_lower_participants=int(group.subject.nunique()),
        component_counts={c:len(orders[c]) for c in CLUSTERS},selection_window=[0,1],display_windows=XLIMS,
        metric=metric,color_mapping=[0,.5],positive_condition_baseline_mask=spec['positive'],
        negative_numeric_values_preserved=True,negative_values_display_white=True,
        upper_curves_and_sound_gt_image_bars_unchanged=True,no_new_inference=True,no_new_NMF=True,
        shared_ylim=ylim,cm_per_electrode_row=row_height,physical_time_scale_equal=True,
        author_approved_shared_colorbar_unit_in_caption=True))
    print(f'RENDERED {key}: {len(group)} lower / {group.subject.nunique()} participants',flush=True)
    return path
