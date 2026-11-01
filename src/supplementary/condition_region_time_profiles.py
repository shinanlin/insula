"""Temporal portraits of the audited assigned PN21 and LD27 region sets."""
from pathlib import Path
import ast
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

ROOT=Path('/cwork/ns458/insula-paper')
OUT=ROOT/'results/fig3/time_profiles'
IMG=ROOT/'img/fig3'
NOTEBOOK=ROOT/'vizpub/fig3.ipynb'
REGIONS=ROOT/'results/fig3/regions'
CLUSTERS=('sensory','sustain','motor')
COLORS=dict(sensory='#2369BD',sustain='#C4A35A',motor='#A9373B')
LABELS=dict(sensory='Sensory',sustain='Sustained',motor='Motor')
PHASES=('stimulus','delay','response')
SPECS={
    'picture':dict(n=21,subjects=14,region='no_detected_difference',conditions=('sound','image'),
        metric='(Sound + Image) / 2',title='Picture Naming · no detected Stimulus difference',
        waves='results/supplementary/sound_image_test/full_input_audit_2026-10-28/waveforms.csv.gz',
        orders='results/supplementary/sound_image_test/responsive_pool_difference_masks_2026-10-28/stimulus_display_to1_2026-10-28/no_difference_sound_minus_image_cluster_masked_matrices.npz',
        geometry='results/supplementary/sound_image_test/plot_validation.json',
        xlims=dict(stimulus=(-.5,1.),delay=(0.,1.),response=(-.5,1.)),
        components=dict(sensory=4,sustain=8,motor=9)),
    'lexical':dict(n=27,subjects=16,region='first_only',conditions=('Decision','Repeat'),
        metric='Decision − Repeat',title='Lexical Delay · Delay-selected Decision > Repeat',
        waves='results/supplementary/fig3_lexical_delay_condition_sets/full_cohort_2026-10-27/waveforms.csv.gz',
        orders='results/supplementary/decision_repeat_test/difference_heatmap_full_cohort_2026-10-27/decision_minus_repeat_heatmap_matrices.npz',
        geometry='results/supplementary/decision_repeat_test/plot_validation.json',
        xlims=dict(stimulus=(-.5,.6),delay=(0.,1.),response=(-.5,1.)),
        components=dict(sensory=3,sustain=14,motor=10)),
}


def shared_ylim(waves,selected,spec):
    """Use the original saved P98/floor-.4 helper without editing its code."""
    path=ROOT/'src/figure3_legacy_hga.py'
    source=path.read_text()
    names={'rfig3_electrode_key','rfig3_compute_shared_ylim'}
    nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==names
    space=dict(np=np,pd=pd,rfig3_clusters=CLUSTERS)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path), 'exec'),space)
    merged=waves.merge(selected[['electrode','functional_cluster']],on='electrode',validate='many_to_one')
    limits=space['rfig3_compute_shared_ylim'](merged,phases=PHASES,compare_field='condition',
        levels=spec['conditions'],xlim_by_phase=spec['xlims'],clusters=CLUSTERS,ymax_floor=.4)
    return limits,dict(source=str(path),functions_copied_verbatim=True,
        function_sha256={n.name:hashlib.sha256(ast.get_source_segment(source,n).encode()).hexdigest() for n in nodes},limits=limits)


def prepare(key):
    from scripts.refresh_lexical_condition_sets import sha,write
    from src.supplementary.lexical_condition_figures import summarize_traces
    spec=SPECS[key]
    selected_path=REGIONS/key/f"{spec['region']}_electrodes.csv"
    selected=pd.read_csv(selected_path)
    assert len(selected)==spec['n'] and selected.subject.nunique()==spec['subjects']
    assert {c:int(selected.functional_cluster.eq(c).sum()) for c in CLUSTERS}==spec['components']
    audit_folder='picture_stimulus_responsive_union_2026-11-01' if key=='picture' else 'lexical_delay_responsive_union_2026-11-01'
    audit=ROOT/'results/audit'/audit_folder
    assert json.loads((audit/'completion.json').read_text())['passed']
    audited=pd.read_csv(audit/'all_eligible_electrodes_reconstructed.csv')
    if key=='picture':
        expected=audited.loc[audited.pool_mask_p05 & audited.exclusive_difference_class.eq('No detected difference')]
    else:expected=audited.loc[audited.response_union_mask_p05 & audited.decision_gt_repeat_rebuilt]
    from src.main_analysis import assigned_functional_sites
    assert set(assigned_functional_sites(expected).electrode)==set(selected.electrode)
    waves=pd.read_csv(ROOT/spec['waves'],float_precision='round_trip')
    waves=waves.loc[waves.electrode.isin(selected.electrode)].copy()
    assert set(waves.condition)==set(spec['conditions']) and set(waves.phase)==set(PHASES)
    assert not waves.duplicated(['electrode','phase','condition','time']).any()
    assert np.isfinite(waves.value).all() and waves.n_finite_trials.gt(0).all()
    waves=pd.concat([waves.loc[waves.phase.eq(p)&waves.time.between(*spec['xlims'][p])] for p in PHASES],ignore_index=True)
    traces=summarize_traces(selected,waves,[('Selected spatial region',selected)])
    assert traces.n_finite_electrodes.eq(spec['n']).all()
    ylim,ylim_proof=shared_ylim(waves,selected,spec)
    with np.load(ROOT/spec['orders'],allow_pickle=False) as prior:
        orders={c:[site for site in prior[f'stimulus_{c}_electrodes'].tolist() if site in set(selected.electrode)] for c in CLUSTERS}
    assert sum(map(len,orders.values()))==spec['n']
    for c in CLUSTERS:assert set(orders[c])==set(selected.loc[selected.functional_cluster.eq(c),'electrode'])
    payload={};matrices={};row_audit=[]
    for phase in PHASES:
        part=waves.loc[waves.phase.eq(phase)]
        times=np.sort(part.time.unique());payload[phase+'_time']=times
        pairs={level:part.loc[part.condition.eq(level)].pivot(index='electrode',columns='time',values='value') for level in spec['conditions']}
        masks=part.loc[part.condition.eq(spec['conditions'][0])].pivot(index='electrode',columns='time',values='baseline_mask')
        for c in CLUSTERS:
            order=orders[c];prefix=f'{phase}_{c}'
            first=pairs[spec['conditions'][0]].reindex(index=order,columns=times).to_numpy(float)
            second=pairs[spec['conditions'][1]].reindex(index=order,columns=times).to_numpy(float)
            assert np.isfinite(first).all() and np.isfinite(second).all()
            raw=(first+second)/2 if key=='picture' else first-second
            display_mask=np.ones(raw.shape,bool) if key=='picture' else masks.reindex(index=order,columns=times).to_numpy(bool)
            shown=np.where(display_mask,raw,np.nan)
            matrices[(phase,c)]=shown
            payload[prefix]=shown;payload[prefix+'_unmasked_values']=raw
            payload[prefix+'_display_mask']=display_mask;payload[prefix+'_first_condition']=first
            payload[prefix+'_second_condition']=second;payload[prefix+'_electrodes']=np.asarray(order,dtype=str)
            for i,site in enumerate(order):row_audit.append(dict(phase=phase,functional_cluster=c,row=i,electrode=site,
                n_paired_bins=len(times),n_displayed_bins=int(display_mask[i].sum()),
                n_negative_values=int((raw[i]<0).sum()),n_above_point5=int((raw[i]>.5).sum())))
    dest=OUT/key;dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'heatmap_matrices.npz',**payload)
    traces.to_csv(dest/'selected_electrode_mean_sem.csv',index=False)
    selected.to_csv(dest/'selected_electrodes.csv',index=False)
    pd.DataFrame(row_audit).to_csv(dest/'row_and_value_audit.csv',index=False)
    write(dest/'data_provenance.json',dict(passed=True,n_sites=spec['n'],n_participants=spec['subjects'],
        exact_same_sites_as_brain_map=True,metric=spec['metric'],component_counts=spec['components'],
        phase_windows=spec['xlims'],row_order_filtered_from_original=True,
        display_mask='No statistical blanking; complete paired condition average' if key=='picture' else 'Unchanged original Decision-baseline display mask',
        colored_bins_are_condition_difference_significance=False,mean_SEM_unit='electrode',
        no_new_tests=True,no_old_pooled_significance_bars_reused=True,
        color_limits=[0,.5],negative_values_preserved_and_displayed_white=True,
        original_shared_y_helper=ylim_proof,source_sha256={str(p):sha(p) for p in
            [selected_path,ROOT/spec['waves'],ROOT/spec['orders'],audit/'completion.json']}))
    return spec,selected,traces,orders,matrices,payload,ylim


def render(key):
    assert os.environ.get('SLURM_JOB_ID'),'Use Slurm'
    from src.paths import save_svg
    from scripts.refresh_lexical_condition_sets import write,sha
    spec,selected,traces,orders,matrices,payload,ylim=prepare(key)
    for font in Path(sys.prefix,'fonts').glob('arial*.ttf'):font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'svg.fonttype':'none','font.family':'Arial','font.size':7})
    previous=json.loads((ROOT/spec['geometry']).read_text())
    g=previous['reference_geometry'];row_height=previous['actual_cm_per_electrode_row']
    time_scale=g['phase_widths_cm'][0]/1.1
    widths=[time_scale*(spec['xlims'][p][1]-spec['xlims'][p][0]) for p in PHASES]
    heights=[g['trace_height_cm']]+[len(orders[c])*row_height for c in CLUSTERS]
    left,right,top,bottom=[g[k] for k in ('left_cm','right_cm','top_cm','bottom_cm')]
    gx,gy=g['gap_x_cm'],g['gap_y_cm']
    nrows=len(heights)
    w,h=left+sum(widths)+2*gx+right,top+sum(heights)+(nrows-1)*gy+bottom
    fig=plt.figure(figsize=(w/2.54,h/2.54))
    grid=fig.add_gridspec(nrows,3,height_ratios=heights,width_ratios=widths,left=left/w,right=1-right/w,
        top=1-top/h,bottom=bottom/h,wspace=gx/np.mean(widths),hspace=gy/np.mean(heights))
    axes=np.empty((nrows,3),object);images={}
    colors={spec['conditions'][0]:'#6A5ACD',spec['conditions'][1]:'#000000'}
    for j,phase in enumerate(PHASES):
        for i in range(nrows):axes[i,j]=fig.add_subplot(grid[i,j],sharex=axes[0,j] if i else None)
        ax=axes[0,j]
        for condition in spec['conditions']:
            frame=traces.loc[traces.phase.eq(phase)&traces.condition.eq(condition)].sort_values('time')
            t,m,se=frame.time.to_numpy(),frame['mean'].to_numpy(),frame['sem'].to_numpy()
            ax.plot(t,m,color=colors[condition],lw=1,label=condition.capitalize())
            ax.fill_between(t,m-se,m+se,color=colors[condition],alpha=.18,linewidth=0)
        ax.axhline(0,color='.2',ls='--',lw=.7);ax.axvline(0,color='.2',ls='--',lw=.7)
        ax.set_xlim(*spec['xlims'][phase]);ax.set_ylim(*ylim)
        ax.set_title(phase.capitalize(),fontsize=7)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelbottom=False,left=j==0,labelleft=j==0,width=.75,length=2)
        if j==0:
            ax.set_ylabel('HGA (z)',fontsize=7);ax.text(.02,.96,f"n={spec['n']}",transform=ax.transAxes,va='top',fontsize=7)
        else:ax.spines['left'].set_visible(False)
        lower,upper=spec['xlims'][phase]
        ticks=[lower,0,upper] if lower<0 else [0,upper]
        for i,c in enumerate(CLUSTERS):
            ax=axes[i+1,j];n=len(orders[c]);matrix=matrices[(phase,c)]
            cmap=LinearSegmentedColormap.from_list('positive_'+c,['#FFFFFF',COLORS[c]])
            cmap.set_bad('white');cmap.set_under('white')
            im=ax.imshow(matrix,extent=(lower,upper,0,n),origin='lower',aspect='auto',cmap=cmap,
                vmin=0,vmax=.5,interpolation='nearest',rasterized=True)
            images[(phase,c)]=im
            np.testing.assert_array_equal(np.asarray(im.get_array()),matrix)
            ax.axvline(0,color='.2',ls='--',lw=.8);ax.set_xlim(lower,upper);ax.set_ylim(0,n);ax.set_yticks([])
            ax.set_xticks(ticks,labels=[f'{v:g}' for v in ticks])
            ax.get_xticklabels()[0].set_ha('left');ax.get_xticklabels()[-1].set_ha('right')
            ax.spines[['top','right','left']].set_visible(False)
            ax.tick_params(labelbottom=i==len(CLUSTERS)-1,bottom=i==len(CLUSTERS)-1,labelsize=7,width=.75,length=2)
            if j==0:ax.text(-.07,.5,f'{LABELS[c]} (n={n})',transform=ax.transAxes,ha='right',va='center',fontsize=7,color=COLORS[c])
    axes[0,-1].legend(frameon=False,fontsize=7,loc='upper left',bbox_to_anchor=(1.02,1),borderaxespad=0)
    axes[-1,1].set_xlabel('Time (s)',fontsize=7)
    fig.text(left/w,1-.09/h,spec['title'],ha='left',va='top',fontsize=7)
    description='Complete paired condition means; no difference mask' if key=='picture' else 'Original Decision-baseline display mask; color denotes difference magnitude'
    caption=fig.text(.01,.025,f"Heatmap: {spec['metric']} HGA (z) · n={spec['n']}\n{description}",
        ha='left',va='bottom',fontsize=6.2,linespacing=1.35)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    bottom_labels=[v for ax in axes[-1] for v in ax.get_xticklabels()]+[axes[-1,1].xaxis.label]
    lowest=min(v.get_window_extent(renderer).y0 for v in bottom_labels)
    caption.set_position((.01,fig.transFigure.inverted().transform((0,lowest-3*fig.dpi/72))[1]));caption.set_verticalalignment('top')
    for i,c in enumerate(CLUSTERS):
        box=axes[i+1,-1].get_position()
        cax=fig.add_axes([box.x1+.15/w,(box.y0+box.y1)/2,.75/w,.10/h])
        bar=fig.colorbar(images[(PHASES[-1],c)],cax=cax,orientation='horizontal',ticks=[0,.5])
        bar.ax.set_xticklabels(['0','0.5']);bar.ax.tick_params(labelsize=4.5,length=1.3,pad=1);bar.outline.set_visible(False)
    for row in axes:
        scales=[ax.get_position().width/np.diff(ax.get_xlim())[0] for ax in row]
        np.testing.assert_allclose(scales,scales[0],rtol=1e-12)
    for j in range(3):
        physical=[axes[i+1,j].get_position().height*h/len(orders[c]) for i,c in enumerate(CLUSTERS)]
        np.testing.assert_allclose(physical,row_height,rtol=1e-12)
    IMG.mkdir(parents=True,exist_ok=True)
    filename='fig3_picture_no_difference_average_time.svg' if key=='picture' else 'fig3_lexical_decision_gt_repeat_difference_time.svg'
    path=save_svg(fig,IMG/filename,close=True)
    write(OUT/key/'plot_validation.json',dict(passed=True,svg=str(path),svg_sha256=sha(path),
        n_trace_sites=spec['n'],n_heatmap_sites=spec['n'],n_participants=spec['subjects'],
        metric=spec['metric'],no_new_inference=True,no_other_population_bars_reused=True,
        row_height_cm=row_height,phase_widths_cm=widths,physical_time_scale_equal=True,
        phase_windows=spec['xlims'],color_limits=[0,.5],ylim=ylim,component_palette=COLORS))
    return path


def notebook_cells(key):
    import nbformat as nbf
    spec=SPECS[key]
    detail='Each heatmap value is the electrode-wise (Sound + Image)/2 of unmasked native condition means. Equal condition weights; no row normalization or condition-difference blanking. This is a descriptive condition average, not proof of equivalence or joint baseline activation.' if key=='picture' else 'Each heatmap value is the paired Decision minus Repeat native HGA mean, retaining the existing Decision-baseline display-blanking convention. Colors show signed difference magnitude through the accepted positive 0–.5 scale, not a new per-time condition-difference significance mask. Negative values remain saved and display white.'
    note=f'''## Temporal portrait · {spec['n']} sites / {spec['subjects']} participants

These are exactly the electrodes in the corresponding functional-color brain map: **{spec['title']}**. Upper curves show both conditions for this same selected set, with electrode-equal mean ± SEM. No 49-site/187-site pooled significance bars are reused and no new tests are run.

{detail}

Retain Stimulus, Delay and Response, original display windows for each comparison, time-proportional widths, original 1-mm electrode rows, component colors and fixed 0–.5 heatmap limits. Original row-order subsets are preserved. All raw signed values, display masks, paired source means and memberships are saved separately. Existing figures, tests and NMF remain unchanged.
'''
    return [nbf.v4.new_markdown_cell(note,id=f'condition-time-{key}-note'),
        nbf.v4.new_code_cell('from src.supplementary.condition_region_time_profiles import render as render_region_time\n'
            f"display(SVG(filename=str(render_region_time('{key}'))))\n",id=f'condition-time-{key}-plot')]
