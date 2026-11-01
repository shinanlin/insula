"""PN difference heatmaps gated only by their own trial-level temporal tests."""
from pathlib import Path
import json
import os

from src.supplementary import sound_image_full_difference as original

ROOT,BASE,FULL,INPUT=original.ROOT,original.BASE,original.FULL,original.INPUT
OUT=BASE/'responsive_pool_difference_masks_2026-10-28'
DISPLAY_OUT=OUT/'stimulus_display_to1_2026-10-28'
AUDIT=ROOT/'results/audit/sound_image_responsive_pool_2026-10-28'
TEST_OUT=BASE/'condition_difference_masks_2026-10-28'
TEST_AUDIT=ROOT/'results/audit/sound_image_condition_difference_masks_2026-10-28'
NOTEBOOK=original.NOTEBOOK
PHASES=original.PHASES
XLIMS={'stimulus':(-.5,1.),'delay':(0.,1.),'response':(-.5,1.)}
CLUSTERS,LABELS,COLORS=original.CLUSTERS,original.LABELS,original.COLORS
SPECS={
    'sound_gt_image':dict(selection='sound_gt_image',direction='sound_gt_image',positive='sound',negative='image',
        title='Stimulus-selected Sound > Image',n=20,svg=original.IMAGE.name),
    'image_gt_sound':dict(selection='image_gt_sound',direction='image_gt_sound',positive='image',negative='sound',
        title='Stimulus-selected Image > Sound',n=5,svg='fig3_image_gt_sound_difference.svg'),
    'no_difference_sound_minus_image':dict(selection='no_difference',direction='sound_gt_image',positive='sound',negative='image',
        title='Responsive, no detected Stimulus difference',n=25,svg='fig3_no_difference_sound_minus_image_difference.svg'),
    'no_difference_image_minus_sound':dict(selection='no_difference',direction='image_gt_sound',positive='image',negative='sound',
        title='Responsive, no detected Stimulus difference',n=25,svg='fig3_no_difference_image_minus_sound_difference.svg'),
}


def pool_manifest():
    info=json.loads((OUT/'pool_manifest.json').read_text())
    assert info['passed'] and info['n_pool']==49 and info['n_pool_subjects']==22
    return info


def notebook_cells():
    import nbformat as nbf
    intro='''# Picture Naming: trial-level, cluster-corrected Sound/Image differences

**Corrected comparison pool: 49 Stimulus-responsive insula sites / 22 participants.** Start from the accepted 128 paired anatomical sites, then retain only those with a native Sound OR Image baseline-response mask hit in Stimulus [0,1] s: Sound-only 25 + Both 13 + Image-only 11. The other 79 sites are excluded from this comparison pool, including its upper curves. This baseline-response gate defines the pool; it is never the heatmap mask.

For each electrode, reuse the original native Repeat Sound versus Image trial-level iEEG `time_perm_cluster`: default t statistic, independent trials, 2,000 permutations, two-sided, `p_thresh=0.05`, `p_cluster=0.05`, seed42. Trial/QC rules, repaired Stimulus sources and reviewed process-local NumPy fix remain unchanged. There is no NMF gate or new per-electrode test.

**Within those 49 responsive sites, select using Stimulus [0,1] s:** Sound > Image 20 sites; Image > Sound 5; neither detected direction 25. Participant/component counts and lists are exported below. One site has opposite effects at different times; the exclusive partition is 19 Sound-direction only + 4 Image-direction only + 1 bidirectional + 25 without a detected difference = 49. These direct comparison sets are distinct from the 25/13/11 baseline-response Venn sets. All existing trial-level results and masks are reused exactly.

**Every colored heatmap bin must pass its own electrode's condition-difference test in that phase.** Delay [0,1] s and Response [−0.5,1] s have now been tested separately with the same method. Each electrode×complete phase is its own original temporal correction family. Direction-specific masks are the returned two-sided corrected temporal mask AND observed data AND original pointwise p<.05 AND the corresponding mean-difference sign. Sound > Image shows Sound − Image; Image > Sound shows Image − Sound. **No baseline-response mask is used.** A blank bin is not a colored significant difference; no mask is extrapolated between phases or to untested times.

The 25 responsive sites without a detected Stimulus difference have blank Stimulus difference heatmaps by definition. They may have significant Delay/Response differences. These 25 comprise 8 Sound-only + 8 Both + 9 Image-only baseline-response sites, not the previously included 68 nonresponsive sites. No detected difference is not equivalence. Both directional views use the same 25 sites and row order.

**Upper panels use the same 49-site responsive pool**, with Sound purple / Image black and electrode-equal mean ± SEM. Preserve the existing shared y limits. Extend only the displayed Stimulus mean/SEM from the already audited native electrode means. Reuse the completed 49-site pooled Sound > Image statistics unchanged: 5,000 subject-block draws, .05/.05, seed42. Upper bars describe the full pool, not the selected lower rows. **The existing pooled Stimulus correction window remains [−0.5,0.6] s; the added [0.6,1] segment has no new pooled inference. An absent upper bar there does not mean a tested negative result.** Lower electrode-specific masks already test the full [0,1] s and are shown completely.

**Author-approved display extension:** Stimulus [−0.5,1], Delay [0,1], Response [−0.5,1] s, with width ratio **1.5:1:1.5**. Retain the original physical width per second, Delay/Response widths, 1-mm row height, row order, y limits, colors, 0–0.5 scales and every statistical result. Full Stimulus [0,1] electrode-specific corrected masks, including [0.6,1], are now visible. Negative pre-Stimulus times remain untested and blank in heatmaps. New display matrices/summaries are saved separately; all previous numerical files remain byte-identical.

Previous 128-site comparison notebook/SVGs are archived; all original numerical/statistical outputs are preserved. Run All uses only this responsive-pool, corrected-mask entry. Main Figure 1/2, NMF, original tests, PN Venn/brain panels and all Lexical outputs remain unchanged.
'''
    cells=[nbf.v4.new_markdown_cell(intro,id='sound-cluster-corrected-scope'),
        nbf.v4.new_code_cell('''from pathlib import Path
import sys
from IPython.display import SVG, display
import pandas as pd
sys.path.insert(0,'/cwork/ns458/insula-paper')
from src.supplementary.sound_image_cluster_difference import render, OUT, DISPLAY_OUT
''',id='sound-cluster-imports')]
    info=pool_manifest()
    for key,spec in SPECS.items():
        n_subjects=info['groups'][spec['selection']]['n_subjects']
        cells += [nbf.v4.new_markdown_cell(f"## {spec['title']} · {spec['n']} sites / {n_subjects} participants",
                    id='cluster-heading-'+key.replace('_','-')),
            nbf.v4.new_code_cell(f"display(SVG(filename=str(render('{key}'))))\n",id='cluster-plot-'+key.replace('_','-'))]
    cells.append(nbf.v4.new_code_cell('''from src.supplementary.sound_image_cluster_difference import summarize_panels
summarize_panels()
display(pd.read_csv(DISPLAY_OUT/'selection_counts.csv'))
display(pd.read_csv(DISPLAY_OUT/'panel_phase_significant_counts.csv'))
display(pd.read_csv(OUT/'all_phase_significant_intervals.csv'))
print('Per-electrode trials, tests, masks, signed matrices and electrode lists:', OUT)
''',id='sound-cluster-audit'))
    return cells


def summarize_panels():
    import pandas as pd
    counts=pd.concat([pd.read_csv(DISPLAY_OUT/f'{key}_phase_counts.csv') for key in SPECS],ignore_index=True)
    counts.to_csv(DISPLAY_OUT/'panel_phase_significant_counts.csv',index=False)
    table=pd.read_csv(OUT/'responsive_pool_electrodes.csv')
    selections=[]
    for key,spec in SPECS.items():
        if spec['selection']=='no_difference':group=table.loc[~table.sound_gt_image & ~table.image_gt_sound]
        else:group=table.loc[table[spec['selection']]]
        selections.append(dict(panel=key,n_sites=len(group),n_participants=group.subject.nunique()))
    pd.DataFrame(selections).to_csv(DISPLAY_OUT/'selection_counts.csv',index=False)


def extended_display_traces(table,waves):
    """Extend only the displayed mean/SEM; never rerun or extend inference."""
    import numpy as np
    import pandas as pd
    from src.supplementary.lexical_condition_figures import summarize_traces
    previous=pd.read_csv(OUT/'pooled_electrode_mean_sem.csv',float_precision='round_trip')
    stimulus=waves.loc[waves.phase.eq('stimulus')&waves.electrode.isin(table.electrode)&waves.time.between(*XLIMS['stimulus'])]
    extended=summarize_traces(table,stimulus,[('Stimulus-responsive pool',table)]).rename(columns={'condition':'level'})
    assert extended.n_finite_electrodes.eq(49).all()
    for level,group in previous.loc[previous.phase.eq('stimulus')].groupby('level'):
        group=group.sort_values('time')
        current=extended.loc[extended.level.eq(level)].set_index('time').loc[group.time]
        for column in ['mean','sem']:
            np.testing.assert_allclose(current[column],group[column],rtol=0,atol=1e-14)
    combined=pd.concat([extended,previous.loc[~previous.phase.eq('stimulus')]],ignore_index=True)
    ylim=pool_manifest()['original_shared_y_helper']['limits']
    assert (combined['mean']+combined['sem']).max()<=ylim[1]
    combined.to_csv(DISPLAY_OUT/'pooled_mean_sem_display.csv',index=False)
    return combined


def prepare(key):
    import numpy as np
    import pandas as pd
    from scripts.refresh_lexical_condition_sets import sha,write
    from scripts.run_sound_image_selection import PARAMS
    done=json.loads((TEST_AUDIT/'phase_test_completion.json').read_text())
    assert done['passed'] and done['parameters']==PARAMS and done['n_site_phase_tests']==384
    assert all(sha(TEST_OUT/name)==h for name,h in done['outputs'].items())
    verified=json.loads((TEST_AUDIT/'stimulus_audit.json').read_text());assert verified['passed']
    run=json.loads((TEST_AUDIT/'phase_test_run.json').read_text())
    assert sha(FULL/'electrodes.csv')==run['cohort_sha256']
    assert sha(INPUT/'waveforms.csv.gz')==run['original_waveforms_sha256']
    spec=SPECS[key]
    DISPLAY_OUT.mkdir(parents=True,exist_ok=True)
    info=pool_manifest()
    assert all(sha(path)==h for path,h in info['source_hashes'].items())
    table=pd.read_csv(OUT/'responsive_pool_electrodes.csv')
    assert len(table)==49 and table.subject.nunique()==22 and table.eligible.all()
    assert (table.sound_responsive | table.image_responsive).all()
    parent=pd.read_csv(FULL/'electrodes.csv')
    assert set(table.electrode)==set(parent.loc[parent.sound_responsive | parent.image_responsive,'electrode'])
    if spec['selection']=='no_difference':selected=table.loc[~table.sound_gt_image & ~table.image_gt_sound].copy()
    else:selected=table.loc[table[spec['selection']]].copy()
    assert len(selected)==spec['n'] and selected.subject.nunique()==info['groups'][spec['selection']]['n_subjects']
    if key=='sound_gt_image':row_source=original.OUTPUT/'sound_minus_image_heatmap_matrices.npz'
    else:row_source=BASE/'direction_groups_2026-10-28'/f'{key}_matrices.npz'
    with np.load(row_source,allow_pickle=False) as previous:
        allowed=set(selected.electrode)
        orders={c:[site for site in previous[f'stimulus_{c}_electrodes'].tolist() if site in allowed] for c in CLUSTERS}
    for c,order in orders.items():
        assert set(order)==set(selected.loc[selected.functional_cluster.eq(c),'electrode'])
    tests=pd.read_csv(TEST_OUT/'all_phase_timeperm.csv.gz',float_precision='round_trip')
    waves=pd.read_csv(INPUT/'waveforms.csv.gz',float_precision='round_trip')
    matrices,payload,counts,rows={},{},[],[]
    for phase in PHASES:
        sub=waves.loc[waves.phase.eq(phase)&waves.time.between(*XLIMS[phase])]
        times=np.sort(sub.time.unique());payload[phase+'_time']=times
        paired={level:sub.loc[sub.condition.eq(level)].pivot(index='electrode',columns='time',values='value')
                for level in ['sound','image']}
        phase_test=tests.loc[tests.phase.eq(phase)]
        difference_mask=phase_test.pivot(index='electrode',columns='time',values=spec['direction']+'_mask')
        observed=phase_test.pivot(index='electrode',columns='time',values='observed')
        for c,order in orders.items():
            raw=(paired[spec['positive']]-paired[spec['negative']]).reindex(index=order,columns=times).to_numpy(float)
            assert np.isfinite(raw).all()
            mask=difference_mask.reindex(index=order,columns=times).fillna(False).to_numpy(bool)
            tested=observed.reindex(index=order,columns=times).fillna(False).to_numpy(bool)
            assert not np.any(mask & ~tested)
            assert np.all(raw[mask]>0)
            if phase=='stimulus' and spec['selection']=='no_difference':assert not mask.any()
            shown=np.where(mask,raw,np.nan)
            matrices[(phase,c)]=shown
            prefix=f'{phase}_{c}'
            payload[prefix]=shown;payload[prefix+'_unmasked_difference']=raw
            payload[prefix+'_condition_difference_mask']=mask;payload[prefix+'_tested']=tested
            payload[prefix+'_electrodes']=np.asarray(order,dtype=str)
            rows.extend(dict(panel=key,phase=phase,functional_cluster=c,row=i,electrode=site,
                n_significant_display_bins=int(mask[i].sum()),n_tested_display_bins=int(tested[i].sum()))
                for i,site in enumerate(order))
        full_hits=set(phase_test.loc[phase_test[spec['direction']+'_mask'] & phase_test.electrode.isin(selected.electrode),'electrode'])
        display_hits={site for c in CLUSTERS for i,site in enumerate(orders[c]) if payload[f'{phase}_{c}_condition_difference_mask'][i].any()}
        counts.append(dict(panel=key,selection='Stimulus 0–1 s',phase=phase,direction=spec['direction'],n_selected=len(selected),
            n_significant_electrodes_full_test_window=len(full_hits),n_significant_electrodes_display_window=len(display_hits),
            n_significant_display_bins=sum(int(payload[f'{phase}_{c}_condition_difference_mask'].sum()) for c in CLUSTERS)))
    np.savez_compressed(DISPLAY_OUT/f'{key}_cluster_masked_matrices.npz',**payload)
    selected.to_csv(DISPLAY_OUT/f'{key}_selected_electrodes.csv',index=False)
    pd.DataFrame(rows).to_csv(DISPLAY_OUT/f'{key}_display_mask_audit.csv',index=False)
    pd.DataFrame(counts).to_csv(DISPLAY_OUT/f'{key}_phase_counts.csv',index=False)
    inference=json.loads((OUT/'pooled_inference_manifest.json').read_text())
    assert inference['n_sites']==49 and inference['n_subjects']==22
    assert inference['parameters']==original.PARAMETERS
    assert sha(OUT/'pooled_paired_subject_contributions.npz')==inference['input_sha256']
    assert all(sha(OUT/name)==h for name,h in inference['outputs'].items())
    traces=extended_display_traces(table,waves)
    intervals=pd.read_csv(OUT/'pooled_significant_intervals.csv',float_precision='round_trip')
    assert traces.n_finite_electrodes.eq(49).all()
    return matrices,orders,traces,intervals,info['original_shared_y_helper']['limits'],counts


def render(key):
    assert os.environ.get('SLURM_JOB_ID'),'Use Slurm'
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap
    from src.paths import save_svg
    from scripts.refresh_lexical_condition_sets import write
    spec=SPECS[key]
    matrices,orders,traces,intervals,ylim,counts=prepare(key)
    n_upper=int(traces.n_finite_electrodes.iloc[0])
    for font in Path(sys.prefix,'fonts').glob('arial*.ttf'):font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'svg.fonttype':'none','font.family':'sans-serif','font.sans-serif':['Arial'],'font.size':7})
    legacy=json.loads((BASE/'plot_validation.json').read_text());geometry=legacy['reference_geometry']
    row_height=legacy['actual_cm_per_electrode_row']
    active=[c for c in CLUSTERS if len(orders[c])]
    heights=[geometry['trace_height_cm']]+[len(orders[c])*row_height for c in active]
    # Preserve the original cm/second scale and the Delay/Response widths.
    cm_per_second=geometry['phase_widths_cm'][1]/(XLIMS['delay'][1]-XLIMS['delay'][0])
    widths=[cm_per_second*(XLIMS[p][1]-XLIMS[p][0]) for p in PHASES]
    np.testing.assert_allclose(widths[1:],geometry['phase_widths_cm'][1:],rtol=1e-12)
    left,right,top,bottom=[geometry[k] for k in ['left_cm','right_cm','top_cm','bottom_cm']]
    gap_x,gap_y=geometry['gap_x_cm'],geometry['gap_y_cm']
    w,h=left+sum(widths)+2*gap_x+right,top+sum(heights)+len(active)*gap_y+bottom
    fig=plt.figure(figsize=(w/2.54,h/2.54))
    gs=fig.add_gridspec(len(heights),3,height_ratios=heights,width_ratios=widths,left=left/w,right=1-right/w,
        top=1-top/h,bottom=bottom/h,wspace=gap_x/np.mean(widths),hspace=gap_y/np.mean(heights))
    axes=np.empty((len(heights),3),object);images={}
    trace_colors={'sound':'#6A5ACD','image':'#000000'}
    ticks={'stimulus':[-.5,0,1],'delay':[0,1],'response':[-.5,0,1]}
    for j,phase in enumerate(PHASES):
        for row in range(len(heights)):axes[row,j]=fig.add_subplot(gs[row,j],sharex=axes[0,j] if row else None)
        ax=axes[0,j]
        for level in ['sound','image']:
            data=traces.loc[traces.phase.eq(phase)&traces.level.eq(level)].sort_values('time')
            t,m,se=data.time.to_numpy(),data['mean'].to_numpy(),data['sem'].to_numpy()
            ax.plot(t,m,color=trace_colors[level],lw=1,label=level.capitalize())
            ax.fill_between(t,m-se,m+se,color=trace_colors[level],alpha=.18,linewidth=0)
        ax.axhline(0,color='0.2',ls='--',lw=.7);ax.axvline(0,color='0.2',ls='--',lw=.7)
        ax.set_xlim(*XLIMS[phase]);ax.set_ylim(*ylim)
        for span in intervals.loc[intervals.phase.eq(phase)].itertuples():
            ax.plot([span.start,span.end],[1.03,1.03],transform=ax.get_xaxis_transform(),
                color=trace_colors['sound'],lw=1.5,solid_capstyle='butt',clip_on=False)
        ax.set_title(phase.capitalize(),fontsize=7);ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelbottom=False,labelleft=j==0,left=j==0,labelsize=7,width=.75,length=2)
        if j==0:
            ax.set_ylabel('HGA (z)',fontsize=7);ax.text(.02,.96,f'n={n_upper}',transform=ax.transAxes,va='top',fontsize=7)
        else:ax.spines['left'].set_visible(False)
        for i,c in enumerate(active):
            ax=axes[i+1,j];n=len(orders[c]);matrix=matrices[(phase,c)]
            cmap=LinearSegmentedColormap.from_list('positive_'+c,['#FFFFFF',COLORS[c]])
            cmap.set_bad('white');cmap.set_under('white')
            image=ax.imshow(matrix,extent=(*XLIMS[phase],0,n),origin='lower',aspect='auto',cmap=cmap,
                vmin=0,vmax=.5,interpolation='nearest',rasterized=True)
            np.testing.assert_array_equal(np.asarray(image.get_array()),matrix)
            images[(phase,c)]=image
            ax.axvline(0,color='0.2',ls='--',lw=.8);ax.set_xlim(*XLIMS[phase]);ax.set_ylim(0,n);ax.set_yticks([])
            ax.set_xticks(ticks[phase],labels=[f'{t:g}' for t in ticks[phase]])
            ax.get_xticklabels()[0].set_ha('left');ax.get_xticklabels()[-1].set_ha('right')
            ax.spines[['top','right','left']].set_visible(False)
            ax.tick_params(labelbottom=i==len(active)-1,bottom=i==len(active)-1,labelsize=7,width=.75,length=2)
            if j==0:ax.text(-.07,.5,f'{LABELS[c]} (n={n})',transform=ax.transAxes,ha='right',va='center',fontsize=7,color=COLORS[c])
    axes[0,-1].legend(frameon=False,fontsize=7,loc='upper left',bbox_to_anchor=(1.02,1),borderaxespad=0)
    axes[-1,1].set_xlabel('Time (s)',fontsize=7)
    fig.text(left/w,1-.09/h,'Picture Naming · Stimulus-responsive pool',ha='left',va='top',fontsize=7)
    metric=f"{spec['positive'].capitalize()} − {spec['negative'].capitalize()}"
    caption=fig.text(.01,.025,
        f"Lower: {spec['title']} · n={spec['n']}\n"
        f"{metric} HGA (z) · phase-specific cluster-corrected mask\n"
        f'Upper bars: Sound > Image, responsive pool (n={n_upper})',ha='left',va='bottom',fontsize=6.5,linespacing=1.35)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    labels=[label for ax in axes[-1] for label in ax.get_xticklabels()]+[axes[-1,1].xaxis.label]
    lowest=min(label.get_window_extent(renderer).y0 for label in labels)
    caption.set_position((.01,fig.transFigure.inverted().transform((0,lowest-3*fig.dpi/72))[1]));caption.set_verticalalignment('top')
    for row in axes:
        scales=[ax.get_position().width/np.diff(ax.get_xlim())[0] for ax in row]
        np.testing.assert_allclose(scales,scales[0],rtol=1e-12)
    for j in range(3):
        physical=[axes[i+1,j].get_position().height*h/len(orders[c]) for i,c in enumerate(active)]
        np.testing.assert_allclose(physical,row_height,rtol=1e-12)
    for i,c in enumerate(active):
        box=axes[i+1,-1].get_position()
        cax=fig.add_axes([box.x1+.15/w,(box.y0+box.y1)/2,.75/w,.10/h])
        bar=fig.colorbar(images[(PHASES[-1],c)],cax=cax,orientation='horizontal',ticks=[0,.5])
        bar.ax.set_xticklabels(['0','0.5']);bar.ax.tick_params(labelsize=4.5,length=1.3,pad=1);bar.outline.set_visible(False)
    path=save_svg(fig,original.IMAGE.parent/spec['svg'],close=True)
    write(DISPLAY_OUT/f'{key}_plot_validation.json',dict(passed=True,svg=str(path),n_upper=n_upper,n_lower=spec['n'],
        n_participants_upper=int(traces.n_participants.iloc[0]),n_participants_lower=pool_manifest()['groups'][spec['selection']]['n_subjects'],
        component_counts={c:len(orders[c]) for c in CLUSTERS},metric=metric,
        heatmap_gate='Electrode-specific, phase-specific cluster-corrected condition-difference mask',
        baseline_masks_used=False,tested_phases=['stimulus','delay','response'],selection_window=[0,1],
        display_windows=XLIMS,color_mapping=[0,.5],row_height_cm=row_height,physical_time_scale_equal=True,
        baseline_used_only_to_define_responsive_pool=True,upper_pooled_statistics_unchanged=False,
        upper_stimulus_inference_not_extended=True,phase_widths_cm=widths,width_ratio=[1.5,1.,1.5],
        delay_response_physical_widths_unchanged=True,upper_y_limits_unchanged=ylim,phase_counts=counts))
    print('RENDERED corrected condition-difference masks:',key,flush=True)
    return path
