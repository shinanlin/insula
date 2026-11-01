"""Full-cohort PN Sound/Image traces and saved Sound > Image site differences."""
from pathlib import Path
import ast
import hashlib
import json
import os

ROOT = Path('/cwork/ns458/insula-paper')
BASE = ROOT / 'results/supplementary/sound_image_test'
FULL = ROOT / 'results/supplementary/fig3_picture_stimulus_condition_sets'
INPUT = BASE / 'full_input_audit_2026-10-28'
OUTPUT = BASE / 'difference_heatmap_full_cohort_2026-10-28'
AUDIT = ROOT / 'results/audit/sound_image_difference_full_cohort_2026-10-28'
NOTEBOOK = ROOT / 'vizpub/supplementary/sound_image_test.ipynb'
IMAGE = ROOT / 'img/supplementary/sound_image_test/fig3_sound_image_test.svg'
PHASES = ('stimulus','delay','response')
XLIMS = {'stimulus':(-.5,.6),'delay':(0.,1.),'response':(-.5,1.)}
CLUSTERS = ('sensory','sustain','motor','Unassigned')
LABELS = {'sensory':'Sensory','sustain':'Sustained','motor':'Motor','Unassigned':'Unassigned'}
COLORS = {'sensory':'#2369BD','sustain':'#C4A35A','motor':'#A9373B','Unassigned':'#858585'}
PARAMETERS = dict(p_thresh=.05,p_cluster=.05,tails=1,n_perm=5000,axis=0,
                  permutation_type='samples',seed=42,n_jobs=1)


def notebook_cells():
    import nbformat as nbf
    return [nbf.v4.new_markdown_cell('''# Picture Naming: full-cohort Sound/Image traces and HGA differences

**Upper sample: 128 eligible native-good paired insula sites / 27 participants**, including sites without NMF assignments. Repeat Sound/Image trial means retain their original native QC and approved exact-Stimulus source repairs. These are electrode-equal mean ± SEM curves; Sound is purple and Image black. The old 81-site / 11-common-responder notebook and figure are archived.

**Lower sample: the existing 24 Sound > Image sites / 13 participants**, selected by the completed per-electrode Stimulus **[0,1] s** test in the same full 128-site cohort. The author explicitly chose this directly tested set rather than the 13 common baseline responders. Functional groups are **Sensory 7 / Sustained 7 / Motor 8 / Unassigned 2**. Retain the two Unassigned sites without fitting or inventing NMF labels. No new per-electrode test or site selection is performed.

Heatmaps show **Sound − Image HGA**, pairing the same electrode/time sample. Subtract unmasked condition means first, then retain the Sound native baseline display mask, following the positive-condition display rule of the Decision/Repeat comparison. Colors alone do not claim significant time-specific differences. The original blue/gold/red component colors (gray for Unassigned), **1-mm rows** and fixed **0–0.5** mapping match that comparison. Negative differences display white; values above 0.5 saturate. Signed raw values remain in the exported matrices.

Display windows and proportional widths retain the original HGA convention: **Stimulus [−0.5,0.6], Delay [0,1], Response [−0.5,1] s**. The **0–1-s selection window** is deliberately distinct from the Stimulus display window. Classified sites retain the reviewed original full-cohort row order where available; Unassigned sites use first native baseline-response time. Upper y limits reuse the original Sound/Image shared mean+SEM-envelope helper (P98, floor .4, padding .02), applied to the full input population.

Upper Sound > Image bars use the same approved method as Decision/Repeat: coherently swap conditions within subjects; S/N-weighted subject sums preserve the plotted electrode mean. Use original iEEG time_perm_cluster, mean_diff, 5000 draws, one-sided .05/.05, seed42, and a separate temporal correction within each complete displayed phase. Save new pooled inference separately; reuse cached results only when paired inputs and parameters match. No original statistical file, native trial/QC rule, Figure 1/2, NMF, Lexical figure or full PN response-set/brain-map notebook is changed.

All 768 site × condition × phase input cells passed the native-data audit. New summaries, paired differences, masks, electrode lists and pooled inference are under `difference_heatmap_full_cohort_2026-10-28/`. The complete PN sampling/Venn/brain displays remain in [the response-set notebook](fig3_picture_stimulus_condition_sets.ipynb).
''',id='sound-title'),nbf.v4.new_code_cell('''from pathlib import Path
import sys
from IPython.display import SVG, display
_si_full_root = Path('/cwork/ns458/insula-paper')
if str(_si_full_root) not in sys.path:
    sys.path.insert(0,str(_si_full_root))
from src.supplementary.sound_image_full_difference import render,OUTPUT,FULL
sound_image_difference_svg = render()
display(SVG(filename=str(sound_image_difference_svg)))
print('Full cohort and existing 24-site test:',FULL)
print('Paired HGA difference data and current inference:',OUTPUT)
''',id='sound-plot'),nbf.v4.new_code_cell('''import pandas as pd
display(pd.read_csv(OUTPUT/'cohort_flow.csv'))
display(pd.read_csv(OUTPUT/'selected_electrodes.csv')[['subject','channel','functional_cluster','sound_gt_image_intervals']])
display(pd.read_csv(OUTPUT/'pooled_significant_intervals.csv'))
''',id='sound-current-cohort-audit')]


def _shared_ylim(waves,table):
    """Execute only the exact original two helper definitions, never analyses."""
    import numpy as np
    import pandas as pd
    path = AUDIT/'before/sound_image_test.ipynb'
    notebook = json.loads(path.read_text())
    source = ''.join(next(c for c in notebook['cells'] if c.get('id')=='sound-original-functions')['source'])
    names = {'rfig3_electrode_key','rfig3_compute_shared_ylim'}
    nodes = [n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==names
    space = dict(np=np,pd=pd,rfig3_clusters=CLUSTERS)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path)+':original shared-y helpers','exec'),space)
    data = waves.merge(table[['electrode','functional_cluster']],on='electrode',validate='many_to_one')
    limits = space['rfig3_compute_shared_ylim'](data,phases=PHASES,compare_field='condition',
        levels=('sound','image'),xlim_by_phase=XLIMS,clusters=CLUSTERS,ymax_floor=.4)
    proof = dict(source_notebook=str(path),functions_copied_verbatim=True,
        functions={n.name:hashlib.sha256(ast.get_source_segment(source,n).encode()).hexdigest() for n in nodes},
        clusters_included=CLUSTERS,limits=limits)
    return limits,proof


def prepare():
    import numpy as np
    import pandas as pd
    from scripts.refresh_lexical_condition_sets import sha,write
    from src.supplementary.lexical_condition_figures import summarize_traces
    input_done = json.loads((ROOT/'results/audit/sound_image_full_input_2026-10-28/completion.json').read_text())
    assert input_done['passed'] and input_done['complete_cells']==768
    assert all(sha(INPUT/name)==h for name,h in input_done['outputs'].items())
    paths = [FULL/'electrodes.csv',FULL/'sound_gt_image_electrodes.csv',INPUT/'waveforms.csv.gz',
             BASE/'frozen_heatmap_row_membership.csv',FULL/'summary.json']
    hashes = {str(p):sha(p) for p in paths}
    table = pd.read_csv(paths[0])
    selected = pd.read_csv(paths[1])
    waves = pd.read_csv(paths[2],float_precision='round_trip')
    old_order = pd.read_csv(paths[3])
    assert len(table)==128 and table.subject.nunique()==27 and table.electrode.is_unique
    assert table.eligible.all() and table.paired_statistics_complete.all()
    assert len(selected)==24 and selected.subject.nunique()==13 and selected.electrode.is_unique
    assert set(selected.electrode)==set(table.loc[table.sound_gt_image,'electrode'])
    assert selected.functional_cluster.value_counts().to_dict()=={'motor':8,'sensory':7,'sustain':7,'Unassigned':2}
    assert set(waves.electrode)==set(table.electrode) and np.isfinite(waves.value).all()
    assert not waves.duplicated(['electrode','phase','condition','time']).any()
    shown_waves = waves.loc[np.logical_or.reduce([
        waves.phase.eq(p)&waves.time.between(*XLIMS[p]) for p in PHASES])].copy()
    traces = summarize_traces(table,shown_waves,[('All eligible',table)]).rename(columns={'condition':'level'})
    assert traces.n_finite_electrodes.eq(128).all() and traces.n_selected_electrodes.eq(128).all()
    ylim,ylim_proof = _shared_ylim(shown_waves,table)
    orders = {}
    for cluster in CLUSTERS:
        group = selected.loc[selected.functional_cluster.eq(cluster)]
        if cluster!='Unassigned':
            original = old_order.loc[old_order.phase.eq('stimulus')&old_order.functional_cluster.eq(cluster)].sort_values('row').electrode
            order = [site for site in original if site in set(group.electrode)]
        else:
            first = shown_waves.loc[shown_waves.electrode.isin(group.electrode)&shown_waves.baseline_mask].groupby('electrode').time.min()
            order = sorted(group.electrode,key=lambda site:(first.get(site,float('inf')),site))
        assert len(order)==len(group) and set(order)==set(group.electrode)
        orders[cluster] = order
    identities = sorted(table.electrode)
    subject_by_site = table.set_index('electrode').loc[identities,'subject'].to_numpy()
    subjects = sorted(table.subject.unique())
    matrices,payload,pooled,audits = {},{},{},[]
    for phase in PHASES:
        sub = shown_waves.loc[shown_waves.phase.eq(phase)]
        times = np.sort(sub.time.unique())
        payload[f'{phase}_time'] = pooled[f'{phase}_time'] = times
        paired = {}
        for level in ('sound','image'):
            pivot = sub.loc[sub.condition.eq(level)].pivot(index='electrode',columns='time',values='value').reindex(
                index=identities,columns=times)
            values = pivot.to_numpy(float)
            assert values.shape==(128,len(times)) and np.isfinite(values).all()
            contribution = np.stack([values[subject_by_site==s].sum(axis=0)*27/128 for s in subjects])
            reference = traces.loc[traces.phase.eq(phase)&traces.level.eq(level)].sort_values('time')
            np.testing.assert_array_equal(times,reference.time.to_numpy())
            np.testing.assert_allclose(values.mean(axis=0),reference['mean'],rtol=0,atol=1e-14)
            np.testing.assert_allclose(contribution.mean(axis=0),reference['mean'],rtol=0,atol=1e-14)
            pooled[f'{phase}_{level}'] = contribution
            paired[level] = pivot
        masks = sub.loc[sub.condition.eq('sound')].pivot(index='electrode',columns='time',values='baseline_mask')
        for cluster in CLUSTERS:
            order = orders[cluster]
            delta = paired['sound'].reindex(order).to_numpy(float)-paired['image'].reindex(order).to_numpy(float)
            mask = masks.reindex(index=order,columns=times).to_numpy(bool)
            displayed = np.where(mask,delta,np.nan)
            matrices[(phase,cluster)] = displayed
            key = f'{phase}_{cluster}'
            payload[key] = displayed
            payload[key+'_unmasked_difference'] = delta
            payload[key+'_sound_baseline_mask'] = mask
            payload[key+'_electrodes'] = np.asarray(order,dtype=str)
            audits.extend(dict(phase=phase,functional_cluster=cluster,row=i,electrode=site,n_samples=len(times),
                n_displayed_bins=int(mask[i].sum()),n_negative_numeric_bins=int((displayed[i]<0).sum()),
                n_above_point5=int((displayed[i]>.5).sum())) for i,site in enumerate(order))
    OUTPUT.mkdir(parents=True,exist_ok=True)
    traces.to_csv(OUTPUT/'pooled_electrode_mean_sem.csv',index=False)
    selected.to_csv(OUTPUT/'selected_electrodes.csv',index=False)
    pd.DataFrame(audits).to_csv(OUTPUT/'paired_sample_audit.csv',index=False)
    np.savez_compressed(OUTPUT/'sound_minus_image_heatmap_matrices.npz',**payload)
    pooled['electrode'],pooled['subject'] = np.asarray(identities,dtype=str),np.asarray(subjects,dtype=str)
    np.savez_compressed(OUTPUT/'pooled_paired_subject_contributions.npz',**pooled)
    flow = [dict(stage='Eligible paired sample',n_sites=128,n_participants=27),
            dict(stage='Existing Sound > Image selection',n_sites=24,n_participants=13)]
    flow += [dict(stage=LABELS[c],n_sites=len(orders[c]),
        n_participants=int(selected.loc[selected.functional_cluster.eq(c)].subject.nunique())) for c in CLUSTERS]
    pd.DataFrame(flow).to_csv(OUTPUT/'cohort_flow.csv',index=False)
    write(OUTPUT/'cohort_audit.json',dict(passed=True,n_upper_sites=128,n_upper_participants=27,
        n_lower_sites=24,n_lower_participants=13,component_counts={c:len(orders[c]) for c in CLUSTERS},
        selection_source=str(FULL/'sound_gt_image_electrodes.csv'),selection_window=[0,1],display_windows=XLIMS,
        no_NMF_or_response_gate_on_upper_sample=True,no_new_per_electrode_test=True,raw_signed_differences_preserved=True,
        color_mapping=[0,.5],baseline_display_rule='Sound native baseline mask after subtracting unmasked condition means',
        original_shared_y_helper=ylim_proof,input_files=hashes))
    print('INPUT/COHORT AUDIT: 128 upper, 24 lower, 7/7/8/2 components',flush=True)
    return traces,matrices,orders,ylim,hashes


def pooled_inference(traces):
    import numpy as np
    import pandas as pd
    from ieeg.calc import stats
    from ieeg.calc.fast import mean_diff
    from scripts.run_sound_image_selection import install_reviewed_runtime_compatibility
    from scripts.refresh_lexical_condition_sets import sha,write
    input_hash = sha(OUTPUT/'pooled_paired_subject_contributions.npz')
    manifest_path = OUTPUT/'pooled_inference_manifest.json'
    if manifest_path.exists():
        cached = json.loads(manifest_path.read_text())
        if cached['parameters']==PARAMETERS and cached['input_sha256']==input_hash:
            assert all(sha(OUTPUT/p)==h for p,h in cached['outputs'].items())
            return pd.read_csv(OUTPUT/'pooled_significant_intervals.csv',float_precision='round_trip')
    compatibility = install_reviewed_runtime_compatibility()
    tables,spans = [],[]
    with np.load(OUTPUT/'pooled_paired_subject_contributions.npz',allow_pickle=False) as data:
        for phase in PHASES:
            time = data[f'{phase}_time']
            sound,image = data[f'{phase}_sound'],data[f'{phase}_image']
            assert sound.shape==image.shape==(27,len(time))
            observed = mean_diff(sound,image,axis=0)
            reference = traces.loc[traces.phase.eq(phase)].pivot(index='time',columns='level',values='mean').sort_index()
            np.testing.assert_allclose(observed,reference.sound-reference.image,rtol=0,atol=1e-14)
            print('128-site Sound > Image pooled time_perm_cluster:',phase,flush=True)
            mask,p_raw = stats.time_perm_cluster(sound,image,stat_func=mean_diff,**PARAMETERS)
            mask,p_raw = np.asarray(mask,bool),np.asarray(p_raw,float)
            assert mask.shape==p_raw.shape==observed.shape and np.isfinite(p_raw).all()
            assert not np.any(mask&(observed<=0))
            tables.append(pd.DataFrame(dict(phase=phase,time=time,mask=mask,p_value=p_raw,
                mean_difference=observed,n_sites=128,n_subjects=27)))
            edges = np.r_[time[0]-(time[1]-time[0])/2,(time[:-1]+time[1:])/2,time[-1]+(time[-1]-time[-2])/2]
            changes = np.diff(np.r_[False,mask,False].astype(int))
            spans.extend(dict(phase=phase,start=max(float(edges[first]),XLIMS[phase][0]),
                end=min(float(edges[end]),XLIMS[phase][1]),first_significant_time=float(time[first]),
                last_significant_time=float(time[end-1]),n_bins=int(end-first))
                for first,end in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)))
    pd.concat(tables,ignore_index=True).to_csv(OUTPUT/'pooled_timeperm.csv',index=False)
    intervals = pd.DataFrame(spans,columns=['phase','start','end','first_significant_time','last_significant_time','n_bins'])
    intervals.to_csv(OUTPUT/'pooled_significant_intervals.csv',index=False)
    write(manifest_path,dict(parameters=PARAMETERS,stat_func='ieeg.calc.fast.mean_diff',input_sha256=input_hash,
        n_sites=128,n_subjects=27,runtime_compatibility=compatibility,installed_library_unchanged=True,
        exchangeability='Coherent subject-block condition swaps; S/N-weighted electrode sums',
        correction_scope='Each complete displayed phase is a separate temporal family',
        outputs={name:sha(OUTPUT/name) for name in ['pooled_timeperm.csv','pooled_significant_intervals.csv']},intervals=spans))
    return intervals


def render():
    assert os.environ.get('SLURM_JOB_ID'),'Use Slurm'
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap
    from src.paths import save_svg
    from scripts.refresh_lexical_condition_sets import sha,write
    import sys
    for font in Path(sys.prefix,'fonts').glob('arial*.ttf'):
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'svg.fonttype':'none','font.family':'sans-serif','font.sans-serif':['Arial'],'font.size':7})
    traces,matrices,orders,ylim,hashes = prepare()
    intervals = pooled_inference(traces)
    legacy = json.loads((BASE/'plot_validation.json').read_text())
    geometry = legacy['reference_geometry']
    cm,fontsize = 1/2.54,7
    row_height = legacy['actual_cm_per_electrode_row']
    row_heights = [geometry['trace_height_cm']]+[len(orders[c])*row_height for c in CLUSTERS]
    widths = geometry['phase_widths_cm']
    left,right,top,bottom = [geometry[k] for k in ['left_cm','right_cm','top_cm','bottom_cm']]
    gap_x,gap_y = geometry['gap_x_cm'],geometry['gap_y_cm']
    w,h = left+sum(widths)+2*gap_x+right,top+sum(row_heights)+4*gap_y+bottom
    fig = plt.figure(figsize=(w*cm,h*cm))
    gs = fig.add_gridspec(5,3,height_ratios=row_heights,width_ratios=widths,left=left/w,right=1-right/w,
        top=1-top/h,bottom=bottom/h,wspace=gap_x/np.mean(widths),hspace=gap_y/np.mean(row_heights))
    axes = np.empty((5,3),dtype=object)
    trace_colors = {'sound':'#6A5ACD','image':'#000000'}
    ticks = {'stimulus':[-.5,0,.6],'delay':[0,1],'response':[-.5,0,1]}
    images = {}
    for j,phase in enumerate(PHASES):
        for row in range(5):
            axes[row,j] = fig.add_subplot(gs[row,j],sharex=axes[0,j] if row else None)
        ax = axes[0,j]
        for level in ['sound','image']:
            data = traces.loc[traces.phase.eq(phase)&traces.level.eq(level)].sort_values('time')
            t,m,se = data.time.to_numpy(),data['mean'].to_numpy(),data['sem'].to_numpy()
            ax.plot(t,m,color=trace_colors[level],lw=1,label=level.capitalize())
            ax.fill_between(t,m-se,m+se,color=trace_colors[level],alpha=.18,linewidth=0)
        ax.axhline(0,color='0.2',ls='--',lw=.7); ax.axvline(0,color='0.2',ls='--',lw=.7)
        ax.set_xlim(*XLIMS[phase]); ax.set_ylim(*ylim)
        for span in intervals.loc[intervals.phase.eq(phase)].itertuples():
            ax.plot([span.start,span.end],[1.03,1.03],transform=ax.get_xaxis_transform(),
                color=trace_colors['sound'],lw=1.5,solid_capstyle='butt',clip_on=False)
        ax.set_title(phase.capitalize(),fontsize=fontsize)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelbottom=False,labelleft=j==0,left=j==0,labelsize=fontsize,width=.75,length=2)
        if j==0:
            ax.set_ylabel('HGA (z)',fontsize=fontsize)
            ax.text(.02,.96,'n=128',transform=ax.transAxes,va='top',fontsize=fontsize)
        else:
            ax.spines['left'].set_visible(False)
        for i,cluster in enumerate(CLUSTERS):
            ax = axes[i+1,j]
            n,matrix = len(orders[cluster]),matrices[(phase,cluster)]
            cmap = LinearSegmentedColormap.from_list('positive_'+cluster,['#FFFFFF',COLORS[cluster]])
            cmap.set_bad('white'); cmap.set_under('white')
            image = ax.imshow(matrix,extent=(*XLIMS[phase],0,n),origin='lower',aspect='auto',
                cmap=cmap,vmin=0.,vmax=.5,interpolation='nearest',rasterized=True)
            images[(phase,cluster)] = image
            np.testing.assert_array_equal(np.asarray(image.get_array()),matrix)
            assert image.norm.vmin==0 and image.norm.vmax==.5
            ax.axvline(0,color='0.2',ls='--',lw=.8)
            ax.set_xlim(*XLIMS[phase]); ax.set_ylim(0,n)
            ax.set_yticks([]); ax.set_xticks(ticks[phase],labels=[f'{t:g}' for t in ticks[phase]])
            ax.get_xticklabels()[0].set_ha('left'); ax.get_xticklabels()[-1].set_ha('right')
            ax.spines[['top','right','left']].set_visible(False)
            ax.tick_params(labelbottom=i==3,bottom=i==3,labelsize=fontsize,width=.75,length=2)
            if j==0:
                ax.text(-.07,.5,f'{LABELS[cluster]} (n={n})',transform=ax.transAxes,
                    ha='right',va='center',fontsize=fontsize,color=COLORS[cluster])
    axes[0,-1].legend(frameon=False,fontsize=fontsize,loc='upper left',bbox_to_anchor=(1.02,1),borderaxespad=0)
    axes[-1,1].set_xlabel('Time (s)',fontsize=fontsize)
    fig.text(left/w,1-.09/h,'Picture Naming · Repeat · 27 participants',ha='left',va='top',fontsize=fontsize)
    caption = fig.text(.01,.025,'Heatmaps: Sound − Image · 24 selected sites',ha='left',va='bottom',fontsize=6.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    time_labels = [label for ax in axes[-1] for label in ax.get_xticklabels()]+[axes[-1,1].xaxis.label]
    lowest = min(label.get_window_extent(renderer).y0 for label in time_labels)
    caption.set_position((.01,fig.transFigure.inverted().transform((0,lowest-3*fig.dpi/72))[1]))
    caption.set_verticalalignment('top')
    for row in range(5):
        scales = [ax.get_position().width/np.diff(ax.get_xlim())[0] for ax in axes[row]]
        np.testing.assert_allclose(scales,scales[0],rtol=1e-12)
    for j in range(3):
        heights = [axes[i+1,j].get_position().height*h/len(orders[c]) for i,c in enumerate(CLUSTERS)]
        np.testing.assert_allclose(heights,row_height,rtol=1e-12)
    for i,cluster in enumerate(CLUSTERS):
        box = axes[i+1,-1].get_position()
        cax = fig.add_axes([box.x1+.15/w,(box.y0+box.y1)/2,.75/w,.10/h])
        bar = fig.colorbar(images[(PHASES[-1],cluster)],cax=cax,orientation='horizontal',ticks=[0,.5])
        bar.ax.set_xticklabels(['0','0.5']); bar.ax.tick_params(labelsize=4.5,length=1.3,pad=1)
        bar.outline.set_visible(False); bar.ax.set_title('ΔHGA (z)',fontsize=5,pad=2)
    path = save_svg(fig,IMAGE,close=True)
    assert all(sha(p)==h for p,h in hashes.items())
    write(OUTPUT/'plot_validation.json',dict(passed=True,n_upper_sites=128,n_upper_participants=27,
        n_lower_sites=24,n_lower_participants=13,component_counts={c:len(orders[c]) for c in CLUSTERS},
        heatmap_metric='Sound minus Image HGA',color_mapping=[0,.5],negative_signed_values_preserved=True,
        negatives_displayed_white=True,selection_window=[0,1],display_windows=XLIMS,
        positive_condition_baseline_display_rule='Sound native baseline mask',
        upper_shared_ylim=ylim,cm_per_electrode_row=row_height,physical_time_scale_equal=True,
        pooled_test_parameters=PARAMETERS,new_per_electrode_tests=False,new_NMF_fit=False,
        inputs=hashes,svg=str(path)))
    return path
