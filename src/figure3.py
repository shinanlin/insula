"""Unified Figure 3 main and supplementary entry points.

Reuse reviewed statistical inputs and drawing functions. No model fitting,
new tests, new NMF labels or changed plotting conventions are introduced.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
from contextlib import contextmanager

ROOT=Path(__file__).resolve().parents[1]
MAIN_RESULTS=ROOT/'results/fig3'
MAIN_IMG=ROOT/'img/fig3'
SAMPLING_RESULTS=ROOT/'results/supplementary/fig3_sampling_responsiveness'
SAMPLING_IMG=ROOT/'img/supplementary/fig3_sampling_responsiveness'
HGA_RESULTS=ROOT/'results/supplementary/fig3_all_condition_hga'
HGA_IMG=ROOT/'img/supplementary/fig3_all_condition_hga'
PHASES=('stimulus','delay','response')
CONDITION_SPECS={
    'picture':dict(source='fig3_picture_stimulus_condition_sets',conditions=('sound','image'),
        names=('Sound > Image','Image > Sound'),title='Picture Naming · Stimulus 0–1 s'),
    'lexical':dict(source='fig3_lexical_delay_condition_sets/full_cohort_2026-10-27',
        conditions=('decision','repeat'),names=('Decision > Repeat','Repeat > Decision'),
        title='Lexical Delay · Delay 0–1 s'),
}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


@contextmanager
def override(module,**attributes):
    before={name:getattr(module,name) for name in attributes}
    try:
        for name,value in attributes.items():setattr(module,name,value)
        yield
    finally:
        for name,value in before.items():setattr(module,name,value)


def style():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for font in Path(sys.prefix,'fonts').glob('arial*.ttf'):font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'svg.fonttype':'none','font.family':'Arial','font.size':7})


def render_main():
    """Generate the agreed diagrams, seven spatial maps and two time portraits."""
    assert os.environ.get('SLURM_JOB_ID'),'Run through Slurm'
    from src.supplementary import condition_difference_group_panels as spatial
    from src.supplementary import condition_region_time_profiles as temporal
    from src.supplementary import assigned_condition_cohorts as cohorts
    style()
    MAIN_RESULTS.mkdir(parents=True,exist_ok=True);MAIN_IMG.mkdir(parents=True,exist_ok=True)
    with override(cohorts,OUT=MAIN_RESULTS/'cohorts'),override(spatial,
        OUT=MAIN_RESULTS/'regions',IMG=MAIN_IMG,NOTEBOOK=ROOT/'vizpub/fig3.ipynb'):
        results=spatial.run()
    with override(temporal,OUT=MAIN_RESULTS/'time_profiles',IMG=MAIN_IMG,
        REGIONS=MAIN_RESULTS/'regions',NOTEBOOK=ROOT/'vizpub/fig3.ipynb'):
        for key in ('picture','lexical'):results[key]['time_portrait']=str(temporal.render(key))
    paths=[]
    for spec in results.values():paths += [spec['diagram'],spec['time_portrait'],*spec['brains'].values()]
    write(MAIN_RESULTS/'manifest.json',dict(passed=True,job_id=os.environ['SLURM_JOB_ID'],
        entry_notebook=str(ROOT/'vizpub/fig3.ipynb'),comparisons=results,
        svg_sha256={path:sha(path) for path in paths},
        functional_components=['sensory','sustain','motor'],unassigned_excluded=True,
        anatomical_unknown_policy_unchanged=True,no_new_tests_or_NMF=True,
        temporal_scope='PN21 no-difference average; LD27 active-pool Decision>Repeat difference; same sites above and below'))
    return results


def render_sampling():
    """Sampling counts and literal baseline-response Venns belong in SI."""
    assert os.environ.get('SLURM_JOB_ID'),'Run through Slurm'
    import pandas as pd
    from src.supplementary import assigned_condition_overviews as overview
    from src.supplementary import assigned_condition_cohorts as cohorts
    from src.main_analysis import COHORT_DIR
    style();SAMPLING_IMG.mkdir(parents=True,exist_ok=True)
    results={};rows=[];subjects=[]
    sampling_source=COHORT_DIR/'sampled_sites_by_subject_task.csv'
    all_sampling=pd.read_csv(sampling_source)
    with override(cohorts,OUT=MAIN_RESULTS/'cohorts'),override(overview,OUT=SAMPLING_RESULTS):
        for key in ('picture','lexical'):
            results[key]=overview.render(key,core_only=True,image_dir=SAMPLING_IMG)
            table,parent=cohorts.load_cohort(key)
            task='PictureNaming' if key=='picture' else 'LexicalDelay'
            anatomical=all_sampling.loc[all_sampling.task.eq(task)].copy()
            anatomical['channel']=anatomical.site_id
            assert set(parent.channel)<=set(anatomical.channel)
            conditions=('sound','image') if key=='picture' else ('decision','repeat')
            response=table[conditions[0]+'_responsive']|table[conditions[1]+'_responsive']
            steps=[('Anatomically accepted sampling inventory (coverage metadata)',anatomical),
                ('Native-good paired-statistics-complete cohort',parent),('Assigned statistical cohort',table),
                ('Either-condition baseline response union',table.loc[response])]
            for label,frame in steps:
                rows.append(dict(comparison=key,stage=label,n_electrodes=len(frame),n_participants=frame.subject.nunique()))
                for subject,part in frame.groupby('subject'):
                    subjects.append(dict(comparison=key,stage=label,subject=subject,n_electrodes=len(part),channels=';'.join(part.channel)))
            parent.to_csv(SAMPLING_RESULTS/key/'anatomical_sampling_inventory.csv',index=False)
            anatomical.to_csv(SAMPLING_RESULTS/key/'full_anatomical_task_sampling.csv',index=False)
            table.to_csv(SAMPLING_RESULTS/key/'assigned_sampling_electrodes.csv',index=False)
    pd.DataFrame(rows).to_csv(SAMPLING_RESULTS/'cohort_counts.csv',index=False)
    pd.DataFrame(subjects).to_csv(SAMPLING_RESULTS/'subject_task_counts_and_lists.csv',index=False)
    write(SAMPLING_RESULTS/'manifest.json',dict(passed=True,panels=results,
        physical_sampling_inventory_is_coverage_metadata=True,
        sampling_source=str(sampling_source),sampling_source_sha256=sha(sampling_source),
        statistical_and_Venn_cohorts_exclude_unassigned=True,
        center_means_both_conditions_baseline_responsive=True,
        not_the_main_condition_difference_schematic=True,no_new_tests=True))
    return results


def render_full_hga(key):
    """Preserve the original component×phase HGA/paired-heatmap drawing style."""
    assert os.environ.get('SLURM_JOB_ID'),'Run through Slurm'
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from src.main_analysis import FUNCTIONAL_COMPONENTS
    from src.paths import save_svg
    from src.nmf.waveform_analysis import FUNCTION_COLORS
    from src import figure3_legacy_hga as original
    from src.supplementary.context_comparisons import fit_hga_canvas
    from src.supplementary.assigned_condition_cohorts import load_cohort
    style();HGA_RESULTS.mkdir(parents=True,exist_ok=True);HGA_IMG.mkdir(parents=True,exist_ok=True)
    cohort_key='lexical' if key=='decision_repeat' else 'picture'
    table,_=load_cohort(cohort_key)
    if key in ('sound_image','decision_repeat'):
        source=ROOT/('results/supplementary/sound_image_test/full_input_audit_2026-10-28/waveforms.csv.gz'
            if key=='sound_image' else 'results/supplementary/fig3_lexical_delay_condition_sets/full_cohort_2026-10-27/waveforms.csv.gz')
        data=pd.read_csv(source,float_precision='round_trip')
        data=data.loc[data.electrode.isin(table.electrode)].copy()
        data=data.merge(table[['electrode','functional_cluster']],on='electrode',validate='many_to_one')
        data['mask']=data.baseline_mask
        data['level']=data.condition
        levels=('sound','image') if key=='sound_image' else ('Decision','Repeat')
        assert data.electrode.nunique()==len(table)
    elif key=='repeat_passive':
        source=ROOT/'results/supplementary/context_comparisons/waveforms.csv.gz'
        data=pd.read_csv(source,float_precision='round_trip')
        data=data.loc[data.contrast.eq(key)&data.electrode.isin(table.electrode)].copy()
        data=data.drop(columns='functional_cluster').merge(table[['electrode','functional_cluster']],on='electrode',validate='many_to_one')
        levels=('Repeat','Passive')
    else:raise ValueError(key)
    assert data.functional_cluster.isin(FUNCTIONAL_COMPONENTS).all()
    assert set(data.level)==set(levels) and set(data.phase)==set(PHASES)
    limits=dict(stimulus=(-.5,1.) if key=='sound_image' else (-.5,.6),delay=(0.,1.),response=(-.5,1.))
    data=pd.concat([data.loc[data.phase.eq(p)&data.time.between(*limits[p])] for p in PHASES],ignore_index=True)
    assert not data.duplicated(['electrode','phase','level','time']).any()
    floor=.4 if key=='sound_image' else .6
    ylim=original.rfig3_compute_shared_ylim(data,phases=PHASES,compare_field='level',levels=levels,xlim_by_phase=limits,ymax_floor=floor)
    captured={}
    def capture(frame,event,arg):
        if event=='return' and frame.f_code is original.rfig3_plot_epoch_raster_traces_stack_phases.__code__:
            captured.update({name:frame.f_locals[name] for name in ('electrodes_by_cluster','phase_raster','phase_subs')})
    previous=sys.getprofile()
    try:
        sys.setprofile(capture)
        fig,axes=original.rfig3_plot_epoch_raster_traces_stack_phases(data,phases=PHASES,
            compare_field='level',levels=levels,xlim_by_phase=limits,ylim=ylim,colors=FUNCTION_COLORS,cm_per_raster_row=.032)
    finally:sys.setprofile(previous)
    fig.axes[0].get_subplotspec().get_gridspec().update(wspace=.26)
    for ax in fig.axes:
        labels=ax.get_xticklabels();ticks=ax.get_xticks()
        if labels and len(labels)==len(ticks) and any(v.get_visible() for v in labels):
            texts=[v.get_text() for v in labels]
            ax.set_xticklabels(['' if 1e-8<abs(t)<.08 else text for t,text in zip(ticks,texts)])
    fit_hga_canvas(fig,width_cm=13.,max_height_cm=None)
    path=save_svg(fig,HGA_IMG/f'supp_fig3_{key}_all_component_hga.svg',close=True)
    selected=data[['electrode','subject','channel','functional_cluster']].drop_duplicates()
    selected.to_csv(HGA_RESULTS/f'{key}_electrodes.csv',index=False)
    rows=[];payload={}
    for phase in PHASES:
        for c,item in zip(FUNCTIONAL_COMPONENTS,captured['phase_raster'][phase]):
            assert item['cluster']==c
            prefix=f'{phase}_{c}';matrix=item['matrix'];order=item['electrodes']
            payload[prefix+'_matrix']=matrix;payload[prefix+'_electrodes']=np.asarray(order,dtype=str)
            rows.extend(dict(phase=phase,functional_cluster=c,row=i,electrode=e) for i,e in enumerate(order))
            assert matrix.shape[0]==2*len(order)
            chunks=captured['phase_subs'][phase]
            times=np.sort(chunks.loc[chunks.functional_cluster.eq(c),'time'].unique())
            payload[prefix+'_time']=times
    np.savez_compressed(HGA_RESULTS/f'{key}_matrices.npz',**payload)
    pd.DataFrame(rows).to_csv(HGA_RESULTS/f'{key}_row_order.csv',index=False)
    counts=data.groupby(['phase','level','functional_cluster']).agg(n_electrodes=('electrode','nunique'),n_participants=('subject','nunique')).reset_index()
    counts.to_csv(HGA_RESULTS/f'{key}_counts.csv',index=False)
    write(HGA_RESULTS/f'{key}_manifest.json',dict(passed=True,svg=str(path),svg_sha256=sha(path),
        source=str(source),source_sha256=sha(source),n_electrodes=len(selected),n_participants=selected.subject.nunique(),
        phase_windows=limits,levels=list(levels),original_drawing_function_bodies_unchanged=True,
        original_component_hues_and_condition_styles=True,heatmap_mask='Each condition native baseline mask, not a condition-difference mask',
        original_P95_color_limits=True,electrode_mean_and_SEM=True,no_new_tests_or_NMF=True,
        passive_response_alignment='Retained nominal alignment, not an actual vocal onset' if key=='repeat_passive' else None))
    return path
