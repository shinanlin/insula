"""Assigned-only adapters for the original Figure 3 comparison renderers.

Reuse all per-electrode matrix values/masks/orders. Recompute electrode means,
SEM and subject-block pooled inference for the actual contributing electrodes.
Original fits, trial tests and previous derived files remain read-only.
"""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd

from src.main_analysis import FUNCTIONAL_COMPONENTS
from src.supplementary.assigned_condition_cohorts import ROOT,OUT,load_cohort
from src.supplementary.lexical_condition_figures import summarize_traces

PHASES=('stimulus','delay','response')
OLD_PN=ROOT/'results/supplementary/sound_image_test/responsive_pool_difference_masks_2026-10-28/stimulus_display_to1_2026-10-28'
OLD_LD=ROOT/'results/supplementary/decision_repeat_test/difference_heatmap_full_cohort_2026-10-27'
PARAMETERS=dict(p_thresh=.05,p_cluster=.05,tails=1,n_perm=5000,
    axis=0,permutation_type='samples',seed=42,n_jobs=1)
_runtime_compatibility_installed=False


@contextmanager
def override(module,**attributes):
    saved={k:getattr(module,k) for k in attributes}
    try:
        for k,v in attributes.items():setattr(module,k,v)
        yield
    finally:
        for k,v in saved.items():setattr(module,k,v)


def pooled(key):
    global _runtime_compatibility_installed
    from scripts.refresh_lexical_condition_sets import sha,write
    from src.supplementary.condition_region_time_profiles import shared_ylim
    assert os.environ.get('SLURM_JOB_ID'),'Use Slurm'
    table,_=load_cohort(key)
    if key=='picture':table=table.loc[table.sound_responsive|table.image_responsive].copy()
    waves_path=ROOT/('results/supplementary/sound_image_test/full_input_audit_2026-10-28/waveforms.csv.gz'
        if key=='picture' else 'results/supplementary/fig3_lexical_delay_condition_sets/full_cohort_2026-10-27/waveforms.csv.gz')
    waves=pd.read_csv(waves_path,float_precision='round_trip')
    waves=waves.loc[waves.electrode.isin(table.electrode)].copy()
    levels=('sound','image') if key=='picture' else ('Decision','Repeat')
    limits=dict(stimulus=(-.5,1. if key=='picture' else .6),delay=(0.,1.),response=(-.5,1.))
    waves=pd.concat([waves.loc[waves.phase.eq(p)&waves.time.between(*limits[p])] for p in PHASES],ignore_index=True)
    traces=summarize_traces(table,waves,[('Assigned statistical cohort',table)]).rename(columns={'condition':'level'})
    assert traces.n_finite_electrodes.eq(len(table)).all()
    target=OUT/key/'comparison_traces';target.mkdir(parents=True,exist_ok=True)
    traces.to_csv(target/'pooled_electrode_mean_sem.csv',index=False)
    table.to_csv(target/'upper_electrodes.csv',index=False)
    sources={str(waves_path):sha(waves_path),str(OUT/key/'eligible_assigned_electrodes.csv'):sha(OUT/key/'eligible_assigned_electrodes.csv')}
    receipt=target/'pooled_inference_manifest.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved['source_hashes']==sources and saved['parameters']==PARAMETERS:
            assert all(sha(target/n)==h for n,h in saved['output_sha256'].items())
            return table,traces,pd.read_csv(target/'pooled_significant_intervals.csv'),target,saved['ylim']
    from ieeg.calc import stats
    from ieeg.calc.fast import mean_diff
    from scripts.run_sound_image_selection import install_reviewed_runtime_compatibility
    if not _runtime_compatibility_installed:
        install_reviewed_runtime_compatibility()
        _runtime_compatibility_installed=True
    rows=[];spans=[];payload={}
    ids=sorted(table.electrode);subject_by_site=table.set_index('electrode').loc[ids,'subject'].to_numpy()
    subjects=sorted(table.subject.unique());ns=len(subjects);n=len(ids)
    for phase in PHASES:
        # PN [.6,1] display extension never extended its original upper test.
        test_limits=(-.5,.6) if phase=='stimulus' else limits[phase]
        part=waves.loc[waves.phase.eq(phase)&waves.time.between(*test_limits)]
        t=np.sort(part.time.unique());contributions=[]
        for condition in levels:
            values=part.loc[part.condition.eq(condition)].pivot(index='electrode',columns='time',values='value').reindex(index=ids,columns=t).to_numpy(float)
            assert np.isfinite(values).all()
            weighted=np.stack([values[subject_by_site==s].sum(axis=0)*ns/n for s in subjects])
            np.testing.assert_allclose(weighted.mean(axis=0),values.mean(axis=0),rtol=0,atol=1e-14)
            contributions.append(weighted);payload[phase+'_'+condition]=weighted
        print(f'Assigned pooled time_perm_cluster: {key} {phase} n={n} subjects={ns}',flush=True)
        mask,p=stats.time_perm_cluster(*contributions,stat_func=mean_diff,**PARAMETERS)
        observed=mean_diff(*contributions,axis=0)
        mask=np.asarray(mask,bool);p=np.asarray(p,float)
        rows.append(pd.DataFrame(dict(phase=phase,time=t,mask=mask,p_value=p,mean_difference=observed,n_sites=n,n_subjects=ns)))
        edges=np.r_[t[0]-(t[1]-t[0])/2,(t[1:]+t[:-1])/2,t[-1]+(t[-1]-t[-2])/2]
        changes=np.diff(np.r_[False,mask,False].astype(int))
        for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
            spans.append(dict(phase=phase,start=max(float(edges[a]),test_limits[0]),end=min(float(edges[b]),test_limits[1]),n_bins=int(b-a)))
        payload[phase+'_time']=t
    intervals=pd.DataFrame(spans,columns=['phase','start','end','n_bins'])
    intervals.to_csv(target/'pooled_significant_intervals.csv',index=False)
    pd.concat(rows,ignore_index=True).to_csv(target/'pooled_timeperm.csv',index=False)
    np.savez_compressed(target/'pooled_paired_subject_contributions.npz',**payload)
    if key=='picture':
        temp=waves.rename(columns={'level':'condition'})
        ylim,_=shared_ylim(temp,table,dict(conditions=levels,xlims=limits))
    else:ylim=json.loads((ROOT/'results/supplementary/decision_repeat_test/plot_validation.json').read_text())['trace_ylim']
    names=['pooled_timeperm.csv','pooled_significant_intervals.csv','pooled_paired_subject_contributions.npz']
    write(receipt,dict(passed=True,source_hashes=sources,parameters=PARAMETERS,
        n_sites=n,n_subjects=ns,ylim=ylim,stat_func='ieeg.calc.fast.mean_diff',
        exchangeability='Coherent subject-block condition swaps; S/N-weighted electrode sums',
        correction_scope='Original separate phase windows; PN Stimulus upper inference remains [-.5,.6]',
        output_sha256={name:sha(target/name) for name in names}))
    return table,traces,intervals,target,ylim


def matrix_subset(path,selected,target,filename):
    from scripts.refresh_lexical_condition_sets import sha
    matrices={};orders={};payload={}
    with np.load(path,allow_pickle=False) as old:
        for phase in PHASES:
            payload[phase+'_time']=old[phase+'_time']
            for c in FUNCTIONAL_COMPONENTS:
                prefix=f'{phase}_{c}';ids=old[prefix+'_electrodes'].tolist()
                keep=np.asarray([site in set(selected.electrode) for site in ids],bool)
                order=[site for site in ids if site in set(selected.electrode)]
                assert set(order)==set(selected.loc[selected.functional_cluster.eq(c),'electrode'])
                if phase=='stimulus':orders[c]=order
                else:assert order==orders[c]
                for k in old.files:
                    if k==prefix or k.startswith(prefix+'_'):payload[k]=old[k][keep]
                matrices[(phase,c)]=old[prefix][keep]
    np.savez_compressed(target/filename,**payload)
    return matrices,orders,{str(path):sha(path)}


def render_picture(panel):
    from src.supplementary import sound_image_cluster_difference as renderer
    from scripts.refresh_lexical_condition_sets import write
    table,traces,intervals,target,ylim=pooled('picture')
    selected=table.loc[~table.sound_gt_image&~table.image_gt_sound] if panel.startswith('no_difference') else table.loc[table[panel]]
    matrices,orders,proof=matrix_subset(OLD_PN/f'{panel}_cluster_masked_matrices.npz',selected,target,f'{panel}_matrices.npz')
    selected.to_csv(target/f'{panel}_selected_electrodes.csv',index=False)
    specs={k:{**v,'n':int((table.loc[~table.sound_gt_image&~table.image_gt_sound] if k.startswith('no_difference') else table.loc[table[k]]).shape[0])} for k,v in renderer.SPECS.items()}
    info={'groups':{specs[panel]['selection']:dict(n_subjects=selected.subject.nunique())}}
    def prepared(key):
        assert key==panel
        return matrices,orders,traces,intervals,ylim,[]
    with override(renderer,CLUSTERS=FUNCTIONAL_COMPONENTS,SPECS=specs,DISPLAY_OUT=target,
        prepare=prepared,pool_manifest=lambda:info):path=renderer.render(panel)
    write(target/f'{panel}_subset_provenance.json',dict(passed=True,old_matrix_hashes=proof,
        n_upper=len(table),n_lower=len(selected),n_lower_participants=selected.subject.nunique(),
        rows_and_all_retained_values_masks_unchanged=True,unassigned_excluded=True))
    return path


def render_lexical():
    from src.supplementary import decision_repeat_full_difference as renderer
    from scripts.refresh_lexical_condition_sets import write
    table,traces,intervals,target,ylim=pooled('lexical')
    selected=table.loc[table.decision_gt_repeat].copy()
    matrices,orders,proof=matrix_subset(OLD_LD/'decision_minus_repeat_heatmap_matrices.npz',selected,target,'decision_minus_repeat_heatmap_matrices.npz')
    selected.to_csv(target/'selected_electrodes.csv',index=False)
    with override(renderer,CLUSTERS=FUNCTIONAL_COMPONENTS,OUTPUT=target,
        prepare=lambda:(traces,matrices,orders,[],proof),pooled_inference=lambda _:intervals):path=renderer.render()
    write(target/'subset_provenance.json',dict(passed=True,old_matrix_hashes=proof,
        n_upper=len(table),n_lower=len(selected),n_lower_participants=selected.subject.nunique(),
        rows_and_all_retained_values_masks_unchanged=True,unassigned_excluded=True))
    return path
