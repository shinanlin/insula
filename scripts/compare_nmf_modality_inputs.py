#!/usr/bin/env python3
"""Controlled sound-only vs pooled-modality NMF on an identical cohort and feature grid."""
from pathlib import Path
import hashlib
import json
import pickle
import re
import sys
import mne
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.nmf.waveform_analysis import fit_one_nmf
SOURCE=Path('/hpc/group/coganlab/nanlinshi/insula-functional')
OUT=ROOT/'results/diagnostics/nmf_sensory_comparison_2026-10-17'
DISPLAY={'stimulus':(-.5,.6),'delay':(0,1),'response':(-.5,1.5)}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_model(raw,channels,features,tag):
    X=np.clip(raw,0,None);X/=np.linalg.norm(X,axis=1,keepdims=True)
    rng=np.random.default_rng(42)
    fits=[fit_one_nmf(X,3,int(seed),max_iter=5000) for seed in rng.integers(0,np.iinfo(np.int32).max,size=20)]
    best=min(fits,key=lambda f:f['error'])
    W,H=np.asarray(best['W']),np.asarray(best['H'])
    means=np.column_stack([H[:,features.phase.eq(p)].mean(axis=1) for p in DISPLAY])
    shares=means/means.sum(axis=1,keepdims=True)
    component,phase=linear_sum_assignment(-shares)
    names={int(c):['sensory','sustain','motor'][int(p)] for c,p in zip(component,phase)}
    labels=np.array([names[c] for c in W.argmax(axis=1)])
    rows=pd.DataFrame({'channel':channels,'cluster':labels})
    rows['subject']=rows.channel.str.split('_').str[0]
    for c,name in names.items():rows['loading_'+name]=W[:,c]
    rows.to_csv(OUT/f'{tag}_assignments.csv',index=False)
    h=[]
    for c,name in names.items():
        d=features.copy();d['component']=name;d['H']=H[c];h.append(d)
    pd.concat(h).to_csv(OUT/f'{tag}_H.csv',index=False)
    return rows,dict(n_sites=len(channels),n_features=raw.shape[1],cluster_counts=rows.cluster.value_counts().to_dict(),
        converged_starts=sum(f['converged'] for f in fits),reconstruction_error=best['error'])


def summarize_cohort_sensitivity():
    """Use the saved fits to quantify removing the nine non-sound-input sites."""
    common=pd.read_csv(OUT/'all_modalities_assignments.csv')
    canonical=pd.read_csv(ROOT/'results/nmf/channel_assignments.csv')
    matched=common[['channel','cluster']].merge(canonical[['channel','functional_cluster']],
        on='channel',validate='one_to_one')
    modality=pd.read_csv(OUT/'modality_control_label_comparison.csv')
    old_sensory=modality.loc[modality.old_four_phase_cluster.eq('sensory')]
    result=dict(full_cohort_sites=len(canonical),control_cohort_sites=len(common),
        full_cohort_sensory_sites=int(canonical.functional_cluster.eq('sensory').sum()),
        control_cohort_sensory_sites=int(common.cluster.eq('sensory').sum()),
        n_shared_labels_changed=int(matched.cluster.ne(matched.functional_cluster).sum()),
        ari=float(adjusted_rand_score(matched.cluster,matched.functional_cluster)),
        old_sensory_sites_in_modality_control=len(old_sensory),
        old_sensory_sound_only_labels=old_sensory.sound_only_cluster.value_counts().to_dict(),
        old_sensory_all_modality_labels=old_sensory.all_modalities_cluster.value_counts().to_dict(),
        interpretation='Same 399 features, modalities, preprocessing, solver, seeds and naming; only the cohort differs')
    matched.to_csv(OUT/'cohort_control_label_comparison.csv',index=False)
    (OUT/'cohort_control_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protected=[p for folder in [ROOT/'results/nmf',SOURCE/'results/nmf'] for p in folder.glob('*') if p.is_file()]
    protected += [ROOT/'vizpub/fig1.ipynb',ROOT/'vizpub/fig2.ipynb']
    before={str(p):sha(p) for p in protected}
    with (ROOT/'results/fig1/available_evidence_stream.pkl').open('rb') as f:
        waves=pickle.load(f)['trace_rows'].copy()
    waves['subject']=waves.channel.str.split('_').str[0]
    for item in json.loads((ROOT/'results/fig1/average_hga_supplemented_epochs.json').read_text()):
        phase=item['phase'];lo,hi=DISPLAY[phase];path=Path(item['path'])
        epochs=mne.read_epochs(path,preload=False,verbose='error');epochs.load_data();epochs.pick(item['channels'])
        evoked=epochs.average(method=lambda d:np.nanmean(d,axis=0));chosen=(evoked.times>=lo)&(evoked.times<=hi)
        d=pd.DataFrame(evoked.data[:,chosen],index=evoked.ch_names,columns=evoked.times[chosen])
        d=d.rename_axis('channel').reset_index().melt(id_vars='channel',var_name='time',value_name='value')
        modality=re.search(r'_recording-([^_]+)',path.name)
        d=d.assign(subject=d.channel.str.split('_').str[0],phase=phase,task=re.search(r'_task-([^_]+)',path.name)[1],modality=modality[1] if modality else 'sound')
        waves=pd.concat([waves,d],ignore_index=True)
    features=pd.read_csv(ROOT/'results/nmf/input_features.csv')
    canonical=np.load(ROOT/'results/nmf/input_matrix.npz')
    ids=set(canonical['channels'])
    waves=waves.loc[waves.channel.isin(ids)].groupby(['subject','channel','task','phase','modality','time'],as_index=False).value.mean()
    waves.to_csv(OUT/'all_phase_waveforms_by_task_modality.csv.gz',index=False)
    blocks={}
    for tag,rows in [('sound_only',waves.loc[waves.modality.eq('sound')]),('all_modalities',waves)]:
        task=rows.groupby(['subject','channel','task','phase','time'],as_index=False).value.mean()
        site=task.groupby(['subject','channel','phase','time'],as_index=False).value.mean()
        phase_blocks=[]
        for phase in DISPLAY:
            wanted=features.loc[features.phase.eq(phase),'time'].to_numpy()
            mat=site.loc[site.phase.eq(phase)].pivot(index='channel',columns='time',values='value').reindex(columns=wanted)
            mat.columns=pd.MultiIndex.from_product([[phase],wanted],names=['phase','time']);phase_blocks.append(mat)
        blocks[tag]=pd.concat(phase_blocks,axis=1)
    common=blocks['sound_only'].dropna().index.intersection(blocks['all_modalities'].dropna().index).sort_values()
    full_all=blocks['all_modalities'].loc[canonical['channels']].to_numpy()
    assert np.allclose(full_all,canonical['raw'],atol=1e-12,rtol=0)
    summaries={};models={}
    for tag,mat in blocks.items():
        rows,summary=fit_model(mat.loc[common].to_numpy(),common.to_numpy(),features,tag)
        summaries[tag]=summary;models[tag]=rows
    comparison=models['sound_only'][['channel','cluster']].rename(columns={'cluster':'sound_only_cluster'}).merge(
        models['all_modalities'][['channel','cluster']].rename(columns={'cluster':'all_modalities_cluster'}),on='channel',validate='one_to_one')
    comparison['changed']=comparison.sound_only_cluster.ne(comparison.all_modalities_cluster)
    old=pd.read_csv(SOURCE/'results/nmf/channel_assignments.csv')
    comparison=comparison.merge(old[['channel','functional_cluster']].rename(columns={'functional_cluster':'old_four_phase_cluster'}),on='channel',how='left')
    comparison.to_csv(OUT/'modality_control_label_comparison.csv',index=False)
    missing=sorted(ids-set(common));pd.DataFrame({'channel':missing}).to_csv(OUT/'sites_without_complete_sound_input.csv',index=False)
    d125=comparison.loc[comparison.channel.eq('D0125_LTMS2-3')].to_dict('records')
    trace=pd.read_csv(OUT/'old_sensory_sites_trace.csv')
    absent=trace.loc[trace.disposition.eq('anatomy_eligible_without_confirmed_response')]
    new_sens=pd.read_csv(ROOT/'results/nmf/channel_assignments.csv')
    new_sens=new_sens.loc[new_sens.functional_cluster.eq('sensory'),['channel']].merge(
        old[['channel','functional_cluster']],on='channel',how='left')
    composition=new_sens.functional_cluster.fillna('not_in_old_255').value_counts().to_dict()
    summary=dict(common_sites=len(common),n_features=len(features),full_input_reproduced=True,
        fit_windows={'stimulus':[0,.6],'delay':[0,1],'response':[-.5,1]},
        cohorts_identical_between_control_fits=True,settings_and_naming_identical=True,
        n_labels_changed=int(comparison.changed.sum()),ari=float(adjusted_rand_score(comparison.sound_only_cluster,comparison.all_modalities_cluster)),
        models=summaries,excluded_from_control_missing_sound=missing,
        D0125_LTMS2_3=d125,current_sensory_composition_from_old_labels=composition,
        four_old_sensory_without_response_confirmation=absent[['channel','response_status','old_sound_peak_0_0p6']].to_dict('records'),
        protected_files_unchanged=all(sha(Path(p))==h for p,h in before.items()))
    assert summary['protected_files_unchanged']
    (OUT/'modality_control_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    summarize_cohort_sensitivity()


if __name__=='__main__':main()
