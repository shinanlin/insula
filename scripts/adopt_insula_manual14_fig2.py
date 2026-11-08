"""Adopt the author-selected 14-site exclusion and refresh the Fig2 data version.

Existing 224-site fits/rank evaluations are reused after exact input checks.
Native saved HGA/masks provide display context; no significance test is rerun.
All operations run through Slurm. Historical figures/models are archived.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import shutil
import sys
import warnings

import h5py
import mne
import nbformat
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

ROOT=Path('/cwork/ns458/insula-paper')
sys.path.insert(0,str(ROOT))
from src.main_analysis import NMF_WINDOWS, RESPONSE_WINDOWS
from src.nmf.reviewed_pca import update_pca_outputs
from src.nmf.rank_selection import choose_k, write_rank_selection_outputs
from scripts.complete_insula_cohort import name

VERSION='insula_manual14_2026-11-07'
RUN=ROOT/'results'/VERSION
NMF=RUN/'nmf'
COHORT=ROOT/'docs/evidence'/VERSION
BASE=ROOT/'results/insula_completion_2026-11-07/nmf'
BASE_COHORT=ROOT/'docs/evidence/insula_completion_2026-11-07'
VALID=ROOT/'results/diagnostics/insula_rank_exclusions_2026-11-07/validation/case_0001/seed_42'
FIG2=ROOT/'results/fig2'
CONFIG=ROOT/'config'/f'{VERSION}.json'
DISPLAY={'stimulus':(-.5,.6),'delay':(0.,1.),'response':(-.5,1.5)}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def prepare():
    config=json.loads(CONFIG.read_text())
    excluded=set(config['excluded_sites'])
    assert len(excluded)==14
    for p in (RUN,NMF,COHORT,FIG2):p.mkdir(parents=True,exist_ok=True)
    before=RUN/'before'
    if not before.exists():
        before.mkdir()
        for rel in ('vizpub/fig2.ipynb','config/current_insula_analysis.json','src/main_analysis.py','docs/MAIN_ANALYSIS.md','AGENTS.md'):
            dest=before/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
        shutil.copytree(ROOT/'img/fig2',before/'img/fig2')
        shutil.copytree(FIG2,before/'results/fig2')
    protected=[]
    for folder in (ROOT/'results/nmf',BASE,BASE_COHORT):
        protected += [p for p in folder.rglob('*') if p.is_file()]
    for n in (1,3,4,5,6):
        protected += [ROOT/f'vizpub/fig{n}.ipynb']
        protected += [p for p in (ROOT/f'img/fig{n}').rglob('*') if p.is_file()]
    write(RUN/'protected_files.json',{str(p):sha(p) for p in protected})
    data=np.load(BASE/'input_matrix.npz')
    channels=data['channels'].astype(str)
    keep=~np.isin(channels,sorted(excluded)); retained=channels[keep]
    assert excluded<=set(channels) and len(retained)==224
    selected=np.load(VALID/'factors_k3.npz')
    assert np.array_equal(selected['channels'],retained)
    payload={k:(data[k][keep] if data[k].ndim and data[k].shape[0]==len(channels) else data[k]) for k in data.files}
    np.savez_compressed(NMF/'input_matrix.npz',**payload)
    assert np.array_equal(np.load(NMF/'input_matrix.npz')['X'],data['X'][keep])
    metadata=pd.read_csv(BASE/'input_sites.csv').set_index('channel').loc[retained].reset_index()
    assert len(metadata)==224 and metadata.subject.nunique()==60
    metadata.to_csv(NMF/'input_sites.csv',index=False)
    shutil.copy2(BASE/'input_features.csv',NMF/'input_features.csv')
    features=pd.read_csv(NMF/'input_features.csv')
    W,H=selected['W'],selected['H']
    assert W.shape==(224,3) and H.shape==(3,399)
    assert np.allclose(np.linalg.norm(H,axis=1),1)
    means=np.column_stack([H[:,features.phase.eq(p).to_numpy()].mean(axis=1) for p in NMF_WINDOWS])
    shares=means/means.sum(axis=1,keepdims=True)
    comps,phases=linear_sum_assignment(-shares)
    names={int(c):('sensory','sustain','motor')[int(p)] for c,p in zip(comps,phases)}
    profiles=pd.DataFrame(means,columns=list(NMF_WINDOWS))
    profiles.insert(0,'component',range(3));profiles['functional_cluster']=profiles.component.map(names)
    profiles.to_csv(NMF/'component_phase_profiles.csv',index=False)
    assignments=metadata.copy();labels=W.argmax(axis=1)
    assignments['component']=labels;assignments['functional_cluster']=[names[int(c)] for c in labels]
    for c,n in names.items():assignments[f'loading_{n}']=W[:,c]
    proportions=W/W.sum(axis=1,keepdims=True);ordered=np.sort(proportions,axis=1)
    assignments['dominance']=proportions.max(axis=1)
    assignments['specificity_max']=ordered[:,-1]-ordered[:,-2]
    assignments['is_mixed']=assignments.specificity_max<.2
    assignments.to_csv(NMF/'channel_assignments.csv',index=False)
    np.savez_compressed(NMF/'factors.npz',W=W,H=H,channels=retained,
                        component_names=np.asarray([names[i] for i in range(3)]))
    pd.concat([features.assign(component=c,functional_cluster=names[c],normalized_H=H[c]) for c in range(3)]).to_csv(NMF/'H_by_phase.csv',index=False)
    starts=pd.read_csv(VALID/'nmf_k3_initializations.csv').rename(columns={'error':'reconstruction_error'})
    starts.to_csv(NMF/'initialization_metrics.csv',index=False)
    assert len(starts)==20 and starts.converged.all()
    error=float(np.linalg.norm(data['X'][keep]-W@H))
    assert np.isclose(error,starts.reconstruction_error.min())
    # Preserve the entire anatomical inventory and actual response evidence;
    # manual analysis exclusions are explicit flags, not negative responses.
    evidence_files=(
        'sampled_sites_unique.csv','sampled_sites_by_subject_task.csv','responsive_sites_by_subject_task.csv',
        'task_site_response_audit.csv','site_anatomy_audit.csv','task_site_anatomy_audit.csv',
        'response_evidence.csv','native_input_manifest.csv','missing_evidence.csv',
        'approved30_original_clean_QC.csv','coordinate_manifest.json','unknown_endpoint_restorations.csv',
        'subject_task_counts_and_lists.csv','task_summary.csv')
    for filename in evidence_files:
        if (BASE_COHORT/filename).is_file():shutil.copy2(BASE_COHORT/filename,COHORT/filename)
    # Old completion-specific final-status/delta exports remain in their source
    # version; do not present their238-input flags as the new224-input decision.
    for p in COHORT.iterdir():
        if p.is_file() and (BASE_COHORT/p.name).is_file() and p.name not in evidence_files and p.name not in ('summary.json','nmf_candidate_sites_not_final_inputs.csv'):
            archive=COHORT/'source_completion';archive.mkdir(exist_ok=True)
            shutil.move(str(p),str(archive/p.name))
    for filename in ('sampled_sites_unique.csv','sampled_sites_by_subject_task.csv','responsive_sites_by_subject_task.csv','task_site_response_audit.csv'):
        frame=pd.read_csv(COHORT/filename)
        idcol='site_id' if 'site_id' in frame else 'channel'
        frame['manual_analysis_excluded']=frame[idcol].isin(excluded)
        if 'confirmed_responsive' in frame:
            frame['analysis_eligible_responsive']=frame.confirmed_responsive & ~frame.manual_analysis_excluded
        frame.to_csv(COHORT/filename,index=False)
    candidates=pd.read_csv(BASE_COHORT/'nmf_candidate_sites_not_final_inputs.csv')
    candidates['manual_analysis_excluded']=candidates.site_id.isin(excluded)
    candidates['NMF_input_complete']=candidates.site_id.isin(retained)
    candidates.loc[candidates.manual_analysis_excluded,'NMF_exclusion_reason']='author_adopted_historical_manual_exclusion'
    candidates.to_csv(COHORT/'nmf_candidate_sites_not_final_inputs.csv',index=False)
    candidates[candidates.manual_analysis_excluded].to_csv(COHORT/'manual_excluded_responsive_sites.csv',index=False)
    summary=json.loads((BASE_COHORT/'summary.json').read_text())
    summary.update(version=VERSION,manual_excluded_responsive_sites=14,
        analysis_eligible_responsive_sites=int((~candidates.manual_analysis_excluded).sum()),
        analysis_eligible_responsive_subjects=int(candidates.loc[~candidates.manual_analysis_excluded,'subject'].nunique()),
        NMF_input_sites=224,NMF_input_subjects=60,
        anatomical_inventory_and_saved_response_flags_unchanged=True)
    write(COHORT/'summary.json',summary)
    (NMF/'exclude_channels.txt').write_text('\n'.join(sorted(excluded))+'\n')
    pd.DataFrame(dict(channel=sorted(excluded),reason='author_adopted_historical_manual_exclusion')).to_csv(COHORT/'manual_exclusions.csv',index=False)
    pairs=pd.read_csv(COHORT/'task_site_anatomy_audit.csv').sort_values(['task','channel']).drop_duplicates('channel')
    unique=pd.read_csv(COHORT/'sampled_sites_unique.csv')
    sampling=unique[['subject','site_id','manual_analysis_excluded']].rename(columns={'site_id':'channel'}).merge(
        pairs[['channel','hemi','x','y','z']],on='channel',validate='one_to_one')
    assert len(sampling)==378 and sampling.hemi.isin(['L','R']).all()
    sampling.to_csv(FIG2/'sampling_coordinates.csv',index=False)
    for name_ in ('waveform_source_membership.csv',):
        membership=pd.read_csv(BASE/name_)
        membership=membership[membership.channel.isin(retained)]
        membership.to_csv(NMF/name_,index=False)
    manifest=json.loads((BASE/'input_manifest.json').read_text())
    manifest.update(version=VERSION,n_sites=224,n_subjects=60,modalities=['sound'],source_cohort=str(COHORT),
        input_sha256=sha(NMF/'input_matrix.npz'),input_matrix_sha256=sha(NMF/'input_matrix.npz'),
        manual_exclusions=sorted(excluded),manual_exclusion_count=14,
        source_before_exclusion_sha256=sha(BASE/'input_matrix.npz'),
        source_cohort_sha256=sha(COHORT/'nmf_candidate_sites_not_final_inputs.csv'))
    write(NMF/'input_manifest.json',manifest)
    rows=[]
    for k in range(2,7):
        values=json.loads((VALID/f'nmf_k{k}_metrics.json').read_text())
        rows.append(dict(k=k,**values,n_boot=200,row_frac=.8))
        shutil.copy2(VALID/f'factors_k{k}.npz',NMF/f'factors_k{k}.npz')
    rank=pd.DataFrame(rows)
    decision=choose_k(rank)
    assert decision['k']==2
    consensus={k:np.load(VALID/f'consensus_k{k}.npy') for k in range(2,7)}
    write_rank_selection_outputs(metrics=rank,consensus=consensus,decision=decision,
        meta={**manifest,'n_boot':200,'row_frac':.8,'random_state':42,'max_iter':5000},
        results_dir=NMF,images_dir=ROOT/'img'/VERSION/'rank')
    shutil.copy2(VALID/'pc_iterations.csv',NMF/'pc_clustering_iterations.csv')
    shutil.copy2(VALID/'pc_summary.csv',NMF/'pc_clustering_metrics.csv')
    pca,_=update_pca_outputs(NMF)
    pca.update(n_iter=500,k_min=2,k_max=10,methods=['kmeans','kmedoids','gmm','ward'])
    write(NMF/'pc_clustering_meta.json',pca)
    previous=pd.read_csv(ROOT/'results/nmf/channel_assignments.csv')
    delta=pd.read_csv(BASE_COHORT/'cohort_changes.csv')
    delta['manual_analysis_excluded']=delta.site_id.isin(excluded)
    delta['in_completed_NMF_input']=delta.site_id.isin(retained)
    delta['in_selected_manual14_NMF']=delta.site_id.isin(retained)
    delta.loc[delta.manual_analysis_excluded,'current_decision_reason']='author_adopted_historical_manual_exclusion'
    delta.to_csv(COHORT/'cohort_changes.csv',index=False)
    delta[delta.in_selected_manual14_NMF & ~delta.in_previous_NMF].to_csv(COHORT/'new_NMF_sites.csv',index=False)
    delta[~delta.in_selected_manual14_NMF & delta.in_previous_NMF].to_csv(COHORT/'removed_previous_NMF_sites.csv',index=False)
    omitted=delta[~delta.in_selected_manual14_NMF & delta.in_previous_NMF]
    summary.update(NMF_run=str(NMF),figures_unchanged=False,Figure2_update_authorized=True,
        other_figure_versions_unchanged=True,
        added_NMF_input_sites=int((delta.in_selected_manual14_NMF & ~delta.in_previous_NMF).sum()),
        removed_previous_NMF_input_sites=len(omitted),
        retained_previous_NMF_input_sites=int((delta.in_selected_manual14_NMF & delta.in_previous_NMF).sum()),
        removed_previous_NMF_reason_counts=omitted.current_decision_reason.value_counts().to_dict())
    write(COHORT/'summary.json',summary)
    comparison=assignments[['channel','subject','functional_cluster']].rename(columns={'functional_cluster':'selected_cluster'}).merge(
        previous[['channel','functional_cluster']].rename(columns={'functional_cluster':'previous_fig2_cluster'}),
        on='channel',how='outer',indicator=True,validate='one_to_one')
    comparison.to_csv(NMF/'comparison_with_previous_fig2.csv',index=False)
    shared=comparison[comparison['_merge']=='both']
    comparison_summary=dict(shared_sites=len(shared),new_sites=int((comparison['_merge']=='left_only').sum()),
        no_longer_included=int((comparison['_merge']=='right_only').sum()),
        changed_labels=int((shared.selected_cluster!=shared.previous_fig2_cluster).sum()),
        ari=float(adjusted_rand_score(shared.selected_cluster,shared.previous_fig2_cluster)))
    model={**manifest,'k':3,'n_init':20,'random_state':42,'naming':
        'Original Hungarian assignment maximizing phase-mean shares: stimulus=sensory, delay=sustain, response=motor; no legacy labels or anatomy used',
        'reconstruction_error':error,'rank_selection':decision,
        'rank_choice':config['rank_choice'],'selection_history':config['selection_history'],
        'fit_source':str(VALID/'factors_k3.npz'),'fit_source_sha256':sha(VALID/'factors_k3.npz'),
        'cluster_counts':assignments.functional_cluster.value_counts().to_dict(),
        'cluster_subject_counts':assignments.groupby('functional_cluster').subject.nunique().to_dict(),
        'previous_fig2_comparison':comparison_summary,'previous_multimodal_comparison':
            {'status':'historical comparison superseded by explicit previous_fig2_comparison'},
        'factor_H_unit_norm':True,'fitted_model_is_not_cophenetic_argmax':True}
    write(NMF/'nmf_manifest.json',model)
    write(RUN/'prepared.json',dict(passed=True,summary=summary,model=model,
        original_238_source_unchanged=sha(BASE/'input_matrix.npz')==manifest['source_before_exclusion_sha256'],
        slurm_job_id=os.environ['SLURM_JOB_ID']))
    print(json.dumps(dict(n_sites=224,n_subjects=60,cluster_counts=model['cluster_counts'],phase_profiles=profiles.to_dict('records')),indent=2),flush=True)


def display_inputs():
    channels=set(pd.read_csv(NMF/'input_sites.csv').channel)
    membership=pd.read_csv(NMF/'waveform_source_membership.csv')
    source_manifest=pd.read_csv(BASE_COHORT/'native_input_manifest.csv',keep_default_na=False)
    frames=[];sources=[]
    for source in source_manifest.itertuples(index=False):
        active=membership[(membership.task==source.task)&(membership.subject==source.subject)&(membership.phase==source.phase)]
        wanted=sorted(set(active.channel)&channels)
        if not wanted:continue
        lo,hi=DISPLAY[source.phase]
        if source.source_type=='native_timeperm_H5':
            ep=mne.read_epochs(source.epoch_path,preload=False,verbose='ERROR')
            epnames=[name(n,source.subject) for n in ep.ch_names]
            picks=[epnames.index(ch) for ch in wanted]
            assert not(set(wanted)&{name(n,source.subject) for n in ep.info['bads']})
            grid=ep.times.copy();display=(grid>=lo)&(grid<=hi)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                means=np.nanmean(ep.get_data(picks=picks,copy=True,verbose='ERROR'),axis=0)[:,display]
            with h5py.File(source.statistics_path,'r') as h:
                names=[name(n,source.subject) for n in h['ch_names'][:]]
                assert names==epnames
                masks=np.asarray(h['mask'][:],bool)[picks][:,display]
            part=pd.DataFrame(dict(channel=np.repeat(wanted,display.sum()),time=np.tile(grid[display],len(wanted)),
                                   value=means.ravel(),mask=masks.ravel()))
            del ep,means
        else:
            parts=[]
            for chunk in pd.read_csv(source.statistics_path,chunksize=200000):
                good=chunk.channel.isin(wanted)&chunk.time.between(lo,hi)
                if 'modality' in chunk:good &= chunk.modality.eq('sound')
                if 'description' in chunk:good &= chunk.description.eq('Repeat')
                if good.any():parts.append(chunk.loc[good,['channel','time','value','mask']])
            assert parts,(source.task,source.subject,source.phase)
            part=pd.concat(parts,ignore_index=True)
        part['subject']=source.subject;part['task']=source.task;part['phase']=source.phase
        part['time']=part.time.round(9)
        part['mask']=part['mask'].eq(True)
        part=part.groupby(['subject','channel','task','phase','time'],as_index=False).agg(value=('value','mean'),mask=('mask','any'))
        frames.append(part)
        sources.append(dict(task=source.task,subject=source.subject,phase=source.phase,
            source_type=source.source_type,epoch_path=source.epoch_path,statistics_path=source.statistics_path,
            included_channels='|'.join(wanted),n_time_samples=int(part.time.nunique())))
    task_waves=pd.concat(frames,ignore_index=True)
    assert not task_waves.duplicated(['channel','task','phase','time']).any()
    task_waves.to_csv(RUN/'display_task_waveforms.csv.gz',index=False)
    waves=task_waves.groupby(['subject','channel','phase','time'],as_index=False).agg(value=('value','mean'),mask=('mask','any'))
    # Explicitly align the displayed samples; no missing signal or mask is imputed.
    blocks=[]
    for phase,(lo,hi) in DISPLAY.items():
        # Native Response epochs end at1.4921875 s:1.5 is the original plotting
        # boundary, not an observed sample. Keep that established half-open edge.
        stop=int(np.floor(hi*128))+(0 if phase=='response' else 1)
        times=np.arange(np.ceil(lo*128),stop)/128
        index=pd.MultiIndex.from_product([sorted(channels),times],names=['channel','time'])
        part=waves[waves.phase==phase].set_index(['channel','time']).reindex(index).reset_index()
        part['phase']=phase;part['subject']=part.channel.str.split('_').str[0]
        part['mask_available']=part['mask'].notna();part['mask']=part['mask'].eq(True)
        assert part.value.notna().all(),(phase,'Missing native display samples',part[part.value.isna()][['channel','time']].head().to_dict('records'))
        blocks.append(part)
    waves=pd.concat(blocks,ignore_index=True)
    features=pd.read_csv(NMF/'input_features.csv')
    data=np.load(NMF/'input_matrix.npz')
    raw=np.column_stack([waves[waves.phase==r.phase].set_index(['channel','time']).value.reindex(
        pd.MultiIndex.from_arrays([data['channels'],np.repeat(r.time,len(data['channels']))])).to_numpy() for r in features.itertuples()])
    assert np.allclose(raw,data['raw'],atol=1e-12,rtol=0), 'Display waveform disagrees with fitted input'
    waves[['subject','channel','phase','time','value']].to_csv(NMF/'site_waveforms.csv.gz',index=False)
    waves[['subject','channel','phase','time','mask','mask_available']].to_csv(FIG2/'site_significance_masks.csv.gz',index=False)
    waves.groupby(['subject','channel','phase']).agg(n_time_samples=('time','size'),
        n_mask_available=('mask_available','sum'),n_significant=('mask','sum')).reset_index().to_csv(FIG2/'site_significance_coverage.csv',index=False)
    evidence=pd.read_csv(COHORT/'response_evidence.csv')
    for phase,(lo,hi) in RESPONSE_WINDOWS.items():
        expected=set(evidence[(evidence.phase==phase)&evidence.significant].channel)&channels
        observed=set(waves[(waves.phase==phase)&waves.time.gt(lo)&waves.time.lt(hi)&waves['mask']].channel)
        assert expected==observed,(phase,'Saved mask selection mismatch',expected^observed)
    report=dict(n_sites=224,n_points=len(waves),n_significant_points=int(waves['mask'].sum()),
        n_points_without_saved_mask=int((~waves.mask_available).sum()),
        existing_saved_masks_only=True,no_new_timeperm_tests=True,phase_selection_counts_reproduced=True,
        fitted_samples_match_original_selected_input=True,display_windows=DISPLAY,
        source_priority='Exact native or packaged source used in completed input; no cross-source mask replacement',
        aggregation='any saved mask across the same native-good tasks contributing to each site waveform',sources=sources)
    write(FIG2/'site_significance_manifest.json',report)
    write(RUN/'display_inputs_completed.json',report)
    print('DISPLAY_INPUTS_COMPLETE',len(waves),flush=True)


def update_notebook():
    path=ROOT/'vizpub/fig2.ipynb';nb=nbformat.read(path,as_version=4)
    byid={c.get('id'):c for c in nb.cells}
    byid['fig2-title'].source=('# Figure 2 — author-selected manual14 exclusion\n\n'
        'The author adopted exclusion of the 14 historical manual-list sites on 2026-11-07. '
        'This Figure uses **224 bipolar sites / 60 participants**, Repeat + sound only, and the '
        'existing three-component interpretation. The maximum-cophenetic rank remains **2**; '
        'the near k=2/3 plateau and decline at k=4 are reported without changing the metric values.\n\n'
        'The version was selected after exploratory exclusion comparisons; algorithmic seed repetition '
        'does not provide independent data validation. Native response statistics and task QC are retained. '
        'Fit windows: Stimulus [0,.6], Delay [0,1], Response [-.5,1] s. '
        'Display context retains Response through 1.5 s. Group means and SEM use electrodes as the unit.\n\n'
        'Decision and exact IDs: `docs/MAIN_ANALYSIS.md`, `config/insula_manual14_2026-11-07.json`. '
        'Historical Figure1 and Figure3–6 outputs/models have not been refreshed by this Figure2 update.')
    imports=byid['fig2-imports']
    imports.source=imports.source.replace('from src.main_analysis import CONDITION, MODALITY, COHORT_DIR, NMF_WINDOWS, COMPONENT_COLORS',
        'from src.main_analysis import CONDITION, MODALITY, FIG2_COHORT_DIR as COHORT_DIR, FIG2_NMF_DIR, NMF_WINDOWS, COMPONENT_COLORS')
    style=byid['fig2-style'];style.source=style.source.replace('NMF_DIR = nmf_results_dir()\nNMF_ASSIGNMENTS = nmf_assignments_path()',
        "NMF_DIR = FIG2_NMF_DIR\nNMF_ASSIGNMENTS = NMF_DIR / 'channel_assignments.csv'")
    load=byid['fig2-load']
    load.source=load.source.replace("assert set(assignments.channel)==set(expected.site_id)",
        "expected=expected.loc[expected.NMF_input_complete]\nassert set(assignments.channel)==set(expected.site_id)\n"
        "excluded=set(json.loads((ROOT/'config/insula_manual14_2026-11-07.json').read_text())['excluded_sites'])\n"
        "assert not (set(assignments.channel)&excluded)")
    load.source=load.source.replace("len(assignments)==COHORT_SUMMARY['responsive_sites']","len(assignments)==COHORT_SUMMARY['NMF_input_sites']==224")
    load.source=load.source.replace("print('Responsive sites:'","print('Figure2 NMF input sites:'")
    byid['fig2-load-md'].source='## Load the author-selected sound-only NMF cohort\n\nOne waveform per bipolar site; equal task means within each site. Group means and SEM are calculated across electrode means.'
    byid['fig2-hga-md'].source='## 3. HGA by largest-loading cluster\n\nMean ± SEM across electrode means. Participant-level waveforms are exported separately and are not the unit of the displayed group average.'
    byid['54d7c386'].source="rank = pd.read_csv(NMF_DIR / 'rank_selection_metrics.csv')\ndisplay(rank)"
    byid['fig2-rank-plot'].source=byid['fig2-rank-plot'].source.replace('ax.set_ylabel("Cosine sil.",','ax.set_ylabel("Mean ARI",')
    byid['fig2-rank-md'].source=('## 5. Rank stability\n\nRanks 2–6 use 200 subsamples of 80% of the selected224 sites. '
        'The curve below is **mean ARI** (correcting the former Cosine sil. label). '
        'The marked k=3 is the retained Figure2 model; the cophenetic maximum remains k=2. '
        'All rank metrics and k=2–6 values are shown in the table above the curve.')
    byid['fig2-pca-kmeans-md'].source=('## 6. Waveform PCA and independent k-means\n\n'
        'PCA uses the same clipped, L2-normalized224-site matrix. Retain PCs whose individual '
        'explained variance exceeds5%, with the original minimum-two-PC fallback. '
        'K-means uses k=3 for comparison with the retained NMF model; colors are aligned '
        'to NMF labels and plotted memberships are k-means assignments.')
    laterality=byid['fig2-laterality-plot']
    laterality.source=laterality.source.replace("ROOT/'results/fig1/insula_plot_coordinates.csv'","FIG2_DATA/'sampling_coordinates.csv'")
    # Existing statistical tests are recomputed by this cell. Its old Sustained
    # star was justified by an old p value, so bind visibility to the new result.
    laterality.source=laterality.source.replace("if 'sustain' in n_rate.index:","if 'sustain' in n_rate.index and fisher_p['sustain'] < 0.05:")
    laterality.source += "\n(FIG2_DATA/'laterality_statistics.json').write_text(json.dumps(dict(fisher_two_sided=fisher_p,chi_square_p=float(p_chi),sampling_sites=int(n_imp.sum()),sampling_subjects=int(implanted.subject.nunique()),manual_exclusions_retained_in_anatomical_coverage=True),indent=2)+'\\n')\n"
    byid['fig2-laterality-md'].source=('## 9. Hemisphere coverage and assigned cluster fractions\n\n'
        'The denominator is the completed anatomical coverage:378 sites /67 participants. '
        'It retains anatomically sampled manual-excluded and unresolved-response sites as coverage metadata. '
        'The numerator uses only the224 selected NMF assignments. The original electrode-count '
        'two-sided Fisher and cluster-by-hemisphere chi-square tests are recomputed for this version; '
        'these are descriptive electrode-level summaries, not independent participant-level inference. '
        'The prior Sustained marker is displayed only if its recomputed raw Fisher p<.05.')
    provenance=byid['e4f7f44d']
    provenance.source=provenance.source.replace("previous_multimodal_comparison=manifest['previous_multimodal_comparison']", "previous_fig2_comparison=manifest['previous_fig2_comparison']")
    provenance.source=provenance.source.replace("n_sites=len(assignments),", "version='insula_manual14_2026-11-07',manual_exclusions=manifest['manual_exclusions'],model_directory=str(NMF_DIR),n_sites=len(assignments),")
    for c in nb.cells:
        if c.cell_type=='code':c.outputs=[];c.execution_count=None
    nbformat.validate(nb);nbformat.write(nb,path)
    write(RUN/'notebook_updated.json',dict(cells=len(nb.cells),preserved_display_code=True,
        corrections=['explicit224-site model path','meanARI axis label','saved-mask source consistency',
                     'PCA and SEM captions','current anatomical denominator','recomputed Sustained marker']))


def main():
    assert os.environ.get('SLURM_JOB_ID'),'Run through Slurm'
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','display-inputs','notebook'));args=p.parse_args()
    {'prepare':prepare,'display-inputs':display_inputs,'notebook':update_notebook}[args.mode]()


if __name__=='__main__':main()
