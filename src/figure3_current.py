"""Versioned Figure3 inputs for the author-selected completed/manual14 cohort.

Native baseline masks, original trial selection and the original 2000-draw
two-sided condition test are retained. Existing figures/models remain separate.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import shutil

ROOT = Path('/cwork/ns458/insula-paper')
VERSION = 'figure3_manual14_2026-11-07'
RUN = ROOT/'results'/VERSION
COHORT = ROOT/'docs/evidence/insula_manual14_2026-11-07'
NMF = ROOT/'results/insula_manual14_2026-11-07/nmf'
PHASES = ('stimulus','delay','response')
SPECS = {
    'picture': dict(task='PictureNaming', phase='stimulus', levels=('sound','image'),
        conditions=('Repeat','Repeat'), modalities=('sound','image'),
        source=ROOT/'results/supplementary/fig3_picture_stimulus_condition_sets',
        audit=ROOT/'results/audit/picture_stimulus_responsive_union_2026-11-01',
        waves=ROOT/'results/supplementary/sound_image_test/full_input_audit_2026-10-28/waveforms.csv.gz',
        xlims={'stimulus':(-.5,1.),'delay':(0.,1.),'response':(-.5,1.)}),
    'lexical': dict(task='LexicalDelay', phase='delay', levels=('Decision','Repeat'),
        conditions=('Decision','Repeat'), modalities=('sound','sound'),
        source=ROOT/'results/supplementary/fig3_lexical_delay_condition_sets/full_cohort_2026-10-27',
        audit=ROOT/'results/audit/lexical_delay_responsive_union_2026-11-01',
        waves=ROOT/'results/supplementary/fig3_lexical_delay_condition_sets/full_cohort_2026-10-27/waveforms.csv.gz',
        xlims={'stimulus':(-.5,.6),'delay':(0.,1.),'response':(-.5,1.)}),
}


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b''):result.update(block)
    return result.hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def entity(path, key, default=''):
    match=re.search(r'(?:^|_)'+key+r'-([^_.]+)',path.name)
    return match[1] if match else default


def product(task,subject,phase,condition,modality):
    """Use exact-onset repaired PN first, then the preserved native products."""
    roots=[]
    if task=='PictureNaming' and phase=='stimulus':
        roots.append(ROOT/'results/native_stimulus_event_repair_2026-10-21'/task/subject/'BIDS')
    roots += [ROOT/'results/insula_completion_2026-11-07/restored_native'/task/subject/'BIDS',
        ROOT/'results/native_preprocessing_restore_2026-10-18'/task/subject/'BIDS',
        Path('/cwork/ns458')/('BIDS-1.3_PictureNaming' if task=='PictureNaming' else 'BIDS-1.0_LexicalDecRepDelay')/'BIDS',
        ROOT/'results/picture_naming_restore_2026-10-24/BIDS']
    for bids in roots:
        def selected(path):
            return (path.is_file() and 'highgamma' in path.name
                and entity(path,'desc')==condition and entity(path,'proc').lower()==phase
                and entity(path,'recording','sound')==modality)
        epochs=[p for p in (bids/'derivatives/epoch(bipolar)'/f'sub-{subject}'/'epoch(band)(zscore)').glob('*') if selected(p)]
        stats=[p for p in (bids/'derivatives/statistics'/f'sub-{subject}'/'bipolar').glob('*') if selected(p)]
        if epochs or stats:
            if len(epochs)==len(stats)==1:return epochs[0],stats[0],''
            return None,None,'partial_or_ambiguous_native_product:'+str(bids)
    return None,None,'native_paired_condition_product_missing'


def snapshot():
    before=RUN/'before'
    if before.exists():return
    before.mkdir(parents=True)
    for relative in ('vizpub/fig3.ipynb','src/figure3.py','src/diagnostics/figure3_weight_distributions.py',
                     'scripts/consolidate_figure3.py','config/current_insula_analysis.json','AGENTS.md','docs/MAIN_ANALYSIS.md'):
        path=ROOT/relative; dest=before/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    for relative in ('results/fig3','img/fig3'):
        shutil.copytree(ROOT/relative,before/relative)
    paths=[]
    for pattern in ('img/fig[12456]/**/*','vizpub/fig[12456].ipynb','results/nmf/**/*',
        'results/insula_manual14_2026-11-07/nmf/**/*','results/insula_completion_2026-11-07/nmf/**/*',
        'results/fig1/**/*','results/fig2/**/*','vizpub/supplementary/*.ipynb',
        'results/supplementary/fig3_picture_stimulus_condition_sets/**/*',
        'results/supplementary/fig3_lexical_delay_condition_sets/full_cohort_2026-10-27/**/*'):
        paths.extend(p for p in ROOT.glob(pattern) if p.is_file())
    write(RUN/'protected_files.json',{str(p):sha(p) for p in sorted(set(paths))})


def audit_inputs():
    assert os.environ.get('SLURM_JOB_ID'),'Run via Slurm'
    import h5py
    import mne
    import pandas as pd
    import numpy as np
    RUN.mkdir(parents=True,exist_ok=True);snapshot()
    sampled=pd.read_csv(COHORT/'sampled_sites_by_subject_task.csv')
    assignments=pd.read_csv(NMF/'channel_assignments.csv')
    assert len(assignments)==224 and assignments.functional_cluster.isin(('sensory','sustain','motor')).all()
    summaries={};workers=[];sources=[]
    for key,spec in SPECS.items():
        sites=sampled[sampled.task.eq(spec['task'])].copy().rename(columns={'site_id':'channel'})
        sites=sites.merge(assignments,on=['subject','channel'],how='left',validate='one_to_one',suffixes=('','_assignment'))
        sites['assigned']=sites.functional_cluster.notna()
        previous=pd.read_csv(spec['audit']/'all_eligible_electrodes_reconstructed.csv')
        old_ids=previous.set_index('channel').electrode.to_dict()
        sites['electrode']=[old_ids.get(ch,subject+'|'+ch) for subject,ch in zip(sites.subject,sites.channel)]
        sites['eligible']=False;sites['paired_statistics_complete']=False;sites['native_qc_bad']=sites.native_QC_excluded
        sites['exclusion_reason']=np.where(sites.assigned,'','No current NMF assignment')
        sites.loc[sites.native_QC_excluded,'exclusion_reason']='Original task native QC exclusion'
        sites.loc[sites.subject_preprocessing_deferred,'exclusion_reason']='Author-deferred preprocessing'
        candidates=sites[sites.assigned & ~sites.native_QC_excluded & ~sites.subject_preprocessing_deferred]
        for subject,group in candidates.groupby('subject',sort=True):
            frames=[];missing=[]
            for level,condition,modality in zip(spec['levels'],spec['conditions'],spec['modalities']):
                ep_path,stat_path,issue=product(spec['task'],subject,spec['phase'],condition,modality)
                if issue:missing.append(level+':'+issue);continue
                epochs=mne.read_epochs(ep_path,preload=False,verbose='ERROR')
                names=[n if n.startswith(subject+'_') else subject+'_'+n for n in epochs.ch_names]
                bads={n if n.startswith(subject+'_') else subject+'_'+n for n in epochs.info['bads']}
                selected=(epochs.times>=0)&(epochs.times<=1)
                assert selected.sum()==129
                with h5py.File(stat_path,'r') as h:
                    sn=[n.decode() if isinstance(n,bytes) else str(n) for n in h['ch_names'][:]]
                    sn=[n if n.startswith(subject+'_') else subject+'_'+n for n in sn]
                    assert sn==names
                    masks=np.asarray(h['mask'][:],bool)[:,selected];p=np.asarray(h['pvals'][:],float)[:,selected]
                for row in group.itertuples():
                    if row.channel in names and row.channel not in bads:
                        index=names.index(row.channel)
                        frames.append(dict(channel=row.channel,level=level,available=True,
                            responsive=bool((masks[index] & (p[index]<.05)).any())))
                    else:frames.append(dict(channel=row.channel,level=level,available=False,responsive=False))
                source=dict(comparison=key,subject=subject,phase=spec['phase'],condition=condition,modality=modality,
                    level=level,epoch_path=str(ep_path),statistics_path=str(stat_path),
                    epoch_sha256=sha(ep_path),statistics_sha256=sha(stat_path),n_trials=len(epochs))
                sources.append(source)
                del epochs
            if missing:
                sites.loc[group.index,'exclusion_reason']=';'.join(missing)
                continue
            data=pd.DataFrame(frames)
            for row in group.itertuples():
                per=data[data.channel.eq(row.channel)]
                good=len(per)==2 and per.available.all()
                sites.loc[row.Index,'eligible']=good;sites.loc[row.Index,'paired_statistics_complete']=good
                sites.loc[row.Index,'exclusion_reason']='' if good else 'Target absent or native bad in a paired condition'
                if good:
                    for level in spec['levels']:
                        sites.loc[row.Index,level.lower()+'_responsive']=bool(per.loc[per.level.eq(level),'responsive'].iloc[0])
            active=sites.loc[group.index];active=active[active.eligible]
            if len(active):workers.append(dict(comparison=key,subject=subject,n_sites=len(active)))
            print('AUDIT',key,subject,'paired',len(active),'/',len(group),flush=True)
        out=RUN/key;out.mkdir(exist_ok=True)
        sites.to_csv(out/'anatomical_inventory.csv',index=False)
        eligible=sites[sites.eligible].copy()
        eligible['test_origin']=np.where(eligible.channel.isin(previous.channel),'Existing test pending source verification','New original-parameter test')
        eligible.to_csv(out/'cohort.csv',index=False)
        summaries[key]=dict(anatomical_sites=len(sites),anatomical_subjects=int(sites.subject.nunique()),
            current_assigned_task_sites=int(sites.assigned.sum()),paired_eligible_sites=len(eligible),
            paired_eligible_subjects=int(eligible.subject.nunique()),
            previous_test_candidates=int(eligible.channel.isin(previous.channel).sum()),
            new_test_candidates=int((~eligible.channel.isin(previous.channel)).sum()),
            excluded_reason_counts=sites.loc[~sites.eligible,'exclusion_reason'].value_counts().to_dict())
        print(key,summaries[key],flush=True)
    pd.DataFrame(workers).to_csv(RUN/'workers.csv',index=False)
    pd.DataFrame(sources).to_csv(RUN/'target_phase_sources.csv',index=False)
    write(RUN/'input_audit.json',dict(passed=True,version=VERSION,comparisons=summaries,
        source_cohort=str(COHORT),NMF=str(NMF),NMF_sha256=sha(NMF/'channel_assignments.csv'),
        coordinate_identity_not_reaudited=True,native_task_QC_preserved=True,
        completed_utc=datetime.now(timezone.utc).isoformat()))


def worker(index):
    """Validate/reuse old tests; compute only new/changed native comparisons."""
    assert os.environ.get('SLURM_JOB_ID'),'Run via Slurm'
    import h5py
    import mne
    import numpy as np
    import pandas as pd
    from ieeg.calc import stats
    from scripts.run_decision_repeat_delay_selection import PARAMS,install_reviewed_runtime_compatibility
    item=pd.read_csv(RUN/'workers.csv').iloc[index]
    key,subject=item.comparison,item.subject;spec=SPECS[key]
    dest=RUN/key/'subjects'/subject;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'completed.json').exists():
        done=json.loads((dest/'completed.json').read_text())
        assert all(sha(dest/name)==digest for name,digest in done['outputs'].items())
        print('Reused verified completion',key,subject,flush=True);return
    group=pd.read_csv(RUN/key/'cohort.csv');group=group[group.subject.eq(subject)].copy()
    group['electrode']=group.subject+'|'+group.channel
    old_tests=pd.read_csv(spec['source']/'timeperm.csv.gz',float_precision='round_trip')
    old_tests=old_tests[old_tests.subject.eq(subject)]
    old_waves=pd.read_csv(spec['waves'],float_precision='round_trip')
    old_waves=old_waves[old_waves.subject.eq(subject)]
    old_sources=pd.read_csv(spec['waves'].parent/'native_sources.csv')
    old_sources=old_sources[old_sources.subject.eq(subject)]
    target_sources=pd.read_csv(RUN/'target_phase_sources.csv')
    target_sources=target_sources[target_sources.comparison.eq(key)&target_sources.subject.eq(subject)]
    compatibility=install_reviewed_runtime_compatibility()
    reused=set(group.channel)&set(old_tests.channel)
    for level in spec['levels']:
        source=target_sources[target_sources.level.eq(level)].iloc[0]
        reference=old_sources[old_sources.phase.eq(spec['phase'])]
        reference=reference[reference.modality.eq(level)] if key=='picture' else reference[reference.condition.eq(level)]
        if len(reference)!=1 or not (reference.epoch_sha256.iloc[0]==source.epoch_sha256
            and reference.statistics_sha256.iloc[0]==source.statistics_sha256):
            reused=set()
    new=set(group.channel)-reused
    frames=[];sources=[];target_values={};target_times=None;trials=[]
    for phase in PHASES:
        lo,hi=spec['xlims'][phase]
        for level,condition,modality in zip(spec['levels'],spec['conditions'],spec['modalities']):
            ep_path,stat_path,issue=product(spec['task'],subject,phase,condition,modality)
            assert not issue,(key,subject,phase,level,issue)
            ep_hash,stat_hash=sha(ep_path),sha(stat_path)
            reference=old_sources[old_sources.phase.eq(phase)]
            reference=reference[reference.modality.eq(level)] if key=='picture' else reference[reference.condition.eq(level)]
            same_source=len(reference)==1 and reference.epoch_sha256.iloc[0]==ep_hash and reference.statistics_sha256.iloc[0]==stat_hash
            cached=old_waves[old_waves.phase.eq(phase)&old_waves.condition.eq(level)&old_waves.channel.isin(group.channel)&old_waves.time.between(lo,hi)].copy()
            old_channels=set(cached.channel) if same_source else set()
            if old_channels:
                cached=cached[cached.channel.isin(old_channels)].copy()
                cached['functional_cluster']=cached.channel.map(group.set_index('channel').functional_cluster)
                frames.append(cached)
            needed=(set(group.channel)-old_channels) | (new if phase==spec['phase'] else set())
            source=dict(comparison=key,subject=subject,phase=phase,condition=condition,level=level,modality=modality,
                epoch_path=str(ep_path),epoch_sha256=ep_hash,statistics_path=str(stat_path),statistics_sha256=stat_hash,
                reused_waveform_channels='|'.join(sorted(old_channels)),loaded_channels='|'.join(sorted(needed)))
            if needed:
                ep=mne.read_epochs(ep_path,preload=False,verbose='ERROR')
                names=[n if n.startswith(subject+'_') else subject+'_'+n for n in ep.ch_names]
                bads={n if n.startswith(subject+'_') else subject+'_'+n for n in ep.info['bads']}
                assert needed<=set(names) and not needed&bads,(key,subject,phase,level,'Native good-channel eligibility')
                channels=sorted(needed);picks=[names.index(ch) for ch in channels]
                times=ep.times.copy();display=(times>=lo)&(times<=hi)
                values=ep.get_data(picks=picks,copy=True,verbose='ERROR')[:,:,display]
                assert not np.isinf(values).any()
                with h5py.File(stat_path,'r') as handle:
                    sn=[n.decode() if isinstance(n,bytes) else str(n) for n in handle['ch_names'][:]]
                    sn=[n if n.startswith(subject+'_') else subject+'_'+n for n in sn]
                    assert sn==names
                    mask=np.asarray(handle['mask'][:],bool)[picks][:,display]
                means=np.nanmean(values,axis=0)
                for i,channel in enumerate(channels):
                    if channel not in old_channels:
                        frames.append(pd.DataFrame(dict(electrode=subject+'|'+channel,subject=subject,channel=channel,
                            functional_cluster=group.set_index('channel').loc[channel,'functional_cluster'],phase=phase,
                            condition=level,time=times[display],value=means[i],baseline_mask=mask[i],
                            n_finite_trials=np.isfinite(values[:,i]).sum(axis=0))))
                if phase==spec['phase']:
                    inverse={v:k for k,v in ep.event_id.items()}
                    events=np.array([inverse[int(n)] for n in ep.events[:,2]])
                    assert not any('bad' in n.lower().split('/') for n in events)
                    if key=='lexical':
                        retained=np.array(['/'.join(n.split('/')[4:])=='CORRECT' for n in events])
                    else:
                        from src.univariate.contrasts_mean import keep_correct_trials
                        selected=keep_correct_trials(pd.DataFrame({'event_name':events})).index.to_numpy()
                        np.testing.assert_array_equal(selected,np.arange(len(events)))
                        assert all(n.split('/')[0]=='Stimulus' for n in events)
                        retained=np.ones(len(events),bool)
                    assert retained.sum()>=2
                    test_window=(times[display]>=0)&(times[display]<=1)
                    current_times=times[display][test_window]
                    assert len(current_times)==129
                    if target_times is None:target_times=current_times
                    np.testing.assert_array_equal(current_times,target_times)
                    target_values[level]={ch:values[retained,i][:,test_window] for i,ch in enumerate(channels) if ch in new}
                    trials += [dict(subject=subject,condition=level,native_trial_index=i,event_name=n,retained_for_condition_test=bool(retained[i])) for i,n in enumerate(events)]
                source['n_trials']=len(ep)
                del ep,values
            sources.append(source)
    tested=[];first,second=spec['levels'];a,b=first.lower(),second.lower()
    for row in group.itertuples():
        if row.channel in reused:
            table=old_tests[old_tests.channel.eq(row.channel)].sort_values('time').copy()
            assert len(table)==129 and table.time.iloc[0]==0 and table.time.iloc[-1]==1
            origin='Reused original two-sided2000 test; native epoch/statistics hashes matched'
        else:
            x,y=target_values[first][row.channel],target_values[second][row.channel]
            print('TEST',key,subject,row.channel,'original2000 two-sided permutations',flush=True)
            mask,p=stats.time_perm_cluster(x,y,**PARAMS)
            observed=(np.isfinite(x).sum(axis=0)>=2)&(np.isfinite(y).sum(axis=0)>=2)
            assert not np.any(observed & ~np.isfinite(p))
            table=pd.DataFrame(dict(time=target_times,observed=observed,
                temporal_mask_two_sided=np.asarray(mask,bool),pointwise_p_two_sided=p,
                mean_difference=np.nanmean(x,axis=0)-np.nanmean(y,axis=0)))
            table['mean_'+a]=np.nanmean(x,axis=0);table['mean_'+b]=np.nanmean(y,axis=0)
            table['n_'+a]=len(x);table['n_'+b]=len(y)
            origin='New test using unchanged original parameters/trial/QC rules'
        table['electrode']=row.electrode;table['subject']=subject;table['channel']=row.channel
        table['functional_cluster']=row.functional_cluster;table['test_origin']=origin
        valid=table.observed&table.temporal_mask_two_sided&table.pointwise_p_two_sided.lt(.05)
        table[a+'_gt_'+b+'_mask']=valid&table.mean_difference.gt(0)
        table[b+'_gt_'+a+'_mask']=valid&table.mean_difference.lt(0)
        tested.append(table)
    waves=pd.concat(frames,ignore_index=True)
    # Strip historical functional labels: every row uses the selected224-site fit.
    waves['functional_cluster']=waves.channel.map(group.set_index('channel').functional_cluster)
    assert not waves.duplicated(['electrode','condition','phase','time']).any()
    assert set(waves.channel)==set(group.channel)
    waves.to_csv(dest/'waveforms.csv.gz',index=False)
    pd.concat(tested,ignore_index=True).to_csv(dest/'timeperm.csv.gz',index=False)
    pd.DataFrame(sources).to_csv(dest/'native_sources.csv',index=False)
    pd.DataFrame(trials).to_csv(dest/'trial_selection.csv',index=False)
    write(dest/'completed.json',dict(passed=True,subject=subject,comparison=key,n_sites=len(group),
        reused_tests=len(reused),new_tests=len(new),parameters=PARAMS,compatibility=compatibility,
        NMF_sha256=sha(NMF/'channel_assignments.csv'),
        outputs={name:sha(dest/name) for name in ('waveforms.csv.gz','timeperm.csv.gz','native_sources.csv','trial_selection.csv')},
        job_id=os.environ['SLURM_JOB_ID'],completed_utc=datetime.now(timezone.utc).isoformat()))
    print('WORKER COMPLETE',key,subject,'reused',len(reused),'new',len(new),flush=True)


def collect():
    assert os.environ.get('SLURM_JOB_ID'),'Run via Slurm'
    import numpy as np
    import pandas as pd
    workers=pd.read_csv(RUN/'workers.csv');summary={}
    assignments=pd.read_csv(NMF/'channel_assignments.csv')
    for key,spec in SPECS.items():
        items=workers[workers.comparison.eq(key)];waves=[];tests=[];sources=[];done=[]
        for item in items.itertuples():
            folder=RUN/key/'subjects'/item.subject
            receipt=json.loads((folder/'completed.json').read_text());assert receipt['passed']
            assert all(sha(folder/name)==digest for name,digest in receipt['outputs'].items())
            done.append(receipt)
            waves.append(pd.read_csv(folder/'waveforms.csv.gz',float_precision='round_trip'))
            tests.append(pd.read_csv(folder/'timeperm.csv.gz',float_precision='round_trip'))
            sources.append(pd.read_csv(folder/'native_sources.csv'))
        wave=pd.concat(waves,ignore_index=True);test=pd.concat(tests,ignore_index=True)
        assert not wave.duplicated(['electrode','phase','condition','time']).any()
        assert not test.duplicated(['electrode','time']).any()
        table=pd.read_csv(RUN/key/'cohort.csv');table['electrode']=table.subject+'|'+table.channel
        assert set(table.channel)==set(wave.channel)==set(test.channel)
        assert table.channel.isin(assignments.channel).all() and not table.native_qc_bad.any()
        current=assignments.set_index('channel').functional_cluster
        assert table.functional_cluster.tolist()==current.loc[table.channel].tolist()
        first,second=spec['levels'];a,b=first.lower(),second.lower()
        hits=test.groupby('channel')[[a+'_gt_'+b+'_mask',b+'_gt_'+a+'_mask']].any()
        table[a+'_gt_'+b]=table.channel.map(hits[a+'_gt_'+b+'_mask'])
        table[b+'_gt_'+a]=table.channel.map(hits[b+'_gt_'+a+'_mask'])
        table['response_union']=table[a+'_responsive']|table[b+'_responsive']
        table['difference_group']=np.select([
            table[a+'_gt_'+b]&table[b+'_gt_'+a],table[a+'_gt_'+b],table[b+'_gt_'+a]],
            ['bidirectional','first_only','second_only'],default='no_detected_difference')
        table['baseline_class']=np.select([table[a+'_responsive']&table[b+'_responsive'],
            table[a+'_responsive'],table[b+'_responsive']],['Both',first+' only',second+' only'],default='Neither')
        table.to_csv(RUN/key/'electrodes.csv',index=False)
        wave.to_csv(RUN/key/'waveforms.csv.gz',index=False);test.to_csv(RUN/key/'timeperm.csv.gz',index=False)
        pd.concat(sources,ignore_index=True).to_csv(RUN/key/'native_sources.csv',index=False)
        pool=table[table.response_union]
        counts=[]
        for name in ('first_only','no_detected_difference','second_only','bidirectional'):
            group=pool[pool.difference_group.eq(name)]
            group.to_csv(RUN/key/f'{name}_electrodes.csv',index=False)
            counts.append(dict(group=name,n_sites=len(group),n_subjects=int(group.subject.nunique()),
                **{c:int(group.functional_cluster.eq(c).sum()) for c in ('sensory','sustain','motor')}))
        pd.DataFrame(counts).to_csv(RUN/key/'group_counts.csv',index=False)
        old=pd.read_csv(spec['audit']/'all_eligible_electrodes_reconstructed.csv')
        old['in_new_paired_cohort']=old.channel.isin(table.channel)
        old['current_functional_cluster']=old.channel.map(current)
        old['functional_label_changed']=old.current_functional_cluster.notna()&old.current_functional_cluster.ne(old.functional_cluster)
        old.to_csv(RUN/key/'previous_cohort_changes.csv',index=False)
        summary[key]=dict(paired_sites=len(table),paired_subjects=int(table.subject.nunique()),
            active_sites=len(pool),active_subjects=int(pool.subject.nunique()),groups=counts,
            reused_tests=sum(x['reused_tests'] for x in done),new_tests=sum(x['new_tests'] for x in done),
            current_NMF_labels_used=True,baseline_mask_and_pointwise_p05=True)
    protected=json.loads((RUN/'protected_files.json').read_text())
    assert all(sha(path)==digest for path,digest in protected.items())
    write(RUN/'data_completed.json',dict(passed=True,comparisons=summary,version=VERSION,
        protected_files=len(protected),NMF_dir=str(NMF),NMF_refit=False,
        condition_test_parameters=json.loads((RUN/'picture/subjects'/workers[workers.comparison.eq('picture')].iloc[0].subject/'completed.json').read_text())['parameters'],
        job_id=os.environ['SLURM_JOB_ID'],completed_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(summary,indent=2),flush=True)


def group_counts():
    import pandas as pd
    result={}
    for key in SPECS:
        table=pd.read_csv(RUN/key/'group_counts.csv')
        result.update({(key,row.group):(int(row.n_sites),int(row.n_subjects)) for row in table.itertuples()})
    return result


def weight_attributes():
    counts=group_counts()
    groups=(('picture','no_detected_difference',*counts[('picture','no_detected_difference')],
             'Sound–Image\nNo detected difference'),
            ('lexical','first_only',*counts[('lexical','first_only')],'Lexical Delay\nDecision > Repeat'))
    return dict(NMF_DIR=NMF,GROUP_COUNTS=counts,GROUPS=groups)


def load_groups(key):
    import pandas as pd
    from src.figure3 import CONDITION_SPECS
    table=pd.read_csv(RUN/key/'electrodes.csv')
    pool=table[table.response_union].copy()
    assert pool.eligible.all() and not pool.native_qc_bad.any()
    assert pool.functional_cluster.isin(('sensory','sustain','motor')).all()
    groups={name:pool[pool.difference_group.eq(name)].copy() for name in
        ('first_only','no_detected_difference','second_only','bidirectional')}
    spec=CONDITION_SPECS[key]
    titles=dict(first_only=spec['names'][0]+' only',no_detected_difference='No detected difference',
        second_only=spec['names'][1]+' only',bidirectional='Bidirectional')
    target=ROOT/'results/fig3/regions'/key;target.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name,points in groups.items():
        points.to_csv(target/f'{name}_electrodes.csv',index=False)
        rows.append(dict(group=name,label=titles[name],n_electrodes=len(points),n_participants=int(points.subject.nunique()),
            **{c:int(points.functional_cluster.eq(c).sum()) for c in ('sensory','sustain','motor')}))
    counts=pd.DataFrame(rows);counts.to_csv(target/'group_counts.csv',index=False)
    pool.to_csv(target/'responsive_pool_electrodes.csv',index=False)
    write(target/'selection_provenance.json',dict(passed=True,version=VERSION,
        parent_n=len(table),parent_subjects=int(table.subject.nunique()),pool_n=len(pool),pool_subjects=int(pool.subject.nunique()),
        groups=rows,NMF_dir=str(NMF),NMF_assignments_sha256=sha(NMF/'channel_assignments.csv'),
        source_electrodes_sha256=sha(RUN/key/'electrodes.csv'),
        baseline_rule='Native mask AND pointwise p<.05 in the unchanged0–1-s target phase',
        center_meaning='Neither significant direction detected; not equivalence or baseline intersection',
        exclusive_partition=True,original_test_parameters=json.loads((RUN/'data_completed.json').read_text())['condition_test_parameters']))
    assert sum(len(frame) for frame in groups.values())==len(pool)
    return pool,groups,counts,titles


def render_maps(key,groups,titles):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from src.figure3 import CONDITION_SPECS
    from src.figure3_brain import brain_pair
    from src.main_analysis import FUNCTIONAL_COMPONENTS,COMPONENT_COLORS
    paths={};dest=ROOT/'results/fig3/regions'/key
    labels=dict(sensory='Sensory',sustain='Sustained',motor='Motor')
    for name,points in groups.items():
        path=ROOT/'img/fig3'/f'fig3_{key}_{name}_functional_brain.svg'
        if points.empty:
            path.unlink(missing_ok=True)
            for suffix in ('display_coordinates.csv','contours.csv','map_provenance.json'):
                (dest/f'{name}_{suffix}').unlink(missing_ok=True)
            continue
        points=points.copy();points['display_group']=points.functional_cluster
        assert np.isfinite(points[['x','y','z']]).all().all() and points.x.ne(0).all()
        fig,coordinates,rings=brain_pair(points,list(FUNCTIONAL_COMPONENTS),COMPONENT_COLORS,labels,
            title=f'{titles[name]} · n={len(points)}\n{CONDITION_SPECS[key]["title"]}')
        fig.savefig(path,format='svg',bbox_inches='tight',pad_inches=.03);plt.close(fig)
        pd.DataFrame(coordinates).to_csv(dest/f'{name}_display_coordinates.csv',index=False)
        contours=pd.DataFrame(rings);contours.to_csv(dest/f'{name}_contours.csv',index=False)
        assert contours.loc[contours.n_sites.lt(3),'n_contours'].eq(0).all()
        write(dest/f'{name}_map_provenance.json',dict(passed=True,version=VERSION,n_sites=len(points),
            n_subjects=int(points.subject.nunique()),coordinates='Unchanged current NMF input coordinates',
            NMF_dir=str(NMF),renderer='Original Figure1 local-insula surface, camera-facing snap and4-mm lift',
            point_size_pixels=24,contour='Original Figure2 KDE, minimum3 per component/hemisphere',
            component_colors=COMPONENT_COLORS,svg=str(path),svg_sha256=sha(path)))
        paths[name]=str(path);print('RENDERED CURRENT',key,name,len(points),flush=True)
    return paths


def prepare_temporal(key):
    import numpy as np
    import pandas as pd
    from src.supplementary import condition_region_time_profiles as temporal
    from src.supplementary.lexical_condition_figures import summarize_traces
    components=('sensory','sustain','motor')
    spec=dict(temporal.SPECS[key])
    region='no_detected_difference' if key=='picture' else 'first_only'
    selected_path=ROOT/'results/fig3/regions'/key/f'{region}_electrodes.csv'
    selected=pd.read_csv(selected_path)
    spec.update(n=len(selected),subjects=int(selected.subject.nunique()),
        components={c:int(selected.functional_cluster.eq(c).sum()) for c in components},
        waves=str((RUN/key/'waveforms.csv.gz').relative_to(ROOT)))
    waves=pd.read_csv(RUN/key/'waveforms.csv.gz',float_precision='round_trip')
    waves=waves[waves.electrode.isin(selected.electrode)].drop(columns='functional_cluster',errors='ignore')
    waves=pd.concat([waves[waves.phase.eq(phase)&waves.time.between(*spec['xlims'][phase])] for phase in PHASES],ignore_index=True)
    assert np.isfinite(waves.value).all() and waves.n_finite_trials.gt(0).all()
    traces=summarize_traces(selected,waves,[('Selected spatial region',selected)])
    assert traces.n_finite_electrodes.eq(len(selected)).all()
    ylim,ylim_proof=temporal.shared_ylim(waves,selected,spec)
    first=waves[waves.baseline_mask].groupby('electrode').time.min()
    # Same original first-significant-time rule over the displayed phases and
    # conditions, now evaluated within the current functional assignments.
    # Keep prior within-component order for tied retained rows.
    with np.load(ROOT/spec['orders'],allow_pickle=False) as prior:
        old_orders={c:prior[f'stimulus_{c}_electrodes'].tolist() for c in components}
    orders={}
    for component in components:
        allowed=selected.loc[selected.functional_cluster.eq(component),'electrode'].tolist()
        tie={site:i for i,site in enumerate(old_orders[component])}
        orders[component]=sorted(allowed,key=lambda site:(first.get(site,np.inf),tie.get(site,len(tie)),site))
    matrices={};payload={};audit=[]
    for phase in PHASES:
        part=waves[waves.phase.eq(phase)];times=np.sort(part.time.unique());payload[phase+'_time']=times
        pairs={level:part[part.condition.eq(level)].pivot(index='electrode',columns='time',values='value') for level in spec['conditions']}
        mask=part[part.condition.eq(spec['conditions'][0])].pivot(index='electrode',columns='time',values='baseline_mask')
        for component in components:
            order=orders[component];prefix=f'{phase}_{component}'
            x,y=[pairs[level].reindex(index=order,columns=times).to_numpy(float) for level in spec['conditions']]
            assert np.isfinite(x).all() and np.isfinite(y).all()
            raw=(x+y)/2 if key=='picture' else x-y
            display=np.ones(raw.shape,bool) if key=='picture' else mask.reindex(index=order,columns=times).to_numpy(bool)
            matrices[(phase,component)]=np.where(display,raw,np.nan)
            payload.update({prefix:matrices[(phase,component)],prefix+'_unmasked_values':raw,
                prefix+'_display_mask':display,prefix+'_first_condition':x,prefix+'_second_condition':y,
                prefix+'_electrodes':np.asarray(order,dtype=str)})
            audit += [dict(phase=phase,functional_cluster=component,row=i,electrode=site,
                n_paired_bins=len(times),n_displayed_bins=int(display[i].sum())) for i,site in enumerate(order)]
    dest=ROOT/'results/fig3/time_profiles'/key;dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'heatmap_matrices.npz',**payload)
    traces.to_csv(dest/'selected_electrode_mean_sem.csv',index=False)
    selected.to_csv(dest/'selected_electrodes.csv',index=False)
    pd.DataFrame(audit).to_csv(dest/'row_and_value_audit.csv',index=False)
    write(dest/'data_provenance.json',dict(passed=True,version=VERSION,n_sites=len(selected),
        n_participants=int(selected.subject.nunique()),component_counts=spec['components'],metric=spec['metric'],
        exact_same_sites_as_brain_map=True,mean_SEM_unit='electrode',phase_windows=spec['xlims'],
        row_order='Original first native-baseline-significant time across displayed phases/conditions; prior order breaks retained ties',
        display_mask='Complete paired average, no significance blanking' if key=='picture' else 'Original Decision-baseline mask',
        color_limits=[0,.5],negative_values_preserved_and_displayed_white=True,
        no_old_pooled_significance_bars_reused=True,original_shared_y_helper=ylim_proof,
        source_sha256={str(selected_path):sha(selected_path),str(RUN/key/'waveforms.csv.gz'):sha(RUN/key/'waveforms.csv.gz')}))
    return spec,selected,traces,orders,matrices,payload,ylim


def render_main():
    assert os.environ.get('SLURM_JOB_ID'),'Run via Slurm'
    from src.figure3 import style,override
    from src.supplementary import condition_difference_group_panels as spatial
    from src.supplementary import condition_region_time_profiles as temporal
    import pandas as pd
    style();(ROOT/'img/fig3').mkdir(exist_ok=True)
    data=json.loads((RUN/'data_completed.json').read_text());assert data['passed']
    from scripts.run_sound_image_selection import PARAMS as PICTURE_PARAMETERS
    from scripts.run_decision_repeat_delay_selection import PARAMS as LEXICAL_PARAMETERS
    assert PICTURE_PARAMETERS==LEXICAL_PARAMETERS==data['condition_test_parameters']
    results={};paths=[]
    for key in SPECS:
        pool,groups,counts,titles=load_groups(key)
        diagram=spatial.diagram(key,pool,groups)
        brains=render_maps(key,groups,titles)
        region='no_detected_difference' if key=='picture' else 'first_only'
        present=tuple(c for c in ('sensory','sustain','motor') if groups[region].functional_cluster.eq(c).any())
        with override(temporal,prepare=prepare_temporal,CLUSTERS=present):
            portrait=temporal.render(key)
        results[key]=dict(diagram=str(diagram),brains=brains,counts=counts.to_dict('records'),
            pool_n=len(pool),pool_subjects=int(pool.subject.nunique()),time_portrait=str(portrait),
            paired_sites=data['comparisons'][key]['paired_sites'])
        paths += [str(diagram),str(portrait),*brains.values()]
        # Keep the explicit statistical cohort as a main-Figure3 artifact.
        dest=ROOT/'results/fig3/cohorts'/key;dest.mkdir(parents=True,exist_ok=True)
        pd.read_csv(RUN/key/'electrodes.csv').to_csv(dest/'eligible_assigned_electrodes.csv',index=False)
    write(ROOT/'results/fig3/manifest.json',dict(passed=True,version=VERSION,job_id=os.environ['SLURM_JOB_ID'],
        entry_notebook=str(ROOT/'vizpub/fig3.ipynb'),comparisons=results,svg_sha256={path:sha(path) for path in paths},
        source_cohort=str(COHORT),NMF_dir=str(NMF),NMF_assignments_sha256=sha(NMF/'channel_assignments.csv'),
        functional_components=['sensory','sustain','motor'],unassigned_excluded=True,
        manual14_excluded_from_functional_analyses=True,native_QC_preserved=True,
        reused_condition_tests=sum(s['reused_tests'] for s in data['comparisons'].values()),
        new_original_parameter_tests=sum(s['new_tests'] for s in data['comparisons'].values()),
        condition_test_parameters=data['condition_test_parameters'],no_NMF_refit=True,
        temporal_scope='Current PN no-difference paired average and LD Decision>Repeat paired difference',
        supplementary_version='Prior supplementary notebook/results retained separately'))
    return results


def build_notebook():
    """Keep all author-selected main panels, replacing cohort-dependent prose."""
    import nbformat
    data=json.loads((RUN/'data_completed.json').read_text())['comparisons']
    counts=group_counts();nb=nbformat.read(RUN/'before/vizpub/fig3.ipynb',as_version=4)
    byid={c.id:c for c in nb.cells}
    def label(task,region):
        n,s=counts[(task,region)];return f'{n} electrodes / {s} participants'
    p,l=data['picture'],data['lexical']
    nb.cells[0].source=f'''# Figure 3 · Input modality and task demands

**Current version: {VERSION}.** Start with the completed378-site/67-participant anatomical inventory. Functional labels and continuous weights come exclusively from the author-selected224-site three-component Figure2 model. Manual14 exclusions and Unassigned sites do not enter functional statistics. Task-native QC and paired condition completeness give **Picture Naming {p['paired_sites']}/{p['paired_subjects']} participants** and **Lexical Delay {l['paired_sites']}/{l['paired_subjects']} participants**.

**Picture Naming · Stimulus [0,1] s:** either-condition native baseline-response pool {p['active_sites']} sites/{p['active_subjects']} participants. Exclusive Sound > Image {counts[('picture','first_only')][0]}, no detected difference {counts[('picture','no_detected_difference')][0]}, Image > Sound {counts[('picture','second_only')][0]}, bidirectional {counts[('picture','bidirectional')][0]} separately. **Lexical Delay · Delay [0,1] s:** pool {l['active_sites']}/{l['active_subjects']}; Decision > Repeat {counts[('lexical','first_only')][0]}, no detected difference {counts[('lexical','no_detected_difference')][0]}, Repeat > Decision {counts[('lexical','second_only')][0]}, bidirectional {counts[('lexical','bidirectional')][0]} separately.

The center means **no detected significant condition difference**, not equivalence or a literal baseline-response intersection. Preserve conventional overlapping circles and count-proportional areas at the original common scale. Maps use the original local-insula surface,24-pixel points,4-mm display lift and unchanged Figure2 density contours, colored by current functional assignments.

Baseline eligibility retains native mask AND pointwise p<.05. Direct comparisons retain the original independent two-sided2000-permutation time_perm_cluster, .05/.05, seed42 and original trial handling (PN retained native trials; LD exact CORRECT trials for the direct test). {p['reused_tests']+l['reused_tests']} tests were reused after native input hash verification; {p['new_tests']+l['new_tests']} required the unchanged original test. No NMF refit or parameter search.

PN temporal portrait: current no-difference sites, both electrode mean±SEM curves above and complete (Sound+Image)/2 below. LD: current Decision>Repeat sites, both curves above and Decision−Repeat below with the original Decision-baseline display mask. Retain original phase windows, proportional widths,1-mm rows, colors and0–.5 heatmaps. Recompute row membership/order using the original first-significant-time rule and current assignments; no old-population significance bars are reused.

All seventeen original main panel types, including both paired ternaries and density overlays, remain executable below; empty condition classes are explicitly labeled0. Main SVGs: `img/fig3/`; versioned data/provenance: `results/{VERSION}/`. Supplementary notebooks retain their separately identified prior versions.
'''
    titles={5:('picture','first_only','Sound > Image only'),7:('picture','no_detected_difference','No detected difference'),
        11:('picture','second_only','Image > Sound only'),13:('picture','bidirectional','Bidirectional'),
        17:('lexical','first_only','Decision > Repeat'),21:('lexical','no_detected_difference','No detected difference'),
        23:('lexical','second_only','Repeat > Decision')}
    for index,(key,region,text) in titles.items():
        nb.cells[index].source=f'### {text} · n={counts[(key,region)][0]}'
        code=nb.cells[index+1]
        code.source=f"if '{region}' in results['{key}']['brains']:\n    display(SVG(filename=results['{key}']['brains']['{region}']))\nelse:\n    print('{text}: n=0; no electrodes to plot.')\n"
    # A newly present reverse-direction class is displayed, never silently dropped.
    if counts[('lexical','bidirectional')][0]:
        insert=next(i for i,c in enumerate(nb.cells) if c.id=='fig3-weight-title')
        nb.cells[insert:insert]=[nbformat.v4.new_markdown_cell('### Lexical Delay · Bidirectional',id='fig3-lexical-bidirectional-md'),
            nbformat.v4.new_code_cell("display(SVG(filename=results['lexical']['brains']['bidirectional']))",id='fig3-lexical-bidirectional')]
    byid={c.id:c for c in nb.cells}
    byid['fig3-weight-title'].source=f'''## Continuous functional-component weights · PN and Lexical Delay

Current Figure2 NMF W proportions: PN no-detected-difference {label('picture','no_detected_difference')}; LD Decision > Repeat {label('lexical','first_only')}. Divide each electrode's three current loadings by their sum. Preserve continuous blue/gold/red point mixtures, gray sigma0.09 clouds and one shared density reference, centroid-to-vertex guides, no mean diamonds or vertex100% labels. These are descriptive composition displays, not new inference or a learned embedding.
'''
    byid['fig3-weight-sound-title'].source=f'''### Sound > Image only · Stimulus0–1 s

{label('picture','first_only')} from the current mutually exclusive first-only class. Use the same current W proportions, sigma0.09 and reference density peak. Bidirectional sites remain separate.
'''
    byid['fig3-weight-picture-pair-title'].source=f'''### Picture Naming · No detected difference and Sound > Image

{label('picture','no_detected_difference')} beside {label('picture','first_only')}. Both panels reuse the current exported coordinates, component point colors and sigma0.09 density arrays on one shared reference scale.
'''
    byid['fig3-weight-lexical-pair-title'].source=f'''### Lexical Delay · No detected difference and Decision > Repeat

{label('lexical','no_detected_difference')} beside {label('lexical','first_only')}. Both use the same current NMF loading proportions, original drawing and shared sigma0.09 reference density scale.
'''
    return nb


def execute_notebook():
    assert os.environ.get('SLURM_JOB_ID'),'Run via Slurm'
    import ast
    import sys
    import nbformat
    import numpy as np
    import pandas as pd
    from nbclient import NotebookClient
    import xml.etree.ElementTree as ET
    data=json.loads((RUN/'data_completed.json').read_text());assert data['passed']
    protected=json.loads((RUN/'protected_files.json').read_text())
    assert all(sha(path)==digest for path,digest in protected.items())
    config=ROOT/'config'/f'{VERSION}.json'
    write(config,dict(version=VERSION,scope='Main Figure3 notebook and all its current panels',
        source_cohort=str(COHORT),NMF_dir=str(NMF),sampled_sites=378,sampled_subjects=67,
        functional_source_sites=224,functional_source_subjects=60,comparisons=data['comparisons'],
        no_NMF_refit=True,condition_test_parameters=data['condition_test_parameters'],
        supplementary_version='Preserved prior main/SI separation; old SI is not claimed as refreshed'))
    nb=build_notebook()
    for i,c in enumerate(nb.cells):
        if c.cell_type=='code':ast.parse(c.source,filename=f'fig3-current-cell-{i}')
    nbformat.validate(nb)
    jupyter=RUN/'jupyter';name='insula-figure3-manual14';spec=jupyter/'kernels'/name;spec.mkdir(parents=True,exist_ok=True)
    write(spec/'kernel.json',dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        display_name='Figure3 current manual14 (ieeg)',language='python'))
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    os.environ['JUPYTER_PATH']=str(jupyter)+os.pathsep+os.environ.get('JUPYTER_PATH','')
    def start(cell,cell_index,**kwargs):
        print(datetime.now(timezone.utc).isoformat(),'CELL',cell_index,cell.id,flush=True)
    try:
        NotebookClient(nb,kernel_name=name,timeout=1800,iopub_timeout=180,raise_on_iopub_timeout=True,
            resources={'metadata':{'path':str(ROOT)}},on_cell_start=start,record_timing=True).execute()
    except Exception:
        nbformat.write(nb,RUN/'fig3_failed.ipynb');raise
    assert not [o for c in nb.cells for o in c.get('outputs',[]) if o.get('output_type')=='error' or 'image/png' in o.get('data',{})]
    assert all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')
    manifest_path=ROOT/'results/fig3/manifest.json';manifest=json.loads(manifest_path.read_text())
    assert manifest['version']==VERSION and manifest['NMF_dir']==str(NMF)
    assignments=pd.read_csv(NMF/'channel_assignments.csv')
    selected=set(assignments.channel)
    exclusion=set(json.loads((ROOT/'config/insula_manual14_2026-11-07.json').read_text())['excluded_sites'])
    for key,comparison in manifest['comparisons'].items():
        table=pd.read_csv(RUN/key/'electrodes.csv');pool=table[table.response_union]
        sets=[]
        for region in ('first_only','no_detected_difference','second_only','bidirectional'):
            points=pd.read_csv(ROOT/'results/fig3/regions'/key/f'{region}_electrodes.csv')
            assert set(points.channel)<=selected and not set(points.channel)&exclusion
            assert points.functional_cluster.tolist()==assignments.set_index('channel').loc[points.channel,'functional_cluster'].tolist()
            sets.append(set(points.channel))
        assert set.union(*sets)==set(pool.channel)
        assert sum(map(len,sets))==len(set.union(*sets))==comparison['pool_n']
        region='no_detected_difference' if key=='picture' else 'first_only'
        selected_table=pd.read_csv(ROOT/'results/fig3/time_profiles'/key/'selected_electrodes.csv')
        expected=pd.read_csv(ROOT/'results/fig3/regions'/key/f'{region}_electrodes.csv')
        assert set(selected_table.channel)==set(expected.channel)
        with np.load(ROOT/'results/fig3/time_profiles'/key/'heatmap_matrices.npz',allow_pickle=False) as arrays:
            for phase in PHASES:
                for component in ('sensory','sustain','motor'):
                    prefix=f'{phase}_{component}'
                    x,y=arrays[prefix+'_first_condition'],arrays[prefix+'_second_condition']
                    raw=(x+y)/2 if key=='picture' else x-y
                    np.testing.assert_allclose(raw,arrays[prefix+'_unmasked_values'],rtol=0,atol=0)
                    np.testing.assert_allclose(np.where(arrays[prefix+'_display_mask'],raw,np.nan),arrays[prefix],rtol=0,atol=0,equal_nan=True)
        # Current continuous weights must agree exactly with this model's W.
    weights=ROOT/'results/fig3/component_weights'
    for folder in (weights,weights/'sound_gt_image'):
        coords=pd.read_csv(folder/'ternary_coordinates.csv',float_precision='round_trip')
        w=assignments.set_index('channel').loc[coords.channel,[f'loading_{c}' for c in ('sensory','sustain','motor')]].to_numpy(float)
        expected=w/w.sum(axis=1,keepdims=True)
        np.testing.assert_allclose(coords[['sensory','sustain','motor']],expected,rtol=0,atol=1e-12)
    assert set(map(str,(ROOT/'img/fig3').glob('*.svg')))==set(manifest['svg_sha256'])
    for path,digest in manifest['svg_sha256'].items():
        assert sha(path)==digest and ET.parse(path).getroot().tag.endswith('svg')
    assert all(sha(path)==digest for path,digest in protected.items())
    nb.metadata['insula_figure3']=dict(version=VERSION,NMF_dir=str(NMF),functional_source_sites=224,
        native_QC_preserved=True,all_code_cells_executed=True,executed_utc=datetime.now(timezone.utc).isoformat())
    path=ROOT/'vizpub/fig3.ipynb';temp=path.with_suffix('.executed.tmp.ipynb');nbformat.write(nb,temp);temp.replace(path)
    receipt=dict(passed=True,version=VERSION,job_id=os.environ['SLURM_JOB_ID'],
        code_cells=sum(c.cell_type=='code' for c in nb.cells),all_code_cells_executed=True,svg_panels=len(manifest['svg_sha256']),
        comparisons=data['comparisons'],NMF_dir=str(NMF),NMF_refit=False,
        reused_tests=manifest['reused_condition_tests'],new_original_parameter_tests=manifest['new_original_parameter_tests'],
        native_QC_preserved=True,all_functional_sites_current_assigned=True,all_weight_proportions_current_W=True,
        main_temporal_memberships_and_signed_values_verified=True,protected_files_unchanged=len(protected),
        supplementary_preserved=True,notebook_sha256=sha(path),completed_utc=datetime.now(timezone.utc).isoformat())
    write(ROOT/'results/fig3/execution_validation.json',receipt);write(RUN/'completed.json',receipt)
    registry=ROOT/'config/current_insula_analysis.json';value=json.loads(registry.read_text())
    value['current_figures'].pop('fig3_to_fig6',None)
    value['current_figures']['fig3']=dict(version=VERSION,notebook=str(path),NMF_dir=str(NMF),
        paired_sites={k:v['paired_sites'] for k,v in data['comparisons'].items()},
        active_sites={k:v['active_sites'] for k,v in data['comparisons'].items()})
    value['current_figures']['fig4_to_fig6']='Preserved prior output/model versions; not refreshed for the224-site model'
    value['status']='adopted_manual14_and_refreshed_figures1_to3'
    write(registry,value)
    print(json.dumps(receipt,indent=2),flush=True)

