"""Original PN sound/image method on the approved full Stimulus [0,1] cohort."""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import inspect
import json
import os

import numpy as np
import pandas as pd

from scripts.refresh_lexical_condition_sets import sha,write
from scripts.run_sound_image_selection import PARAMS,install_reviewed_runtime_compatibility
from scripts.run_decision_repeat_delay_selection import contiguous_intervals

ROOT=Path('/cwork/ns458/insula-paper')
OUT=ROOT/'results/supplementary/fig3_picture_stimulus_condition_sets'
AUDIT=ROOT/'results/audit/picture_condition_sets_full_2026-10-27'
COHORT=ROOT/'results/audit/fig3_all_anatomical_denominators_2026-10-23'
IMG=ROOT/'img/supplementary/fig3_picture_stimulus_condition_sets'
NB=ROOT/'vizpub/supplementary/fig3_picture_stimulus_condition_sets.ipynb'
GROUPS=('Sound only','Both','Image only','Neither')
COMPONENTS=('sensory','sustain','motor','Unassigned')


def protected():
    files=json.loads((AUDIT/'protected.json').read_text())
    changed=[p for p,h in files.items() if not Path(p).is_file() or sha(p)!=h]
    assert not changed,f'Protected files changed: {changed}'
    return len(files)


def setup():
    from ieeg.calc import stats
    AUDIT.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True);IMG.mkdir(parents=True,exist_ok=True)
    if not (AUDIT/'protected.json').exists():
        paths=list((ROOT/'vizpub').glob('fig[1-6].ipynb'))
        paths += [ROOT/'vizpub/supplementary/fig3_lexical_delay_condition_sets.ipynb',ROOT/'vizpub/supplementary/sound_image_test.ipynb']
        for folder in ['img/fig1','img/fig2','results/nmf',
                'img/supplementary/fig3_lexical_delay_condition_sets','img/supplementary/sound_image_test']:
            paths += [p for p in (ROOT/folder).rglob('*') if p.is_file()]
        for folder in ['results/supplementary/sound_image_test/stimulus_electrode_selection',
                'results/supplementary/fig3_lexical_delay_condition_sets/full_cohort_2026-10-27']:
            paths += [p for p in (ROOT/folder).glob('*') if p.is_file()]
        paths += [COHORT/'PictureNaming_all_anatomical_sites.csv',COHORT/'native_product_sources.csv']
        write(AUDIT/'protected.json',{str(p):sha(p) for p in sorted(set(paths))})
    protected()
    audit=pd.read_csv(COHORT/'PictureNaming_all_anatomical_sites.csv')
    sites=audit.loc[audit.eligible].copy()
    assert len(audit)==162 and len(sites)==128 and sites.subject.nunique()==27
    assert sites.electrode.is_unique and not sites.native_qc_bad.any()
    assignments=pd.read_csv(ROOT/'results/nmf/channel_assignments.csv')
    fields=['subject','channel','functional_cluster','x','y','z','x_native','y_native','z_native']
    sites=sites.merge(assignments[fields],on=['subject','channel'],how='left',validate='one_to_one')
    assert sites.functional_cluster.notna().sum()==81
    sites['functional_cluster']=sites.functional_cluster.fillna('Unassigned')
    coordinate_path=ROOT/'results/audit/sound_only_cohort_2026-10-17/all_coordinate_rows.pkl.gz'
    raw=pd.read_pickle(coordinate_path);raw=raw.loc[raw.task.eq('PictureNaming')]
    coords=[]
    for row in sites.itertuples():
        candidates=raw.loc[raw.subject.eq(row.subject)&raw.channel.eq(row.channel)]
        assert len(candidates),row.channel
        xyz=candidates[['x','y','z']].to_numpy(float)
        assert np.isfinite(xyz).all() and np.allclose(xyz,xyz[0],atol=1e-5,rtol=0)
        native=candidates[['x_native','y_native','z_native']].to_numpy(float)[0]
        if row.functional_cluster!='Unassigned':
            np.testing.assert_allclose([row.x_native,row.y_native,row.z_native],native,atol=1e-5,rtol=0)
            chosen=[row.x,row.y,row.z]
            origin='Existing frozen NMF template coordinates; native identity verified'
        else:
            chosen=xyz[0];origin='Frozen PictureNaming task coordinates; no NMF label assigned'
        coords.append(dict(electrode=row.electrode,x=chosen[0],y=chosen[1],z=chosen[2],
            x_native=native[0],y_native=native[1],z_native=native[2],coordinate_origin=origin))
    sites=sites.drop(columns=['x','y','z','x_native','y_native','z_native']).merge(pd.DataFrame(coords),on='electrode',validate='one_to_one')
    assert np.isfinite(sites[['x','y','z']]).all().all() and sites.x.ne(0).all()
    sites['baseline_class']=np.select([sites.sound_responsive & ~sites.image_responsive,
        sites.sound_responsive & sites.image_responsive,~sites.sound_responsive & sites.image_responsive],GROUPS[:3],default='Neither')
    assert sites.baseline_class.value_counts().to_dict()=={'Neither':79,'Sound only':25,'Both':13,'Image only':11}
    sites=sites.sort_values(['subject','channel']).reset_index(drop=True)
    sites.to_csv(OUT/'cohort.csv',index=False)
    sites.groupby('subject',sort=True).size().rename('n_sites').reset_index().to_csv(OUT/'subjects.csv',index=False)
    source=pd.read_csv(COHORT/'native_product_sources.csv')
    source=source.loc[source.task.eq('PictureNaming')&source.phase.eq('Stimulus')&source.subject.isin(sites.subject)].copy()
    assert len(source)==54
    source.to_csv(OUT/'audited_native_sources.csv',index=False)
    write(AUDIT/'run_manifest.json',dict(authorized_scope='PN Stimulus [0,1] analogous to full Lexical panels',
        n_sites=128,n_participants=27,n_nmf=81,n_unassigned=47,window=[0,1],parameters=PARAMS,
        task='PictureNaming',description='Repeat',modalities=['sound','image'],
        trial_rule='Original PN native-QC ListenSpeak trials; no lexical CORRECT-token gate',
        exact_Stimulus_repairs_take_precedence=True,old_0_0_6_tests_reused=False,
        cohort_sha256=sha(OUT/'cohort.csv'),audited_source_sha256=sha(COHORT/'native_product_sources.csv'),
        stats_source=inspect.getsourcefile(stats.time_perm_cluster),stats_sha256=sha(inspect.getsourcefile(stats.time_perm_cluster)),
        job_id=os.environ['SLURM_JOB_ID'],started_utc=datetime.now(timezone.utc).isoformat()))
    print(sites.groupby('subject').size().to_string(),flush=True)


def worker(index):
    import mne,h5py
    from ieeg.calc import stats
    from src.univariate.contrasts_mean import keep_correct_trials
    protected()
    run=json.loads((AUDIT/'run_manifest.json').read_text())
    assert sha(inspect.getsourcefile(stats.time_perm_cluster))==run['stats_sha256']
    compat=install_reviewed_runtime_compatibility()
    subject=pd.read_csv(OUT/'subjects.csv').iloc[index].subject
    sites=pd.read_csv(OUT/'cohort.csv');sites=sites.loc[sites.subject.eq(subject)].copy()
    dest=OUT/'subjects'/subject;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'completed.json').exists():
        previous=json.loads((dest/'completed.json').read_text())
        assert previous['cohort_sha256']==sha(OUT/'cohort.csv')
        assert all(sha(dest/n)==h for n,h in previous['outputs'].items())
        print(subject,'verified existing outputs',flush=True);return
    sources=pd.read_csv(OUT/'audited_native_sources.csv')
    values={};native_sources=[];trials=[]
    for modality in ['sound','image']:
        source=sources.loc[sources.subject.eq(subject)&sources.modality.eq(modality)]
        assert len(source)==1
        source=source.iloc[0]
        assert source.condition=='Repeat' and sha(source.statistics_path)==source.statistics_sha256
        validation=json.loads(Path(source.validation).read_text());assert validation['validation_passed']
        epochs=mne.read_epochs(source.epoch_path,preload=False,verbose='error')
        channels=sites.channel.tolist()
        assert set(channels)<=set(epochs.ch_names) and not set(channels)&set(epochs.info['bads'])
        inverse={v:k for k,v in epochs.event_id.items()}
        names=np.asarray([inverse[int(n)] for n in epochs.events[:,2]])
        assert not any('bad' in name.lower().split('/') for name in names)
        # Preserve the original PN event parser and its retained native trials.
        retained=keep_correct_trials(pd.DataFrame({'event_name':names})).index.to_numpy()
        np.testing.assert_array_equal(retained,np.arange(len(names)))
        assert all(name.split('/')[0]=='Stimulus' for name in names), (subject,modality,'Non-onset event in native Stimulus epochs')
        keep=(epochs.times>=0)&(epochs.times<=1)
        times=epochs.times[keep]
        assert len(times)==129 and times[0]==0 and times[-1]==1
        data=epochs.get_data(picks=channels,copy=True,verbose='error')[:,:,keep]
        assert len(data)>=2 and not np.isinf(data).any()
        values[modality]=data
        with h5py.File(source.statistics_path,'r') as h:
            stat_names=[v.decode() if isinstance(v,bytes) else str(v) for v in h['ch_names'][:]]
            assert stat_names==epochs.ch_names
            masks=np.asarray(h['mask'][:],dtype=bool)[[stat_names.index(c) for c in channels]][:,keep]
        for i,row in enumerate(sites.itertuples()):
            assert bool(masks[i].any())==bool(getattr(row,modality+'_responsive')),(subject,row.channel,'Baseline audit drift')
        trials.extend(dict(subject=subject,modality=modality,native_trial_index=i,event_name=name,retained_native_QC=True) for i,name in enumerate(names))
        native_sources.append(dict(subject=subject,modality=modality,epoch=source.epoch_path,epoch_sha256=sha(source.epoch_path),
            statistics=source.statistics_path,statistics_sha256=source.statistics_sha256,validation=source.validation,
            n_trials=len(data),n_time=len(times),repaired_source=bool(source.repaired_source)))
        del epochs
    tested=[]
    for i,row in enumerate(sites.itertuples()):
        print(subject,row.channel,'testing original PN method on 0–1 s, 2000 permutations',flush=True)
        sound,image=values['sound'][:,i],values['image'][:,i]
        mask,p=stats.time_perm_cluster(sound,image,**PARAMS)
        observed=(np.isfinite(sound).sum(axis=0)>=2)&(np.isfinite(image).sum(axis=0)>=2)
        assert not np.any(observed&~np.isfinite(p))
        difference=np.nanmean(sound,axis=0)-np.nanmean(image,axis=0)
        valid=observed&np.asarray(mask,dtype=bool)&(p<.05)
        tested.append(pd.DataFrame(dict(electrode=row.electrode,subject=subject,channel=row.channel,time=times,
            observed=observed,temporal_mask_two_sided=np.asarray(mask,dtype=bool),pointwise_p_two_sided=p,
            mean_sound=np.nanmean(sound,axis=0),mean_image=np.nanmean(image,axis=0),mean_difference=difference,
            sound_gt_image_mask=valid&(difference>0),image_gt_sound_mask=valid&(difference<0),
            n_sound=len(sound),n_image=len(image))))
    pd.concat(tested,ignore_index=True).to_csv(dest/'timeperm.csv.gz',index=False)
    pd.DataFrame(trials).to_csv(dest/'trial_selection.csv',index=False)
    write(dest/'sources.json',native_sources)
    protected()
    write(dest/'completed.json',dict(subject=subject,n_sites=len(sites),cohort_sha256=sha(OUT/'cohort.csv'),
        parameters=PARAMS,compatibility=compat,outputs={n:sha(dest/n) for n in ['timeperm.csv.gz','trial_selection.csv','sources.json']},
        job_id=os.environ['SLURM_JOB_ID'],completed_utc=datetime.now(timezone.utc).isoformat()))
    print(subject,'complete',flush=True)


def collect():
    protected()
    sites=pd.read_csv(OUT/'cohort.csv')
    tests=[];sources=[]
    for subject in sorted(sites.subject.unique()):
        dest=OUT/'subjects'/subject
        done=json.loads((dest/'completed.json').read_text())
        assert done['cohort_sha256']==sha(OUT/'cohort.csv') and done['parameters']==PARAMS
        assert all(sha(dest/n)==h for n,h in done['outputs'].items())
        tests.append(pd.read_csv(dest/'timeperm.csv.gz'))
        sources.extend(json.loads((dest/'sources.json').read_text()))
    tests=pd.concat(tests,ignore_index=True)
    assert len(tests)==128*129 and tests.electrode.nunique()==128 and tests.observed.all()
    assert not tests.duplicated(['electrode','time']).any()
    intervals=[];records=[]
    for electrode,group in tests.groupby('electrode',sort=False):
        group=group.sort_values('time');row=sites.loc[sites.electrode.eq(electrode)].iloc[0].to_dict()
        row['n_sound']=int(group.n_sound.iloc[0]);row['n_image']=int(group.n_image.iloc[0])
        for field,direction in [('sound_gt_image','Sound > Image'),('image_gt_sound','Image > Sound')]:
            spans=contiguous_intervals(group.time.to_numpy(),group[field+'_mask'].to_numpy(bool))
            row[field]=bool(spans);row[field+'_intervals']='; '.join(f"{s['start']:.4f}–{s['end']:.4f}" for s in spans)
            intervals.extend(dict(electrode=electrode,subject=row['subject'],channel=row['channel'],functional_cluster=row['functional_cluster'],direction=direction,interval=i+1,**s) for i,s in enumerate(spans))
        row['difference_class']=('Both directions' if row['sound_gt_image'] and row['image_gt_sound'] else 'Sound > Image' if row['sound_gt_image'] else 'Image > Sound' if row['image_gt_sound'] else 'No detected difference')
        records.append(row)
    table=pd.DataFrame(records)
    table.to_csv(OUT/'electrodes.csv',index=False);tests.to_csv(OUT/'timeperm.csv.gz',index=False)
    pd.DataFrame(intervals,columns=['electrode','subject','channel','functional_cluster','direction','interval','start','end','first_significant_time','last_significant_time','n_bins']).to_csv(OUT/'significant_intervals.csv',index=False)
    write(OUT/'native_sources.json',sources)
    counts=[]
    sets=[(n,table.loc[table.baseline_class.eq(n)]) for n in GROUPS]
    sets += [(n,table.loc[table[f]]) for n,f in [('Sound > Image','sound_gt_image'),('Image > Sound','image_gt_sound')]]
    for name,group in sets:
        counts.append(dict(set=name,n_electrodes=len(group),n_participants=group.subject.nunique(),participants=';'.join(sorted(group.subject.unique()))))
    counts=pd.DataFrame(counts);counts.to_csv(OUT/'set_counts.csv',index=False)
    composition=[]
    for name,group in [('All eligible',table)]+sets:
        for c in COMPONENTS:
            part=group.loc[group.functional_cluster.eq(c)];denominator=int(table.functional_cluster.eq(c).sum())
            composition.append(dict(set=name,component=c,n_electrodes=len(part),n_participants=part.subject.nunique(),component_denominator=denominator))
    pd.DataFrame(composition).to_csv(OUT/'component_composition.csv',index=False)
    for field in ['sound_gt_image','image_gt_sound']:
        table.loc[table[field]].to_csv(OUT/(field+'_electrodes.csv'),index=False)
    table.groupby('subject').agg(n_sites=('electrode','size'),sound_gt_image=('sound_gt_image','sum'),image_gt_sound=('image_gt_sound','sum')).reset_index().to_csv(OUT/'subject_counts.csv',index=False)
    table.groupby(['baseline_class','difference_class']).agg(n_electrodes=('electrode','size'),n_participants=('subject','nunique')).reset_index().to_csv(OUT/'response_difference_counts.csv',index=False)
    report=dict(n_sites=128,n_participants=27,window=[0,1],parameters=PARAMS,n_nmf=81,n_unassigned=47,
        baseline_sound=int(table.sound_responsive.sum()),baseline_image=int(table.image_responsive.sum()),
        counts=counts.to_dict('records'),n_both_directions=int((table.sound_gt_image&table.image_gt_sound).sum()),
        trial_rule='Original PN native-QC onset epochs; no artificial lexical accuracy gate',
        baseline_rule='Any retained native baseline mask hit in inclusive Stimulus [0,1] s',
        difference_rule='Original two-sided corrected temporal mask AND pointwise p<.05; sign of Sound minus Image',
        correction='Time within each electrode; no additional across-electrode correction',
        old_0_0_6_results_unchanged=True,pending_PN_expansion_excluded=True,
        protected_files_unchanged=protected(),job_id=os.environ['SLURM_JOB_ID'],completed_utc=datetime.now(timezone.utc).isoformat())
    write(OUT/'summary.json',report);print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID'),'Run through Slurm'
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['setup','worker','collect']);p.add_argument('--index',type=int)
    a=p.parse_args()
    if a.mode=='setup':setup()
    elif a.mode=='worker':worker(a.index if a.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID']))
    else:collect()
