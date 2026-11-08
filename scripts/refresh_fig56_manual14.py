"""Refresh current Figure5/6 inputs without changing their numerical engines.

All entrypoints run under Slurm. Old RT/projection outputs and Figures1–4
remain protected. Models and cache versions are explicit in each notebook.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/cwork/ns458/insula-paper')
sys.path.insert(0, str(ROOT))
RUN = ROOT/'results/figure56_manual14_2026-11-07'
COHORT = ROOT/'docs/evidence/insula_manual14_2026-11-07'
NMF = ROOT/'results/insula_manual14_2026-11-07/nmf'
RT = RUN/'rt'
PROJ = RUN/'projection'
INPUT = RUN/'inputs'
METADATA = RUN/'rt_reviewed_contact_metadata.csv'
SAMPLE = RUN/'rt_eligible_anatomical_task_sites.csv'
TASKS = ('LexicalDelay','PhonemeSequence','PictureNaming')
PHASES = ('Delay','Go','Response')
DATASETS = {'LexicalDelay':'BIDS-1.0_LexicalDecRepDelay',
            'PhonemeSequence':'BIDS-1.4_Phoneme_sequencing',
            'PictureNaming':'BIDS-1.3_PictureNaming'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)


def strings(a):
    return [x.decode() if isinstance(x,bytes) else str(x) for x in a]


def configure_rt(metadata_path=None):
    os.environ['INSULA_RT_COHORT']=str(SAMPLE)
    os.environ['INSULA_RT_METADATA']=str(metadata_path or METADATA)


def root_for(task):return INPUT/task/'BIDS'


def old_model(task,subject,phase):
    from src.reaction_time.insula_rt_io import phase_output_path
    return phase_output_path(ROOT/'results/rt',task=task,subject=subject,phase=phase,description='Repeat')


def source_epochs(task,subject):
    roots=[RUN/'preprocessing'/task/subject/'BIDS',
           ROOT/'results/paper_trial_inputs'/task/'BIDS',
           ROOT/'results/insula_completion_2026-11-07/restored_native'/task/subject/'BIDS',
           ROOT/'results/native_preprocessing_restore_2026-10-18'/task/subject/'BIDS',
           Path('/cwork/ns458')/DATASETS[task]/'BIDS',
           ROOT/'results/picture_naming_restore_2026-10-24/BIDS']
    from src.reaction_time.insula_rt_data import find_phase_files,sibling_phase_path
    for bids in roots:
        paths={p:find_phase_files(bids,subject=subject,phase=p,description='Repeat',band='highgamma',ref='bipolar') for p in PHASES}
        if task=='PictureNaming':paths={p:[x for x in a if '_recording-sound_' in x.name] for p,a in paths.items()}
        if not all(paths.values()):continue
        if not all(sibling_phase_path(x,p,t).is_file() for p,a in paths.items() for x in a for t in ('Go','Response')):continue
        return bids,paths
    return None,{}


def snapshot():
    before=RUN/'before'
    if before.exists():return
    before.mkdir(parents=True)
    for rel in ('vizpub/fig5.ipynb','vizpub/fig6.ipynb','src/main_analysis.py',
                'config/current_insula_analysis.json','docs/MAIN_ANALYSIS.md'):
        dest=before/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    # This previously clean loader gets only an explicit metadata-provider branch.
    dest=before/'src/reaction_time/insula_rt_data.py';dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(subprocess.check_output(['git','show','HEAD:src/reaction_time/insula_rt_data.py'],cwd=ROOT,text=True))
    for rel in ('img/fig5','img/fig6','results/fig5','results/fig6'):
        shutil.copytree(ROOT/rel,before/rel)
    paths=[]
    for pattern in ('img/fig[1-4]/**/*','vizpub/fig[1-4].ipynb','results/nmf/**/*',
                    'results/insula_manual14_2026-11-07/nmf/**/*',
                    'results/insula_completion_2026-11-07/nmf/**/*',
                    'results/rt/**/*','results/diagnostics/decoding_exclusions_2026-11-07/**/*'):
        paths.extend(p for p in ROOT.glob(pattern) if p.is_file())
    write(RUN/'protected_files.json',{str(p):sha(p) for p in sorted(set(paths))})


def prepare_projection():
    import pandas as pd
    RUN.mkdir(parents=True,exist_ok=True)
    subjects=sorted(pd.read_csv(COHORT/'sampled_sites_by_subject_task.csv').subject.unique())
    assert len(subjects)==67
    data=dict(subjects=subjects,cohort_subjects=67,basis=str(NMF/'H_by_phase.csv'),
        basis_sha256=sha(NMF/'H_by_phase.csv'),assignments=str(NMF/'channel_assignments.csv'),
        assignment_sha256=sha(NMF/'channel_assignments.csv'),phases=['stimulus','delay','response'],
        tasks=['PhonemeSequence','LexicalDelay','PictureNaming','SentenceRep'],
        significance_mode='mask-any',min_coverage=.95,min_tasks=1,ridge=0.,
        bandwidth=8.,max_distance=15.,focused_min_subjects=3,focused_positive_fraction=.5,
        focused_quantile=.5,focused_max_clusters=5,focused_min_cluster_vertices=20)
    path=RUN/'figure6_spec.json'
    if path.exists():assert json.loads(path.read_text())==data
    else:write(path,data)
    write(RUN/'fig6_subjects.json',subjects)
    print('Frozen Figure6 scope:67 subjects,224-site discovery dictionary',flush=True)


def prepare():
    import h5py
    import mne
    import numpy as np
    import pandas as pd
    from src.reaction_time.insula_rt_data import load_phase_data
    RUN.mkdir(exist_ok=True,parents=True);snapshot()
    print('Snapshot and protected-file registry ready',flush=True)
    sample=pd.read_csv(COHORT/'sampled_sites_by_subject_task.csv')
    assignments=pd.read_csv(NMF/'channel_assignments.csv')
    exclusions=set(json.loads((ROOT/'config/insula_manual14_2026-11-07.json').read_text())['excluded_sites'])
    assert len(assignments)==224 and not set(assignments.channel)&exclusions
    subjects=sorted(sample.subject.unique());assert len(subjects)==67
    write(RUN/'fig6_subjects.json',subjects)
    sample=sample[sample.task.isin(TASKS)].copy()
    eligible=sample[~sample.native_QC_excluded & ~sample.subject_preprocessing_deferred].copy()
    coordinates=pd.read_csv(ROOT/'docs/evidence/insula_completion_2026-11-07/task_site_anatomy_audit.csv')
    metadata=[];matrix=[];sources=[];unavailable=[];input_hashes={}
    for (task,subject),group in sample.groupby(['task','subject'],sort=True):
        print('Inspecting native RT input',task,subject,flush=True)
        if group.subject_preprocessing_deferred.all():
            unavailable.append(dict(task=task,subject=subject,status='author_deferred_preprocessing'));continue
        accepted=set(eligible.loc[eligible.task.eq(task)&eligible.subject.eq(subject),'site_id'])
        if not accepted:
            unavailable.append(dict(task=task,subject=subject,status='native_QC_zero_usable'));continue
        bids,paths=source_epochs(task,subject)
        if bids is None:
            unavailable.append(dict(task=task,subject=subject,status='missing_native_RT_phase_epochs'));continue
        present_by_phase={phase:set(accepted) for phase in PHASES};bad=set()
        for phase,files in paths.items():
            for p in files:
                ep=mne.read_epochs(p,preload=False,verbose='ERROR')
                present_by_phase[phase] &= set(ep.ch_names);bad.update(ep.info['bads'])
                target=root_for(task)/'derivatives/epoch(bipolar)'/f'sub-{subject}'/'epoch(band)(zscore)'/p.name
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():target.symlink_to(p.resolve())
                source=str(p.resolve());input_hashes.setdefault(source,sha(p))
                sources.append(dict(task=task,subject=subject,phase=phase,path=str(target),native_path=source,
                                    sha256=input_hashes[source],n_trials=len(ep),n_channels=len(ep.ch_names)))
        # Keep original phase-specific availability: a missing channel in one
        # phase must not remove a usable channel from the other phases.
        accepted=set.union(*present_by_phase.values())-bad
        rejected=set(group.site_id)-accepted
        eligible=eligible[~(eligible.task.eq(task)&eligible.subject.eq(subject)&eligible.site_id.isin(rejected))]
        if not accepted:
            unavailable.append(dict(task=task,subject=subject,status='native_epoch_good_channel_intersection_empty'));continue
        old_meta=None
        for phase in PHASES:
            p=old_model(task,subject,phase)
            if p.exists():
                with h5py.File(p) as f:
                    old_meta=pd.DataFrame({k:strings(v[:]) if v.dtype.kind in 'OSU' else v[:] for k,v in f['channels'].items()})
                break
        old_names=[] if old_meta is None else old_meta.channel.tolist()
        order=[c for c in old_names if c in accepted]+sorted(accepted-set(old_names))
        for channel in order:
            if channel in old_names:
                row=old_meta[old_meta.channel.eq(channel)].iloc[0].to_dict()
                row['metadata_source']='prior_RT_saved_native_and_template_coordinates'
            else:
                original=coordinates[coordinates.task.eq(task)&coordinates.channel.eq(channel)].iloc[0]
                row=dict(channel=channel,roi=original.roi,hemi=original.hemi,label=original.label,
                         center='',mix=original['mix'],metadata_source='author_reviewed_current_anatomical_inventory')
                for axis in 'xyz':
                    row[axis+'_template']=original[axis]
                    row[axis+'_native']=original[axis+'_native']
                    row[axis+'_mni']=np.nan
            row.update(task=task,subject=subject)
            metadata.append(row)
        matrix.append(dict(index=len(matrix),task=task,subject=subject,bids_root=str(root_for(task)),
                           channels=order,source_bids=str(bids),old_channels=old_names))
    eligible.to_csv(SAMPLE,index=False)
    pd.DataFrame(metadata).to_csv(METADATA,index=False)
    pd.DataFrame(sources).to_csv(RUN/'native_RT_epoch_sources.csv',index=False)
    pd.DataFrame(unavailable).to_csv(RUN/'RT_unavailable_subject_tasks.csv',index=False)
    configure_rt()
    # Verify identities/trials using exactly the original RT loader. Signal
    # fitting is independent per electrode, but correction families stay whole.
    for row in matrix:
        print('Checking RT cache identities',row['index'],row['task'],row['subject'],flush=True)
        changes=[];reference=[]
        for phase in PHASES:
            data=load_phase_data(row['bids_root'],task=row['task'],subject=row['subject'],phase=phase)
            p=old_model(row['task'],row['subject'],phase)
            old_channels=[]
            if p.exists():
                with h5py.File(p) as f:
                    old_channels=strings(f['channels/channel'][:])
                    same=(old_channels==data.channel_meta.channel.tolist()
                          and strings(f['trials/trial_uid'][:])==data.trial_meta.trial_uid.tolist()
                          and np.array_equal(f['trials/rt_log'][:],data.trial_meta.rt_log.to_numpy()))
                    assert f.attrs['n_perm']==1000 and f.attrs['n_folds_requested']==10 and f.attrs['inner_folds_requested']==5
                    # Same resolved native source is necessary for cache reuse.
                    prior_sources={str(Path(v).resolve()) for v in strings(f['trials/source_file'][:])}
                    now_sources={str(Path(v).resolve()) for v in data.trial_meta.source_file}
                    same=same and prior_sources==now_sources
                reference.append(dict(phase=phase,path=str(p),sha256=sha(p)))
            else:same=False
            if not same:changes.append(phase)
            sources.append(dict(task=row['task'],subject=row['subject'],phase=phase,
                n_usable_channels=len(data.channel_meta),n_RT_trials=len(data.trial_meta),reusable_model=same))
            del data
        row.update(refit=bool(changes),changed_phases=changes,original_models=reference)
    write(RUN/'rt_run_plan.json',dict(models=matrix,refit_count=sum(r['refit'] for r in matrix),
        cohort_sha256=sha(SAMPLE),metadata_sha256=sha(METADATA),assignment_sha256=sha(NMF/'channel_assignments.csv'),
        no_HGA_significance_gate=True,correction_families=[['Delay','Go'],['Response']],
        original_methods=dict(n_perm=1000,n_folds=10,inner_folds=5,window_s=.2,step_s=.02,random_state=42),
        metadata_provider='Previously saved RT metadata plus current author-approved anatomical contacts; automatic labels retained, no invented Hammers parcellation.'))
    write(RUN/'native_input_hashes.json',input_hashes)
    prepare_projection()
    print(json.dumps(dict(RT_subject_task_groups=len(matrix),RT_refit_groups=sum(r['refit'] for r in matrix),
        RT_reused_groups=sum(not r['refit'] for r in matrix),fig6_subjects=len(subjects),unavailable=unavailable),indent=2),flush=True)


def rt_worker(index):
    from src.reaction_time.run_insula_rt_ridge import run_subject
    plan=json.loads((RUN/'rt_run_plan.json').read_text());row=plan['models'][index]
    metadata_path=Path(row.get('metadata_path',str(METADATA)))
    assert sha(SAMPLE)==plan['cohort_sha256'] and sha(metadata_path)==row.get('metadata_sha256',plan['metadata_sha256'])
    configure_rt(metadata_path)
    if row['refit']:
        paths=[]
        for phases in (('Delay','Go'),('Response',)):
            paths+=run_subject(bids_root=row['bids_root'],output_root=RT,task=row['task'],
                subject=row['subject'],phases=phases,description='Repeat',band='highgamma',ref='bipolar',
                atlas='hammers',window_s=.2,step_s=.02,n_folds=10,inner_folds=5,n_perm=1000,
                random_state=42,n_jobs=int(os.environ.get('SLURM_CPUS_PER_TASK','4')),
                raw_sfreq=2048.,rt_min_s=.05,overwrite=False)
        assert len(paths)==3
    else:
        from src.reaction_time.insula_rt_io import phase_output_path
        paths=[]
        for item in row['original_models']:
            assert sha(item['path'])==item['sha256']
            dest=phase_output_path(RT,task=row['task'],subject=row['subject'],phase=item['phase'],description='Repeat')
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():dest.symlink_to(item['path'])
            paths.append(dest)
    write(RUN/'rt_receipts'/f'{index:03d}.json',dict(index=index,task=row['task'],subject=row['subject'],
          refitted=row['refit'],outputs=[dict(path=str(p),sha256=sha(p)) for p in paths],job_id=os.environ['SLURM_JOB_ID']))


def projection_worker(index):
    from src.nmf.whole_brain_projection import run_whole_brain_projection
    spec=json.loads((RUN/'figure6_spec.json').read_text());subject=spec['subjects'][index]
    assert sha(spec['basis'])==spec['basis_sha256'] and sha(spec['assignments'])==spec['assignment_sha256']
    manifest=run_whole_brain_projection([subject],tasks=spec['tasks'],phases=spec['phases'],
        significance_mode=spec['significance_mode'],min_coverage=spec['min_coverage'],min_tasks=spec['min_tasks'],
        ridge=spec['ridge'],basis_path=Path(spec['basis']),assignments_path=Path(spec['assignments']),
        output_dir=PROJ/'subjects'/subject,make_plots=False)
    write(RUN/'projection_receipts'/f'{index:03d}.json',dict(subject=subject,manifest=manifest,job_id=os.environ['SLURM_JOB_ID']))


def collect_projection():
    import pandas as pd
    spec=json.loads((RUN/'figure6_spec.json').read_text())
    tables=[];details={};skipped={};inputs={}
    for i,subject in enumerate(spec['subjects']):
        receipt=json.loads((RUN/'projection_receipts'/f'{i:03d}.json').read_text());m=receipt['manifest']
        assert m['basis_sha256']==spec['basis_sha256'] and m['n_discovery_channels']==224
        if m['n_subjects_used']:
            table=pd.read_csv(m['combined_table']);assert set(table.subject)=={subject}
            tables.append(table);details.update(m['subjects_detail'])
            for path in m['subjects_detail'][subject]['input_paths']:inputs[path]=sha(path)
        else:skipped.update(m['subjects_skipped'])
    combined=pd.concat(tables,ignore_index=True)
    assert set(combined.subject)<=set(spec['subjects']) and not combined.channel.duplicated().any()
    PROJ.mkdir(parents=True,exist_ok=True);combined.to_csv(PROJ/'electrode_projection.csv',index=False)
    template=json.loads((RUN/'projection_receipts/000.json').read_text())['manifest']
    template.update(subjects=spec['subjects'],subjects_used=sorted(combined.subject.unique()),subjects_skipped=skipped,
        n_subjects_used=combined.subject.nunique(),n_subjects_skipped=len(skipped),subjects_detail=details,
        combined_table=str(PROJ/'electrode_projection.csv'),output_dir=str(PROJ),
        n_electrodes=len(combined),n_electrodes_discovery=int(combined.in_discovery.sum()))
    write(PROJ/'manifest.json',template);write(RUN/'projection_input_hashes.json',inputs)
    print('projection',len(combined),'sites',combined.subject.nunique(),'subjects',flush=True)


def surface():
    import pandas as pd
    import scripts.plot_nmf_whole_brain_projection as viz
    from src.nmf.whole_brain_projection_viz import prepare_projection_frame,subject_roi_scores,summarize_rois
    data=prepare_projection_frame(pd.read_csv(PROJ/'electrode_projection.csv'))
    out=PROJ/'visualization';out.mkdir(parents=True,exist_ok=True)
    figures=RUN/'projection_QC';figures.mkdir(exist_ok=True)
    data.to_csv(out/'heldout_projection_scores.csv',index=False)
    sr=subject_roi_scores(data);sr.to_csv(out/'subject_roi_scores.csv',index=False)
    summary=summarize_rois(sr,min_subjects=10,bootstrap_samples=2000);summary.to_csv(out/'roi_motif_summary.csv',index=False)
    viz.plot_roi_heatmap(summary,figures/'wholebrain_roi_motifs.svg')
    viz.plot_surface_maps(data,figures/'wholebrain_surface_motifs.svg',out/'surface_motif_maps.npz',
        bandwidth=8.,max_distance=15.,min_surface_subjects=5)
    viz.plot_focused_surface_maps(data,figures/'wholebrain_surface_motifs_focused.svg',out/'surface_motif_maps_focused.npz',
        bandwidth=8.,max_distance=15.,min_surface_subjects=3,min_positive_fraction=.5,
        surface_quantile=.5,max_clusters=5,min_cluster_vertices=20)


def summarize_rt():
    import h5py
    import pandas as pd
    from src.reaction_time.summarize_insula_rt_ridge import summarize_results
    from src.reaction_time.summarize_insula_rt_encoding import summarize_hga_rt_encoding
    plan=json.loads((RUN/'rt_run_plan.json').read_text());configure_rt(plan.get('complete_metadata_path'))
    configuration=[]
    for i in range(len(plan['models'])):
        r=json.loads((RUN/'rt_receipts'/f'{i:03d}.json').read_text())
        assert all(sha(x['path'])==x['sha256'] for x in r['outputs'])
        assert r['task']==plan['models'][i]['task'] and r['subject']==plan['models'][i]['subject']
        assert len(r['outputs'])==3
        phases=[]
        for item in r['outputs']:
            with h5py.File(item['path']) as f:
                expected=dict(n_perm=1000,n_folds_requested=10,inner_folds_requested=5,
                              window_requested_s=.2,step_requested_s=.02,random_state=42,
                              rt_min_s=.05,band='highgamma',ref='bipolar',description='Repeat')
                assert all(f.attrs[key]==value for key,value in expected.items()),item['path']
                assert f['scores/permutation_r'].shape[-1]==1000
                assert f.attrs['subject']==r['subject'] and f.attrs['task']==r['task']
                phases.append(f.attrs['phase'])
                configuration.append(dict(task=r['task'],subject=r['subject'],phase=f.attrs['phase'],
                    n_channels=len(f['channels/channel']),n_trials=len(f['trials/rt_log']),
                    n_windows=len(f['windows/center']),parameters_verified=True,
                    source_h5=item['path'],refitted=r['refitted']))
        assert set(phases)==set(PHASES)
    pd.DataFrame(configuration).to_csv(RUN/'RT_model_configuration_audit.csv',index=False)
    coverage,electrodes,clusters=summarize_results(RT,assignments_path=NMF/'channel_assignments.csv')
    labels=pd.read_csv(NMF/'channel_assignments.csv').set_index('channel').functional_cluster
    assert electrodes.functional_cluster.eq(electrodes.channel.map(labels)).all()
    assert set(electrodes.channel)<=set(labels.index)
    summarize_hga_rt_encoding(RT,assignments_path=NMF/'channel_assignments.csv',
        bids_roots={task:root_for(task) for task in TASKS})
    for phase in ('delay','go','response'):
        f=pd.read_csv(RT/'summaries'/f'hga_rt_{phase}_traces.csv')
        if f.empty:continue
        group=f.groupby(['subject','channel','functional_cluster','time'],as_index=False)[['hga_fast','hga_slow']].mean()
        long=group.melt(id_vars=['subject','channel','functional_cluster','time'],value_vars=['hga_fast','hga_slow'],var_name='rt_split',value_name='hga')
        group.to_csv(RUN/f'{phase}_electrode_traces.csv.gz',index=False)
        long.groupby(['functional_cluster','rt_split','time']).hga.agg(['mean','sem','count']).reset_index().to_csv(RUN/f'{phase}_electrode_mean_sem.csv',index=False)
    write(RUN/'rt_data_completed.json',dict(coverage_rows=len(coverage),modeled_assigned_sites=electrodes.channel.nunique(),
        significant_assigned_sites=clusters.channel.nunique(),model_subject_task_groups=len(plan['models']),
        new_assignment_sha256=sha(NMF/'channel_assignments.csv'),job_id=os.environ['SLURM_JOB_ID']))


def append_restored_rt():
    """Add three restored groups without changing the frozen first96 inputs."""
    import mne
    import numpy as np
    import pandas as pd
    from src.reaction_time.insula_rt_data import load_phase_data
    plan=json.loads((RUN/'rt_run_plan.json').read_text())
    assert sha(SAMPLE)==plan['cohort_sha256'] and sha(METADATA)==plan['metadata_sha256']
    if plan.get('restored_groups_appended'):return
    write(RUN/'rt_run_plan_before_trial_restore.json',plan)
    target_groups=[('LexicalDelay','D0086'),('PhonemeSequence','D0086'),('PictureNaming','D0139')]
    sample=pd.read_csv(SAMPLE);coords=pd.read_csv(ROOT/'docs/evidence/insula_completion_2026-11-07/task_site_anatomy_audit.csv')
    metadata=pd.read_csv(METADATA);added=[];native_hashes={};rows=[]
    for task,subject in target_groups:
        assert json.loads((RUN/'preprocessing'/task/subject/'completed.json').read_text())['subject']==subject
        assert json.loads((RUN/'preprocessing'/task/subject/'normalization_completed.json').read_text())['original_normalization_calls_verified']
        bids,paths=source_epochs(task,subject);assert bids is not None,(task,subject)
        accepted=set(sample.loc[sample.task.eq(task)&sample.subject.eq(subject),'site_id'])
        assert accepted
        for phase,files in paths.items():
            for p in files:
                ep=mne.read_epochs(p,preload=False,verbose='ERROR')
                assert not accepted & set(ep.info['bads'])
                assert accepted <= set(ep.ch_names),(task,subject,phase,accepted-set(ep.ch_names))
                target=root_for(task)/'derivatives/epoch(bipolar)'/f'sub-{subject}'/'epoch(band)(zscore)'/p.name
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():target.symlink_to(p.resolve())
                native_hashes[str(p.resolve())]=sha(p)
        for channel in sorted(accepted):
            original=coords[coords.task.eq(task)&coords.channel.eq(channel)].iloc[0]
            metadata_source='author_reviewed_current_anatomical_inventory'
            required=['x','y','z','x_native','y_native','z_native','mix']
            if original[required].isna().any():
                # The author guarantees same-contact identity across tasks.
                # Recover missing metadata only; do not re-audit consistency.
                available=coords[coords.channel.eq(channel)].dropna(subset=required)
                assert len(available),f'Missing reviewed metadata:{channel}'
                original=available.iloc[0]
                metadata_source=f'author_reviewed_same_contact_{original.task}_metadata_missing_in_{task}'
            row=dict(task=task,subject=subject,channel=channel,roi=original.roi,hemi=original.hemi,
                     label=original.label,center='',mix=original['mix'],
                     metadata_source=metadata_source)
            for axis in 'xyz':
                row[axis+'_template']=original[axis];row[axis+'_native']=original[axis+'_native'];row[axis+'_mni']=np.nan
            added.append(row)
        rows.append(dict(index=len(plan['models'])+len(rows),task=task,subject=subject,bids_root=str(root_for(task)),
            channels=sorted(accepted),source_bids=str(bids),old_channels=[],refit=True,changed_phases=list(PHASES),original_models=[]))
    combined=pd.concat([metadata,pd.DataFrame(added)],ignore_index=True)
    path=RUN/'rt_reviewed_contact_metadata_with_restored.csv';combined.to_csv(path,index=False)
    configure_rt(path)
    for row in rows:
        row.update(metadata_path=str(path),metadata_sha256=sha(path))
        for phase in PHASES:
            data=load_phase_data(row['bids_root'],task=row['task'],subject=row['subject'],phase=phase)
            assert len(data.trial_meta)>=10 and set(data.channel_meta.channel)==set(row['channels'])
    plan['models']+=rows;plan['refit_count']+=len(rows);plan['complete_metadata_path']=str(path)
    plan['restored_groups_appended']=True;plan['complete_metadata_sha256']=sha(path)
    write(RUN/'restored_native_input_hashes.json',native_hashes)
    write(RUN/'rt_run_plan.json',plan)
    unavailable=pd.read_csv(RUN/'RT_unavailable_subject_tasks.csv')
    unavailable.to_csv(RUN/'RT_unavailable_before_trial_restore.csv',index=False)
    unavailable=unavailable[~unavailable.apply(lambda r:(r.task,r.subject) in target_groups,axis=1)]
    unavailable.to_csv(RUN/'RT_unavailable_subject_tasks.csv',index=False)
    print('Appended restored RT model indices',[r['index'] for r in rows],flush=True)


if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID'),'Run via Slurm'
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','prepare_projection','rt','projection','collect_projection','surface','summarize_rt','append_restored_rt','execute','finalize'])
    p.add_argument('--index',type=int);p.add_argument('--figure',type=int,choices=[5,6]);a=p.parse_args()
    if a.mode in ('rt','projection'):
        (rt_worker if a.mode=='rt' else projection_worker)(a.index if a.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID']))
    elif a.mode in ('execute','finalize'):
        from src.figure56_current import execute,finalize
        if a.mode=='execute':execute(a.figure)
        else:finalize()
    else:{'prepare':prepare,'prepare_projection':prepare_projection,'collect_projection':collect_projection,'surface':surface,'summarize_rt':summarize_rt,'append_restored_rt':append_restored_rt}[a.mode]()
