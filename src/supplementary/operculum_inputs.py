"""Prepare reviewable anatomical/signal candidates without fitting a new control.

Hammers IFG is only a broad candidate region. It is never silently relabeled
as frontal operculum. Exact opercular labels and endpoint anatomy require
review before these candidates enter an orthogonalization analysis.
"""
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = {'PhonemeSequence':'BIDS-1.4_Phoneme_sequencing',
            'LexicalDelay':'BIDS-1.0_LexicalDecRepDelay',
            'PictureNaming':'BIDS-1.3_PictureNaming','SentenceRep':'BIDS-1.4_SentenceRep'}


def audit_inputs(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True,exist_ok=True)
    sample = pd.read_csv(root/'docs/evidence/sound_only_cohort_2026-10-17/sampled_sites_by_subject_task.csv')
    coords = pd.read_csv(root/'results/audit/sound_only_cohort_2026-10-17/coordinate_variants.csv').fillna('')
    coords = coords.loc[coords.description.eq('Repeat') & coords.modality.eq('sound')].copy()
    cols = ['task','subject','channel','label','roi','hemi','x_native','y_native','z_native',
            'contact_1','contact_2','contact_1_label','contact_2_label']
    coords = coords[cols].drop_duplicates()
    for column in ['x_native','y_native','z_native']:
        coords[column]=pd.to_numeric(coords[column],errors='coerce')
    variants = coords.groupby(['task','channel']).size()
    conflicts = set(variants.loc[variants.gt(1)].index)
    coords['coordinate_or_label_variants'] = [(r.task,r.channel) in conflicts for r in coords.itertuples()]
    coords = coords.drop_duplicates(['task','channel'])
    legacy_root=Path('/hpc/group/coganlab/nanlinshi/insula-analysis-legacy/results')
    legacy_rows=[]
    for (task,subject),group in sample.groupby(['task','subject']):
        folder=legacy_root/f'{task}(bipolar)'/f'sub-{subject}'/'HGA'
        for path in sorted(folder.glob('*desc-Repeat_coord.csv')):
            frame=pd.read_csv(path,usecols=lambda c:c in ['channel','label','x','y','z']).fillna('')
            if not {'channel','label'}.issubset(frame.columns): continue
            for row in frame.loc[frame.label.str.contains('G_front_inf-Opercular',case=False,na=False)].itertuples():
                legacy_rows.append(dict(task=task,subject=subject,channel=row.channel,
                                        legacy_Destrieux_label=row.label,label_source=str(path)))
    legacy=pd.DataFrame(legacy_rows,columns=['task','subject','channel','legacy_Destrieux_label','label_source']).drop_duplicates()
    legacy.to_csv(output/'legacy_opercular_label_evidence.csv',index=False)
    label_map=legacy.drop_duplicates(['task','channel'])[['task','channel','legacy_Destrieux_label']]
    coords=coords.merge(label_map,on=['task','channel'],how='left')
    coords['legacy_Destrieux_label']=coords.legacy_Destrieux_label.fillna('')
    counts = coords.groupby(['task','roi','label']).size().rename('n_sites').reset_index()
    counts.to_csv(output/'anatomical_label_inventory.csv',index=False)
    explicit = (coords.label.str.contains('opercul|parsoperc',case=False,regex=True)
                | coords.legacy_Destrieux_label.str.contains('G_front_inf-Opercular',case=False,regex=False))
    broad = coords.roi.eq('IFG') | coords.label.str.contains('inferior frontal',case=False,regex=False)
    oper = coords.loc[explicit | broad].copy()
    oper['candidate_label_status'] = np.where(explicit.loc[oper.index],
        'explicit_opercular_label_pending_endpoint_review','broad_IFG_only_not_confirmed_operculum')
    oper.to_csv(output/'opercular_region_candidates.csv',index=False)
    ins = sample[['task','subject','site_id']].merge(
        coords,left_on=['task','subject','site_id'],right_on=['task','subject','channel'],how='inner')
    ins = ins.drop_duplicates(['task','channel'])
    inventory, pairs = [], []
    for (task,subject), subjects_sites in sample.groupby(['task','subject']):
        canonical = Path('/cwork/ns458')/DATASETS[task]/'BIDS'
        staged = root/'results/native_preprocessing_restore_2026-10-18'/task/subject/'BIDS'
        staged_files = list((staged/'derivatives/bipolar'/f'sub-{subject}'/'ieeg').glob('*ieeg.edf'))
        bids = staged if staged_files else canonical
        folder = bids/'derivatives/bipolar'/f'sub-{subject}'/'ieeg'
        edfs = sorted(folder.glob('*ieeg.edf'))
        tables = sorted(folder.glob('*channels.tsv'))
        native_sets=[];bads=set()
        for table in tables:
            frame=pd.read_csv(table,sep='\t').fillna('')
            native_sets.append(set(frame.loc[~frame.status.eq('bad'),'name']))
            bads.update(frame.loc[frame.status.eq('bad'),'name'])
        valid=set.intersection(*native_sets) if native_sets else set()
        candidates = oper.loc[oper.task.eq(task)&oper.subject.eq(subject)]
        insula = ins.loc[ins.task.eq(task)&ins.subject.eq(subject)]
        inv=dict(task=task,subject=subject,n_sampled_insula=len(subjects_sites),
                 n_coordinate_insula=len(insula),n_broad_IFG_candidates=len(candidates),
                 n_explicit_opercular_candidates=int(candidates.candidate_label_status.str.startswith('explicit').sum()),
                 n_bipolar_EDFs=len(edfs),n_channel_tables=len(tables),
                 bipolar_input_root=str(bids),n_native_good_channels_all_runs=len(valid),
                 native_quality_flags_retained=True)
        inventory.append(inv)
        for a in insula.itertuples():
            aa=np.array([a.x_native,a.y_native,a.z_native],dtype=float)
            if not np.isfinite(aa).all(): continue
            potential=[]
            named=candidates.loc[candidates.candidate_label_status.str.startswith('explicit')]
            search=named if len(named) else candidates
            for b in search.itertuples():
                bb=np.array([b.x_native,b.y_native,b.z_native],dtype=float)
                if not np.isfinite(bb).all() or a.hemi!=b.hemi: continue
                shared={a.contact_1,a.contact_2}&{b.contact_1,b.contact_2}
                if shared: continue
                potential.append((float(np.linalg.norm(aa-bb)),b))
            if not potential: continue
            distance,b=min(potential,key=lambda item:item[0])
            pairs.append(dict(task=task,subject=subject,insula_site=a.channel,
                              insula_hammers_label=a.label,insula_display_region=a.roi,
                              candidate_site=b.channel,candidate_label=b.label,
                              candidate_Destrieux_label=b.legacy_Destrieux_label,
                              candidate_label_status=b.candidate_label_status,
                              hemisphere=a.hemi,native_distance_mm=distance,
                              shared_contacts=False,
                              coordinate_or_label_conflict=bool(a.coordinate_or_label_variants or b.coordinate_or_label_variants),
                              both_channels_good_in_all_available_runs=a.channel in valid and b.channel in valid,
                              n_signal_runs=len(edfs),bipolar_input_root=str(bids),
                              final_pair_approved=False))
    inv=pd.DataFrame(inventory)
    pair_columns=['task','subject','insula_site','insula_hammers_label','insula_display_region',
                  'candidate_site','candidate_label','candidate_Destrieux_label','candidate_label_status','hemisphere',
                  'native_distance_mm','shared_contacts','coordinate_or_label_conflict',
                  'both_channels_good_in_all_available_runs','n_signal_runs','bipolar_input_root','final_pair_approved']
    paired=pd.DataFrame(pairs,columns=pair_columns)
    inv.to_csv(output/'subject_task_signal_inventory.csv',index=False)
    paired.to_csv(output/'nearest_opercular_or_IFG_candidates.csv',index=False)
    summary=dict(scope='Input and anatomy audit only; no new orthogonalization, xcorr, or task-response fit',
                 subject_task_pairs=len(inv),n_broad_region_candidates=len(oper),
                 n_explicit_opercular_labels=int(explicit.sum()),
                 legacy_opercular_evidence_rows=len(legacy),
                 archived_Destrieux_labels_require_native_identity_and_endpoint_review=True,
                 n_nearest_nonshared_candidates=len(paired),
                 n_candidates_with_both_native_good_channels=int(paired.both_channels_good_in_all_available_runs.sum()) if len(paired) else 0,
                 exact_opercular_boundary_requires_review=True,
                 Hammers_IFG_not_equated_to_operculum=True,
                 historical_opercular_notebook='/hpc/group/coganlab/nanlinshi/insula-aicl-delay/notebooks/opercular.ipynb',
                 historical_method='Per-trial HGA linear projection residual, without intercept',
                 historical_example_shares_contact=True,
                 analytical_signal_method_pending_review=False,
                 analytical_signal_method_user_approved='Subband analytic-signal orthogonalization before HGA; no shared contacts',
                 new_orthogonalization_results_generated=False,
                 original_inputs_or_main_figures_modified=False)
    (output/'input_audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return dict(label_counts=counts.loc[counts.roi.eq('IFG')|counts.label.str.contains('opercul',case=False)],
                subject_task_inventory=inv,nearest_candidates=paired,summary=summary)
