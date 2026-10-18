#!/usr/bin/env python3
"""Build the authoritative sound-only sampling and responsive-site evidence."""
from pathlib import Path
import hashlib,json,shutil,sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.main_analysis import CONDITION,MODALITY,TASKS,RESPONSE_WINDOWS,COHORT_DIR
OLD=ROOT/'results/audit/four_task_selection_2026-10-17'
OUT=ROOT/'results/audit/sound_only_cohort_2026-10-17'


def main():
 OUT.mkdir(parents=True,exist_ok=True);COHORT_DIR.mkdir(parents=True,exist_ok=True)
 coords=pd.read_pickle(OLD/'all_coordinate_rows.pkl.gz')
 coords=coords.loc[coords.description.eq(CONDITION)&coords.modality.eq(MODALITY)].copy()
 assert len(coords) and not coords.modality.ne(MODALITY).any()
 coords.to_pickle(OUT/'all_coordinate_rows.pkl.gz')
 cols=[c for c in coords if c not in ['HGA','significant','phase','band','source_path']]
 variants=coords.drop_duplicates(subset=cols)
 variants.to_csv(OUT/'coordinate_variants.csv',index=False)
 source_ids=set(coords.source_path)
 manifest=[x for x in json.loads((OLD/'coordinate_manifest.json').read_text()) if x['path'] in source_ids]
 (OUT/'coordinate_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 pairs=coords[['task','channel']].drop_duplicates()
 anatomy=pd.read_csv(OLD/'task_site_anatomy_audit.csv').merge(pairs,on=['task','channel'],validate='one_to_one')
 anatomy.to_csv(OUT/'task_site_anatomy_audit.csv',index=False)
 global_anatomy=pd.read_csv(OLD/'site_anatomy_audit.csv')
 global_anatomy=global_anatomy.loc[global_anatomy.channel.isin(anatomy.channel)]
 global_anatomy.to_csv(OUT/'site_anatomy_audit.csv',index=False)
 evidence=pd.read_csv(OLD/'response_evidence.csv')
 evidence=evidence.loc[evidence.modality.eq(MODALITY)].copy()
 evidence.to_csv(OUT/'response_evidence.csv',index=False)
 groups={k:d for k,d in evidence.groupby(['task','channel'])}
 records=[]
 for r in anatomy.itertuples():
  d=groups.get((r.task,r.channel),evidence.iloc[:0])
  valid=d.loc[~d.evidence_conflict]
  row=dict(task=r.task,channel=r.channel,subject=r.subject,anatomy_status=r.status,modality=MODALITY)
  for phase in RESPONSE_WINDOWS:
   one=valid.loc[valid.phase.eq(phase)]
   row[phase+'_available']=bool(len(one))
   row[phase+'_significant']=bool(one.significant.any())
  complete=all(row[p+'_available'] for p in RESPONSE_WINDOWS)
  sig=any(row[p+'_significant'] for p in RESPONSE_WINDOWS)
  row.update(response_status='responsive' if sig else 'not_responsive' if complete else 'incomplete',
   all_three_phases_available=complete,any_significant=sig,
   old_window_any_significant=bool(valid.old_window_significant.any()),modalities=MODALITY,
   missing_phase_modality_cells='|'.join(p+'/sound' for p in RESPONSE_WINDOWS if not row[p+'_available']),
   any_mask_conflict=bool(d.evidence_conflict.any()))
  records.append(row)
 response=pd.DataFrame(records);response.to_csv(OUT/'task_site_response_audit.csv',index=False)
 included=set(global_anatomy.loc[global_anatomy.status.eq('include'),'channel'])
 eligible=response.loc[response.channel.isin(included)]
 positive=set(eligible.loc[eligible.any_significant,'channel'])
 pending=set(eligible.loc[~eligible.all_three_phases_available,'channel'])-positive
 negative=included-positive-pending
 summary=dict(condition=CONDITION,modality=MODALITY,sampled_sites=len(included),
  sampled_subjects=int(global_anatomy.loc[global_anatomy.channel.isin(included),'subject'].nunique()),
  responsive_sites=len(positive),responsive_subjects=int(global_anatomy.loc[global_anatomy.channel.isin(positive),'subject'].nunique()),
  incomplete_response_sites=len(pending),confirmed_nonresponsive_sites=len(negative),
  phase_counts={p:int(eligible.loc[eligible[p+'_significant'],'channel'].nunique()) for p in RESPONSE_WINDOWS},
  selection_windows=RESPONSE_WINDOWS,coordinate_rows=int(len(coords)),coordinate_sources=len(manifest))
 old=pd.read_csv(ROOT/'docs/evidence/subject_task_site_lists_2026-10-17/nmf_candidate_sites_not_final_inputs.csv')
 removed=old.loc[~old.site_id.isin(positive)].copy()
 removed['reason']='no_confirmed_sound_Repeat_response_in_main_windows'
 removed.to_csv(OUT/'removed_from_multimodal_responsive_list.csv',index=False)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 for name in ['task_site_response_audit.csv','site_anatomy_audit.csv','response_evidence.csv','coordinate_manifest.json','summary.json','removed_from_multimodal_responsive_list.csv']:
  shutil.copy2(OUT/name,COHORT_DIR/name)
 print(json.dumps(summary,indent=2),flush=True)
 # Reuse the subject/task exporter with sound-only evidence and dynamic counts.
 import build_subject_task_site_lists as exporter
 exporter.AUDIT=OUT;exporter.OUT=OUT;exporter.EVIDENCE=COHORT_DIR
 exporter.main()


if __name__=='__main__':main()
