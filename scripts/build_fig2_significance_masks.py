#!/usr/bin/env python3
"""Recover saved sound Repeat pointwise masks for the original heatmap renderer."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json,re,sys
import h5py
import mne
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.main_analysis import COHORT_DIR,CONDITION,MODALITY,RESPONSE_WINDOWS
from audit_zero_endpoints import ROOTS

OUT=ROOT/'results/fig2'
SOURCE=Path('/hpc/group/coganlab/nanlinshi/insula-functional/results/hga')


def entity(path,key,default=''):
 match=re.search(r'(?:^|_)'+key+r'-([^_]+)',path.name)
 return match[1] if match else default


def main():
 OUT.mkdir(parents=True,exist_ok=True)
 members=pd.read_csv(ROOT/'results/nmf/waveform_source_membership.csv')
 assert members.modality.eq(MODALITY).all()
 allowed={(task,phase):set(rows.channel) for (task,phase),rows in members.groupby(['task','phase'])}
 frames=[];sources=[];grids={}
 evidence=pd.read_csv(COHORT_DIR/'response_evidence.csv')
 conflicts=set(map(tuple,evidence.loc[evidence.evidence_conflict,['task','channel','phase']].drop_duplicates().to_numpy()))
 paths=[]
 for task in members.task.unique():
  for path in sorted((SOURCE/task).glob('sub-*/HGA/*desc-Repeat_time.csv')):
   phase=entity(path,'proc').lower()
   if entity(path,'recording','sound')==MODALITY and (task,phase) in allowed:
    paths.append((task,phase,path))
 def read_csv(item):
  task,phase,path=item
  frame=pd.read_csv(path,usecols=['channel','time','mask','description','modality'])
  grid=np.sort(frame.time.unique())
  frame=frame.loc[frame.description.eq(CONDITION)&frame.modality.eq(MODALITY)&frame.channel.isin(allowed[(task,phase)])].copy()
  frame['mask']=frame['mask'].eq(True)
  frame['phase']=phase;frame['task']=task
  frame['source_priority']=0
  return item,grid,frame[['task','channel','phase','time','mask','source_priority']]
 with ThreadPoolExecutor(max_workers=4) as pool:
  for (task,phase,path),grid,frame in pool.map(read_csv,paths):
   grids[(task,path.name)]=grid
   if len(frame):
    frames.append(frame)
    sources.append(dict(task=task,path=str(path),kind='packaged_mask',bytes=path.stat().st_size))
 # Full-channel statistics include channels omitted from the old significant-
 # union CSVs. Read their saved masks; do not refit tests or infer significance.
 for task,subject in members[['task','subject']].drop_duplicates().itertuples(index=False,name=None):
  folder=ROOTS[task]/'derivatives/statistics'/f'sub-{subject}'/'bipolar'
  for path in sorted(folder.glob('*desc-Repeat_highgamma.h5')):
   phase=entity(path,'proc').lower()
   if (task,phase) not in allowed or entity(path,'recording','sound')!=MODALITY:continue
   with h5py.File(path,'r') as handle:
    mask=np.asarray(handle['mask'][:],bool)
    names=np.array([v.decode() if isinstance(v,bytes) else str(v) for v in handle['ch_names'][:]])
   selected=np.flatnonzero(np.isin(names,list(allowed[(task,phase)])))
   if not len(selected):continue
   csv_name=path.name.removesuffix('_highgamma.h5')+'_time.csv'
   grid=grids.get((task,csv_name))
   if grid is None or len(grid)!=mask.shape[1]:
    epoch=ROOTS[task]/'derivatives/epoch(bipolar)'/f'sub-{subject}'/'epoch(band)(zscore)'/path.name
    if not epoch.is_file():raise RuntimeError(f'Cannot verify statistics time grid: {path}')
    grid=mne.read_epochs(epoch,preload=False,verbose='error').times
   assert mask.shape==(len(names),len(grid)),path
   frame=pd.DataFrame(dict(channel=np.repeat(names[selected],len(grid)),time=np.tile(grid,len(selected)),mask=mask[selected].ravel()))
   frame['task']=task;frame['phase']=phase;frame['source_priority']=1
   frames.append(frame)
   sources.append(dict(task=task,path=str(path),kind='full_channel_statistics_mask',bytes=path.stat().st_size))
 raw=pd.concat(frames,ignore_index=True)
 raw=raw.loc[~pd.MultiIndex.from_frame(raw[['task','channel','phase']]).isin(conflicts)].copy()
 raw['time']=raw.time.round(9)
 # Preserve the original packaged pointwise mask wherever it exists. Native
 # full-channel statistics supply only entries omitted from those CSVs.
 raw=raw.sort_values('source_priority',kind='stable').drop_duplicates(['task','channel','phase','time'])
 # Any saved significant task at a site makes that time point visible, as in
 # the original groupby(mask='any'). Missing task evidence cannot add a True.
 grouped=raw.groupby(['channel','phase','time'],as_index=False)['mask'].any()
 waves=pd.read_csv(ROOT/'results/nmf/site_waveforms.csv.gz')
 waves['time']=waves.time.round(9)
 output=waves[['subject','channel','phase','time']].merge(grouped,on=['channel','phase','time'],how='left',validate='one_to_one')
 output['mask_available']=output['mask'].notna()
 output['mask']=output['mask'].fillna(False).astype(bool)
 output.to_csv(OUT/'site_significance_masks.csv.gz',index=False)
 coverage=output.groupby(['subject','channel','phase']).agg(n_time_samples=('time','size'),
  n_mask_available=('mask_available','sum'),n_significant=('mask','sum')).reset_index()
 coverage.to_csv(OUT/'site_significance_coverage.csv',index=False)
 expected=pd.read_csv(COHORT_DIR/'task_site_response_audit.csv')
 for phase,(lo,hi) in RESPONSE_WINDOWS.items():
  expected_ids=set(expected.loc[expected[phase+'_significant'],'channel'])&set(waves.channel)
  observed=set(output.loc[output.phase.eq(phase)&output.time.gt(lo)&output.time.lt(hi)&output['mask'],'channel'])
  assert observed==expected_ids,(phase,'Pointwise masks disagree with sound selection',sorted(observed^expected_ids))
 assert set(output.loc[output['mask'],'channel'])==set(waves.channel)
 report=dict(condition=CONDITION,modality=MODALITY,n_sites=int(waves.channel.nunique()),
  n_points=len(output),n_significant_points=int(output['mask'].sum()),n_points_without_saved_mask=int((~output.mask_available).sum()),
  missing_site_phase_pairs=int(coverage.n_mask_available.eq(0).sum()),
  existing_saved_masks_only=True,phase_selection_counts_reproduced=True,
  source_priority='original packaged masks first; full-channel statistics fill missing channel/time entries',
  aggregation='any across contributing sound Repeat tasks; unobserved evidence remains flagged and blank',
  sources=sources)
 (OUT/'site_significance_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k!='sources'},indent=2),flush=True)


if __name__=='__main__':main()
