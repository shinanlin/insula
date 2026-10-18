#!/usr/bin/env python3
"""Reuse fitted NMF for a redraw only after verifying its inputs are identical."""
from pathlib import Path
import hashlib,json,shutil,sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.main_analysis import COHORT_DIR


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
 nmf=ROOT/'results/nmf';fig1=ROOT/'results/fig1'
 model=json.loads((nmf/'nmf_manifest.json').read_text())
 assert model['condition']=='Repeat' and model['modality']=='sound'
 assert model['source_cohort_sha256']==sha(COHORT_DIR/'nmf_candidate_sites_not_final_inputs.csv')
 for current,saved,keys in [
  ('average_hga_site_waveforms.csv.gz','site_waveforms.csv.gz',['subject','channel','phase','time']),
  ('average_hga_source_membership.csv','waveform_source_membership.csv',['subject','channel','task','phase','modality']),
 ]:
  left=pd.read_csv(fig1/current).sort_values(keys).reset_index(drop=True)
  right=pd.read_csv(nmf/saved).sort_values(keys).reset_index(drop=True)
  pd.testing.assert_frame_equal(left,right,check_exact=True)
 protected={name:sha(nmf/name) for name in [
  'input_matrix.npz','channel_assignments.csv','factors.npz','rank_selection_metrics.csv']}
 old_notebook=model['figure1_notebook_sha256']
 new_notebook=sha(ROOT/'vizpub/fig1.ipynb')
 # These fields describe the current execution. Data, factors and labels stay
 # byte-identical; the prior execution identity remains in the redraw record.
 for name in ['input_manifest.json','nmf_manifest.json','rank_selection_meta.json']:
  path=nmf/name;metadata=json.loads(path.read_text())
  metadata['figure1_notebook_sha256']=new_notebook
  metadata['figure1_cache_sha256']=sha(fig1/'available_evidence_stream.pkl')
  metadata['figure1_group_average_weighting']='equal electrode means; SEM across electrodes'
  path.write_text(json.dumps(metadata,indent=2)+'\n')
 assert all(sha(nmf/name)==digest for name,digest in protected.items())
 record=dict(passed=True,identical_sound_site_waveforms=True,identical_source_membership=True,
  cohort_sha256=model['source_cohort_sha256'],previous_figure1_notebook_sha256=old_notebook,
  current_figure1_notebook_sha256=new_notebook,nmf_refitted=False,
  unchanged_nmf_data_factors_labels_and_rank=protected)
 output=ROOT/'results/fig2/redraw_input_verification.json'
 output.write_text(json.dumps(record,indent=2)+'\n')
 shutil.copy2(output,ROOT/'docs/evidence/figure2_sound_only_2026-10-17'/output.name)
 print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
