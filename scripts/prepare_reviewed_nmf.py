#!/usr/bin/env python3
"""Prepare main NMF from Figure 1 sound-selected sites and sound-only waveforms."""
from pathlib import Path
import hashlib,json,shutil,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.main_analysis import CONDITION,MODALITY,NMF_WINDOWS,RESPONSE_WINDOWS,COHORT_DIR
OUT=ROOT/'results/nmf'
FIG1=ROOT/'results/fig1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
 OUT.mkdir(parents=True,exist_ok=True)
 cohort=json.loads((COHORT_DIR/'summary.json').read_text())
 figure=json.loads((FIG1/'figure_manifest.json').read_text())
 assert cohort['modality']==figure['modality']==MODALITY=='sound'
 assert cohort['condition']==figure['condition']==CONDITION=='Repeat'
 candidates=pd.read_csv(COHORT_DIR/'nmf_candidate_sites_not_final_inputs.csv')
 ids=set(candidates.site_id)
 assert len(ids)==cohort['responsive_sites'] and candidates.subject.nunique()==cohort['responsive_subjects']
 membership=pd.read_csv(FIG1/'average_hga_source_membership.csv')
 assert membership.modality.eq(MODALITY).all()
 waves=pd.read_csv(FIG1/'average_hga_site_waveforms.csv.gz')
 assert set(waves.channel)==ids and set(waves.phase)==set(NMF_WINDOWS)
 assert waves.groupby('phase').channel.nunique().eq(len(ids)).all()
 # Verify exact agreement with the approved electrode-level Figure 1 trace.
 group=waves.groupby(['phase','time']).value.mean()
 expected=pd.read_csv(FIG1/'average_hga_group_trace.csv').set_index(['phase','time'])['mean']
 assert len(group)==len(expected) and np.allclose(group.sort_index(),expected.sort_index(),atol=1e-10)
 blocks=[];features=[]
 for phase,(lo,hi) in NMF_WINDOWS.items():
  rows=waves.loc[waves.phase.eq(phase)&waves.time.between(lo,hi)]
  matrix=rows.pivot(index='channel',columns='time',values='value').sort_index().sort_index(axis=1)
  assert set(matrix.index)==ids and np.isfinite(matrix.to_numpy()).all()
  blocks.append(matrix.to_numpy(float));features.extend(dict(phase=phase,time=float(t)) for t in matrix.columns)
 raw=np.concatenate(blocks,axis=1);positive=np.clip(raw,0,None);norms=np.linalg.norm(positive,axis=1)
 assert np.all(norms>np.finfo(float).eps)
 X=positive/norms[:,None];channels=matrix.index.to_numpy(str)
 metadata=pd.read_csv(FIG1/'insula_plot_coordinates.csv').set_index('channel').loc[channels].reset_index()
 assert metadata.channel.is_unique and metadata.modality.eq(MODALITY).all()
 metadata.to_csv(OUT/'input_sites.csv',index=False)
 pd.DataFrame(features).to_csv(OUT/'input_features.csv',index=False)
 np.savez_compressed(OUT/'input_matrix.npz',X=X,raw=raw,channels=channels,positive_l2_norm=norms)
 shutil.copy2(FIG1/'average_hga_site_waveforms.csv.gz',OUT/'site_waveforms.csv.gz')
 shutil.copy2(FIG1/'average_hga_source_membership.csv',OUT/'waveform_source_membership.csv')
 manifest=dict(n_sites=len(ids),n_subjects=int(metadata.subject.nunique()),n_features=X.shape[1],
  condition=CONDITION,modality=MODALITY,modalities=[MODALITY],phases=list(NMF_WINDOWS),
  fit_windows=NMF_WINDOWS,significance_windows=RESPONSE_WINDOWS,
  phase_samples=pd.DataFrame(features).groupby('phase').size().to_dict(),
  weighting='sound only; equal available task means within each site',
  preprocessing='negative HGA clipped at zero; L2 normalization across full concatenated waveform',
  selection_policy='sound-only anatomy sampling and sound-only Repeat significance; no image/text inclusion',
  excluded_subjects=['D0121'],old_manual_exclusions_applied=False,
  anatomy_selection='reviewed insular midpoint; endpoints insular GM, confirmed WM, or native Unknown retained and explicitly flagged by user policy',
  retained_unknown_endpoint_sampled_sites=cohort.get('retained_unknown_endpoint_sites',0),
  incomplete_response_evidence_sites=cohort['incomplete_response_sites'],
  missing_evidence_interpretation='excluded pending sound evidence; not negative',
  source_cohort=str(COHORT_DIR/'nmf_candidate_sites_not_final_inputs.csv'),
  source_cohort_sha256=sha(COHORT_DIR/'nmf_candidate_sites_not_final_inputs.csv'),
  figure1_cache_sha256=sha(FIG1/'available_evidence_stream.pkl'),figure1_notebook_sha256=sha(ROOT/'vizpub/fig1.ipynb'),
  figure1_group_mean_reproduced=True,sound_only_source_membership_verified=True,
  supplemented_epochs=json.loads((FIG1/'average_hga_supplemented_epochs.json').read_text()),
  outputs={p.name:sha(p) for p in [OUT/'input_matrix.npz',OUT/'input_sites.csv',OUT/'input_features.csv',OUT/'site_waveforms.csv.gz']})
 (OUT/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({k:v for k,v in manifest.items() if k not in ['supplemented_epochs','outputs']},indent=2),flush=True)


if __name__=='__main__':main()
