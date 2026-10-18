#!/usr/bin/env python3
"""Execute Figure 2 and verify the sound-only main-analysis policy end to end."""
from datetime import datetime,timezone
import ast,hashlib,json,os,shutil,sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.paths import nmf_chosen_k
from src.main_analysis import CONDITION,MODALITY,COHORT_DIR,NMF_WINDOWS
SOURCE=Path('/hpc/group/coganlab/nanlinshi/insula-functional')
OUT=ROOT/'results/fig2';NMF=ROOT/'results/nmf';NOTEBOOK=ROOT/'vizpub/fig2.ipynb'
EVIDENCE=ROOT/'docs/evidence/figure2_sound_only_2026-10-17'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
 OUT.mkdir(parents=True,exist_ok=True)
 cohort=json.loads((COHORT_DIR/'summary.json').read_text());n_sites=cohort['responsive_sites'];n_subjects=cohort['responsive_subjects']
 protected=[ROOT/'vizpub/fig1.ipynb',SOURCE/'vizpub/fig2.ipynb']+list((ROOT/'img/fig1').glob('*.svg'))
 protected += [p for p in (SOURCE/'results/nmf').glob('*') if p.is_file()]
 for name in ['figure2_refresh_2026-10-17','figure2_response_window_2026-10-17']:
  protected += [p for p in (ROOT/'docs/evidence'/name).rglob('*') if p.is_file()]
 before={str(p):sha(p) for p in protected}
 model=json.loads((NMF/'nmf_manifest.json').read_text())
 assert model['figure1_notebook_sha256']==sha(ROOT/'vizpub/fig1.ipynb')
 assert nmf_chosen_k()==model['k']==3
 assert model['condition']==CONDITION=='Repeat' and model['modality']==MODALITY=='sound' and model['modalities']==['sound']
 assert model['fit_windows']=={p:list(w) for p,w in NMF_WINDOWS.items()}
 factors=np.load(NMF/'factors.npz');inputs=np.load(NMF/'input_matrix.npz')
 sites=pd.read_csv(NMF/'channel_assignments.csv');expected=pd.read_csv(COHORT_DIR/'nmf_candidate_sites_not_final_inputs.csv')
 assert len(sites)==n_sites and sites.subject.nunique()==n_subjects and sites.channel.is_unique
 assert set(sites.channel)==set(expected.site_id) and sites.modality.eq('sound').all()
 assert pd.read_csv(NMF/'waveform_source_membership.csv').modality.eq('sound').all()
 assert np.array_equal(factors['channels'],inputs['channels']) and sites.channel.tolist()==inputs['channels'].tolist()
 features=pd.read_csv(NMF/'input_features.csv')
 assert inputs['X'].shape==(n_sites,len(features))
 assert features.groupby('phase').size().to_dict()=={'stimulus':77,'delay':129,'response':193}
 response_times=features.loc[features.phase.eq('response'),'time']
 assert np.isclose(response_times.min(),-.5) and np.isclose(response_times.max(),1.)
 assert np.allclose(np.linalg.norm(inputs['X'],axis=1),1) and np.isfinite(inputs['X']).all() and (inputs['X']>=0).all()
 assert np.allclose(np.linalg.norm(factors['H'],axis=1),1)
 assert np.array_equal(sites.functional_cluster.to_numpy(),factors['component_names'][factors['W'].argmax(axis=1)])
 assert np.allclose(np.linalg.norm(inputs['X']-factors['W']@factors['H']),model['reconstruction_error'])
 starts=pd.read_csv(NMF/'initialization_metrics.csv')
 assert len(starts)==20 and starts.converged.all() and np.isclose(starts.reconstruction_error.min(),model['reconstruction_error'])
 rank_meta=json.loads((NMF/'rank_selection_meta.json').read_text());rank=pd.read_csv(NMF/'rank_selection_metrics.csv')
 assert rank_meta['input_matrix_sha256']==sha(NMF/'input_matrix.npz') and rank_meta['modality']=='sound'
 assert set(rank.k)==set(range(2,7)) and rank.n_boot.eq(200).all() and rank.row_frac.eq(.8).all()
 nb=nbformat.read(NOTEBOOK,as_version=4)
 for i,c in enumerate(nb.cells):
  if c.cell_type=='code':ast.parse(c.source,filename=f'fig2-cell-{i}')
 nbformat.validate(nb)
 os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
 jupyter=OUT/'jupyter';spec=jupyter/'kernels/insula-paper-fig2';spec.mkdir(parents=True,exist_ok=True)
 (spec/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],display_name='Insula Figure 2 (ieeg)',language='python')))
 os.environ['JUPYTER_PATH']=str(jupyter)+os.pathsep+os.environ.get('JUPYTER_PATH','')
 def started(cell,cell_index,**kwargs):print(f'{datetime.now(timezone.utc).isoformat()} cell {cell_index}: {cell.source.splitlines()[0][:100] if cell.source else "empty"}',flush=True)
 client=NotebookClient(nb,kernel_name='insula-paper-fig2',timeout=1800,iopub_timeout=180,raise_on_iopub_timeout=True,
  resources={'metadata':{'path':str(ROOT/'vizpub')}},on_cell_start=started,record_timing=True)
 try:client.execute()
 except Exception:
  nbformat.write(nb,OUT/'fig2_failed.ipynb');raise
 assert all(sha(Path(p))==h for p,h in before.items())
 errors=[(i,o.get('ename'),o.get('evalue')) for i,c in enumerate(nb.cells) for o in c.get('outputs',[]) if o.get('output_type')=='error']
 assert not errors and all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')
 assert not [i for i,c in enumerate(nb.cells) for o in c.get('outputs',[]) if 'image/png' in o.get('data',{})]
 manifest=json.loads((OUT/'figure_manifest.json').read_text())
 assert manifest['n_sites']==n_sites and manifest['n_subjects']==n_subjects and manifest['assignments_sha256']==sha(NMF/'channel_assignments.csv')
 assert manifest['modality']=='sound' and manifest['condition']=='Repeat' and manifest['fit_windows']==model['fit_windows']
 assert len(manifest['figures'])>=15
 nb.metadata['insula_figure2']=dict(cohort_version='sound_only_2026-10-17',condition=CONDITION,modality=MODALITY,n_sites=n_sites,n_subjects=n_subjects,k=3,
  phases=list(NMF_WINDOWS),fit_windows=model['fit_windows'],executed_utc=datetime.now(timezone.utc).isoformat(),svg_inline_only=True)
 temporary=NOTEBOOK.with_suffix('.executed.tmp.ipynb');nbformat.write(nb,temporary);temporary.replace(NOTEBOOK)
 EVIDENCE.mkdir(parents=True,exist_ok=True)
 for name in ['input_manifest.json','input_matrix.npz','input_sites.csv','input_features.csv','site_waveforms.csv.gz','waveform_source_membership.csv',
  'channel_assignments.csv','factors.npz','H_by_phase.csv','component_phase_profiles.csv','initialization_metrics.csv','nmf_manifest.json','rank_selection_metrics.csv',
  'rank_selection_meta.json','chosen_k.json','pc_scores.csv','pc_clustering_meta.json','comparison_with_old_four_phase.csv','comparison_with_previous_multimodal.csv']:
  shutil.copy2(NMF/name,EVIDENCE/name)
 for path in OUT.glob('*.csv*'):shutil.copy2(path,EVIDENCE/path.name)
 shutil.copy2(OUT/'figure_manifest.json',EVIDENCE/'figure_manifest.json')
 for folder,source in [('figures',ROOT/'img/fig2'),('rank_figures',ROOT/'img/nmf')]:
  destination=EVIDENCE/folder;destination.mkdir(exist_ok=True)
  for path in source.glob('*.svg'):shutil.copy2(path,destination/path.name)
 validation=dict(passed=True,condition=CONDITION,modality=MODALITY,sound_only_selection_and_input_verified=True,
  code_cells=sum(c.cell_type=='code' for c in nb.cells),all_code_cells_executed=True,output_errors=errors,svg_panels=len(manifest['figures']),svg_inline_only=True,
  n_sites=n_sites,n_subjects=n_subjects,k=3,n_features=len(features),fit_windows=model['fit_windows'],H_components_unit_norm=True,
  assignments_match_normalized_W_argmax=True,best_of_20_verified=True,rank_same_sound_input_verified=True,
  fitted_model_rank_used_by_downstream=True,figure1_and_legacy_results_unchanged=True,previous_multimodal_archives_unchanged=True,
  notebook_sha256=sha(NOTEBOOK),protected_file_hashes=before)
 (OUT/'execution_validation.json').write_text(json.dumps(validation,indent=2)+'\n');shutil.copy2(OUT/'execution_validation.json',EVIDENCE/'execution_validation.json')
 (OUT/'fig2_failed.ipynb').unlink(missing_ok=True)
 print(json.dumps({k:v for k,v in validation.items() if k!='protected_file_hashes'},indent=2),flush=True)


if __name__=='__main__':main()
