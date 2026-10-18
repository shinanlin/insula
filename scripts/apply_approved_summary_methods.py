#!/usr/bin/env python3
"""Apply and verify approved electrode summaries and the original PCA rule."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.main_analysis import COHORT_DIR,NMF_WINDOWS,RESPONSE_WINDOWS,PCA_VARIANCE_THRESHOLD
from src.nmf.reviewed_pca import update_pca_outputs
from src.nmf.pc_clustering import n_pcs_above_threshold

OUT=ROOT/'docs/evidence/approved_summary_methods_2026-10-17'
NMF=ROOT/'results/nmf'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 checkpoint=OUT/'before.json'
 if not checkpoint.exists():
  protected=[NMF/name for name in ['input_matrix.npz','factors.npz','channel_assignments.csv',
   'H_by_phase.csv','site_waveforms.csv.gz','rank_selection_metrics.csv','initialization_metrics.csv']]
  protected += [COHORT_DIR/name for name in ['sampled_sites_unique.csv','nmf_candidate_sites_not_final_inputs.csv']]
  model=json.loads((NMF/'nmf_manifest.json').read_text())
  previous=json.loads((NMF/'pc_clustering_meta.json').read_text())
  state=dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
   protected_sha256={str(p.relative_to(ROOT)):sha(p) for p in protected},
   unchanged_model_fields={key:model[key] for key in ['k','phases','fit_windows','significance_windows',
    'naming','assignment','cluster_counts','source_cohort_sha256']},previous_pca_embedding_pcs=previous['n_embedding_pcs'],
   previous_nmf_kmeans_ari=previous['nmf_kmeans_ari'])
  before=OUT/'before';before.mkdir()
  for path in [ROOT/'results/fig1/average_hga_group_trace.csv',ROOT/'results/fig2/cluster_waveform_summary.csv',
   ROOT/'results/fig2/pooled_waveform_summary.csv',NMF/'pc_clustering_meta.json',NMF/'pc_scores.csv']:
   shutil.copy2(path,before/path.name)
  checkpoint.write_text(json.dumps(state,indent=2)+'\n')
 metadata,_=update_pca_outputs(NMF)
 model=json.loads((NMF/'nmf_manifest.json').read_text())
 model.update(nmf_kmeans_ari=metadata['nmf_kmeans_ari'],pca_embedding_rule=metadata['embedding_rule'],
  pca_variance_threshold=metadata['variance_threshold'],pca_embedding_pcs=metadata['n_embedding_pcs'])
 (NMF/'nmf_manifest.json').write_text(json.dumps(model,indent=2)+'\n')
 print(json.dumps({key:metadata[key] for key in ['n_embedding_pcs','variance_threshold','embedding_rule','nmf_kmeans_ari']},indent=2),flush=True)


def summary(rows,keys):
 result=rows.groupby(keys).value.agg(['mean','std','count']).reset_index()
 result['sem']=result['std']/np.sqrt(result['count'])
 return result


def compare_summary(expected,path,keys):
 actual=pd.read_csv(path).set_index(keys).sort_index()
 expected=expected.set_index(keys).sort_index()
 pd.testing.assert_index_equal(actual.index,expected.index)
 for column in ['mean','std','count','sem']:
  assert np.allclose(actual[column],expected[column],equal_nan=True,atol=1e-12), (str(path),column)


def verify():
 before=json.loads((OUT/'before.json').read_text())
 assert all(sha(ROOT/path)==digest for path,digest in before['protected_sha256'].items())
 model=json.loads((NMF/'nmf_manifest.json').read_text())
 assert {key:model[key] for key in before['unchanged_model_fields']}==before['unchanged_model_fields']
 assert model['fit_windows']=={key:list(value) for key,value in NMF_WINDOWS.items()}
 assert model['significance_windows']=={key:list(value) for key,value in RESPONSE_WINDOWS.items()}
 waves=pd.read_csv(NMF/'site_waveforms.csv.gz')
 assert not waves.duplicated(['channel','phase','time']).any()
 mean=summary(waves,['phase','time'])
 assert mean['count'].eq(waves.channel.nunique()).all()
 compare_summary(mean,ROOT/'results/fig1/average_hga_group_trace.csv',['phase','time'])
 compare_summary(mean,ROOT/'results/fig2/pooled_waveform_summary.csv',['phase','time'])
 sites=pd.read_csv(NMF/'channel_assignments.csv')
 grouped=waves.merge(sites[['channel','functional_cluster']],on='channel',validate='many_to_one')
 compare_summary(summary(grouped,['functional_cluster','phase','time']),
  ROOT/'results/fig2/cluster_waveform_summary.csv',['functional_cluster','phase','time'])
 metadata=json.loads((NMF/'pc_clustering_meta.json').read_text())
 assert metadata['variance_threshold']==PCA_VARIANCE_THRESHOLD==.05
 assert metadata['n_embedding_pcs']==n_pcs_above_threshold(metadata['explained_variance_ratio'],threshold=PCA_VARIANCE_THRESHOLD)
 plotted=pd.read_csv(ROOT/'results/fig2/pca_kmeans_membership.csv').merge(
  sites[['channel','component']],on='channel',validate='one_to_one')
 assert len(plotted)==len(sites)
 assert np.isclose(adjusted_rand_score(plotted.component,plotted.kmeans_cluster),metadata['nmf_kmeans_ari'])
 for figure in ['fig1','fig2']:
  validation=json.loads((ROOT/f'results/{figure}/execution_validation.json').read_text())
  assert validation['passed']
 assert json.loads((ROOT/'results/fig2/heatmap_plot_validation.json').read_text())['non_significant_points_blank']
 result=dict(passed=True,group_average_unit='electrode',error_bar='SEM across electrodes',
  per_electrode_waveforms_unchanged=True,n_sites=int(waves.channel.nunique()),
  group_curve_mean_and_SEM_recomputed_and_verified=True,figure1_figure2_means_identical=True,
  previous_pca_embedding_pcs=before['previous_pca_embedding_pcs'],pca_embedding_pcs=metadata['n_embedding_pcs'],
  pca_variance_threshold=PCA_VARIANCE_THRESHOLD,nmf_kmeans_ari=metadata['nmf_kmeans_ari'],
  input_matrix_factors_assignments_and_rank_unchanged=True,nmf_refitted=False,
  approved_windows_unchanged=True,component_naming_and_assignment_rules_unchanged=True,
  future_deviations_require_user_review=True)
 (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--verify',action='store_true')
 args=parser.parse_args()
 verify() if args.verify else prepare()
