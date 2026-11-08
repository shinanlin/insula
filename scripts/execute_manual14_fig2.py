"""Execute and validate Figure2 against the adopted224-site model version."""
from pathlib import Path
from datetime import datetime,timezone
import ast
import hashlib
import json
import os
import shutil
import sys
import time

import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency,fisher_exact

ROOT=Path('/cwork/ns458/insula-paper')
sys.path.insert(0,str(ROOT))
from src.main_analysis import FIG2_NMF_DIR as NMF,FIG2_COHORT_DIR as COHORT,NMF_WINDOWS
from scripts.adopt_insula_manual14_fig2 import RUN,VERSION,sha,write

NOTEBOOK=ROOT/'vizpub/fig2.ipynb'
OUT=ROOT/'results/fig2'


def main():
    assert os.environ.get('SLURM_JOB_ID'),'Run through Slurm'
    config=json.loads((ROOT/'config'/f'{VERSION}.json').read_text())
    excluded=set(config['excluded_sites'])
    model=json.loads((NMF/'nmf_manifest.json').read_text())
    data=np.load(NMF/'input_matrix.npz');factors=np.load(NMF/'factors.npz')
    assignments=pd.read_csv(NMF/'channel_assignments.csv')
    summary=json.loads((COHORT/'summary.json').read_text())
    candidates=pd.read_csv(COHORT/'nmf_candidate_sites_not_final_inputs.csv')
    assert model['k']==3 and model['rank_selection']['k']==2
    assert model['n_sites']==224 and model['n_subjects']==60
    assert assignments.channel.is_unique and len(assignments)==224 and assignments.subject.nunique()==60
    assert not(set(assignments.channel)&excluded)
    assert summary['sampled_sites']==378 and summary['sampled_subjects']==67
    assert summary['responsive_sites']==239 and summary['analysis_eligible_responsive_sites']==225
    response_columns=['site_id','confirmed_responsive','response_evidence_status']
    old_response=pd.read_csv(ROOT/'docs/evidence/insula_completion_2026-11-07/sampled_sites_unique.csv')
    new_response=pd.read_csv(COHORT/'sampled_sites_unique.csv')
    pd.testing.assert_frame_equal(
        old_response[response_columns].sort_values('site_id').reset_index(drop=True),
        new_response[response_columns].sort_values('site_id').reset_index(drop=True))
    assert set(assignments.channel)==set(candidates.loc[candidates.NMF_input_complete,'site_id'])
    assert np.array_equal(data['channels'],factors['channels'])
    assert assignments.channel.tolist()==data['channels'].tolist()
    assert data['X'].shape==(224,399) and np.isfinite(data['X']).all() and (data['X']>=0).all()
    assert np.allclose(np.linalg.norm(data['X'],axis=1),1) and np.allclose(np.linalg.norm(factors['H'],axis=1),1)
    assert np.allclose(np.linalg.norm(data['X']-factors['W']@factors['H']),model['reconstruction_error'])
    assert np.array_equal(assignments.functional_cluster,factors['component_names'][factors['W'].argmax(axis=1)])
    for i,name in enumerate(factors['component_names']):
        assert np.allclose(assignments[f'loading_{name}'],factors['W'][:,i])
    members=pd.read_csv(NMF/'waveform_source_membership.csv')
    assert members.modality.eq('sound').all() and set(members.channel)==set(assignments.channel)
    assert model['fit_windows']=={p:list(w) for p,w in NMF_WINDOWS.items()}
    display=json.loads((RUN/'display_inputs_completed.json').read_text())
    assert display['fitted_samples_match_original_selected_input'] and display['no_new_timeperm_tests']
    assert display['phase_selection_counts_reproduced'] and display['n_points_without_saved_mask']==0
    protected=json.loads((RUN/'protected_files.json').read_text())
    assert all(sha(p)==h for p,h in protected.items())
    nb=nbformat.read(NOTEBOOK,as_version=4);nbformat.validate(nb)
    for i,c in enumerate(nb.cells):
        if c.cell_type=='code':ast.parse(c.source,filename=f'fig2-cell-{i}')
    # Retain all the author's drawing code outside the scoped data/provenance
    # updates. The rank axis fixes a mislabeled metric; lateral tests use new data.
    before=nbformat.read(RUN/'before/vizpub/fig2.ipynb',as_version=4)
    allowed={'fig2-imports','fig2-style','fig2-load','54d7c386','fig2-rank-plot','fig2-laterality-plot','e4f7f44d'}
    assert [c.id for c in nb.cells]==[c.id for c in before.cells]
    for old,new in zip(before.cells,nb.cells):
        if old.cell_type=='code' and old.id not in allowed:
            assert ast.dump(ast.parse(old.source))==ast.dump(ast.parse(new.source)),old.id
    jupyter=RUN/'jupyter';kernel=jupyter/'kernels/insula-manual14-fig2';kernel.mkdir(parents=True,exist_ok=True)
    write(kernel/'kernel.json',dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        display_name='Insula manual14 Figure2 (ieeg)',language='python'))
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    os.environ['JUPYTER_PATH']=str(jupyter)+os.pathsep+os.environ.get('JUPYTER_PATH','')
    def started(cell,cell_index,**kwargs):
        print(datetime.now(timezone.utc).isoformat(),'cell',cell_index,cell.get('id'),flush=True)
    began=time.time()
    client=NotebookClient(nb,kernel_name='insula-manual14-fig2',timeout=1800,iopub_timeout=180,
        raise_on_iopub_timeout=True,resources={'metadata':{'path':str(ROOT/'vizpub')}},
        on_cell_start=started,record_timing=True)
    try:client.execute()
    except Exception:
        nbformat.write(nb,RUN/'fig2_failed.ipynb');raise
    errors=[(i,o.get('ename'),o.get('evalue')) for i,c in enumerate(nb.cells) for o in c.get('outputs',[]) if o.get('output_type')=='error']
    assert not errors and all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')
    assert not [i for i,c in enumerate(nb.cells) for o in c.get('outputs',[]) if 'image/png' in o.get('data',{})]
    manifest=json.loads((OUT/'figure_manifest.json').read_text())
    assert manifest['version']==VERSION and manifest['n_sites']==224 and manifest['n_subjects']==60
    assert manifest['rank_cophenetic_maximum']==2 and manifest['k']==3
    assert manifest['assignments_sha256']==sha(NMF/'channel_assignments.csv')
    old_names={p.name for p in (RUN/'before/img/fig2').glob('*.svg')}
    assert set(manifest['figures'])==old_names and len(old_names)==16
    for name,digest in manifest['figures'].items():
        p=ROOT/'img/fig2'/name
        assert sha(p)==digest and p.stat().st_mtime>=began,(name,'Panel not regenerated')
    masks=pd.read_csv(OUT/'site_significance_masks.csv.gz')
    heatmap=json.loads((OUT/'heatmap_plot_validation.json').read_text())
    assert heatmap['non_significant_points_blank']
    assert heatmap['order']=='first significant time separately per phase'
    assert heatmap['n_significant_points']==int(masks['mask'].sum())
    assert set(masks.channel)==set(assignments.channel)
    pooled=pd.read_csv(OUT/'pooled_waveform_summary.csv')
    curves=pd.read_csv(OUT/'cluster_waveform_summary.csv')
    assert pooled['count'].eq(224).all()
    counts=assignments.groupby('functional_cluster').channel.nunique()
    assert curves['count'].eq(curves.functional_cluster.map(counts)).all()
    sampling=pd.read_csv(OUT/'sampling_coordinates.csv')
    n_sampling=sampling.groupby('hemi').channel.nunique()
    n_clusters=assignments.groupby(['functional_cluster','hemi']).size().unstack(fill_value=0).reindex(columns=['L','R'],fill_value=0)
    lateral=json.loads((OUT/'laterality_statistics.json').read_text())
    assert lateral['sampling_sites']==378 and lateral['sampling_subjects']==67
    for group in n_clusters.index:
        n_l,n_r=int(n_clusters.loc[group,'L']),int(n_clusters.loc[group,'R'])
        expected=fisher_exact([[n_l,n_r],[int(n_sampling.L)-n_l,int(n_sampling.R)-n_r]],alternative='two-sided')[1]
        assert np.isclose(lateral['fisher_two_sided'][group],expected)
    assert np.isclose(lateral['chi_square_p'],chi2_contingency(n_clusters.to_numpy())[1])
    assert all(sha(p)==h for p,h in protected.items()),'An unrelated protected model/figure changed'
    nb.metadata['insula_figure2']=dict(version=VERSION,n_sites=224,n_subjects=60,k=3,
        rank_cophenetic_maximum=2,manual_exclusion_count=14,cohort_dir=str(COHORT),nmf_results_dir=str(NMF),
        condition='Repeat',modality='sound',fit_windows=model['fit_windows'],
        executed_utc=datetime.now(timezone.utc).isoformat(),svg_inline_only=True)
    temporary=NOTEBOOK.with_suffix('.executed.tmp.ipynb');nbformat.write(nb,temporary);temporary.replace(NOTEBOOK)
    registry=dict(version=VERSION,status='adopted_manual14_and_regenerated_figure2',
        cohort_dir=str(COHORT),nmf_results_dir=str(NMF),nmf_input_sites=224,nmf_input_subjects=60,
        sampled_sites=378,sampled_subjects=67,responsive_sites=239,analysis_eligible_responsive_sites=225,
        manual_exclusions=sorted(excluded),fitted_rank=3,rank_choice=config['rank_choice'],
        reassessed_rank=model['rank_selection'],selection_history=config['selection_history'],
        previous_completed_cohort=str(ROOT/'docs/evidence/insula_completion_2026-11-07'),
        previous_completed_NMF=str(ROOT/'results/insula_completion_2026-11-07/nmf'),
        previous_figure_NMF=str(ROOT/'results/nmf'),previous_snapshot=str(RUN/'before'),
        current_figures={'fig2':dict(version=VERSION,notebook=str(NOTEBOOK),n_sites=224,k=3),
                        'fig1_and_fig3_to_fig6':'preserved prior outputs/models; not refreshed for the224-site version'},
        old_downstream_frozen_guards='Historical scientific-input hashes remain frozen; they are not silently revalidated as the new Fig2 model.')
    write(ROOT/'config/current_insula_analysis.json',registry)
    evidence=COHORT/'figure2';evidence.mkdir(exist_ok=True)
    for p in OUT.iterdir():
        if p.is_file() and p.suffix in ('.csv','.gz','.json'):shutil.copy2(p,evidence/p.name)
    validation=dict(passed=True,version=VERSION,manual_exclusions=sorted(excluded),
        n_sites=224,n_subjects=60,n_features=399,k=3,cophenetic_maximum_k=2,
        all_code_cells_executed=True,code_cells=sum(c.cell_type=='code' for c in nb.cells),
        code_execution_errors=errors,svg_panels=16,svg_inline_only=True,all_panels_regenerated=True,
        exact_224_row_subset_verified=True,best_of_20_and_factor_labels_verified=True,
        display_context_from_same_saved_sources=True,raw_fitted_waveforms_unchanged=True,
        saved_point_masks_verified=True,heatmap_non_significant_points_blank=True,
        heatmap_phase_specific_order_preserved=True,group_means_and_SEM_use_electrodes=True,
        original_PC_variance_threshold_preserved=True,original_laterality_tests_recomputed=True,
        metric_axis_corrected_to_mean_ARI=True,original_drawing_cells_preserved=True,
        historical_NMF_and_other_figures_unchanged=True,protected_files=len(protected),
        notebook_sha256=sha(NOTEBOOK),slurm_job_id=os.environ['SLURM_JOB_ID'])
    write(OUT/'execution_validation.json',validation)
    write(RUN/'completed.json',validation)
    write(evidence/'execution_validation.json',validation)
    (RUN/'fig2_failed.ipynb').unlink(missing_ok=True)
    print(json.dumps(validation,indent=2),flush=True)


if __name__=='__main__':main()
