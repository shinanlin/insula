#!/usr/bin/env python3
"""Execute Figure 2, validate its common cohort/model, and archive reproducible evidence."""
from datetime import datetime, timezone
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.paths import nmf_chosen_k
SOURCE=Path('/hpc/group/coganlab/nanlinshi/insula-functional')
OUT=ROOT/'results/fig2'
NMF=ROOT/'results/nmf'
NOTEBOOK=ROOT/'vizpub/fig2.ipynb'
EVIDENCE=ROOT/'docs/evidence/figure2_response_window_2026-10-17'
PREVIOUS=ROOT/'docs/evidence/figure2_refresh_2026-10-17'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protected = [ROOT/'vizpub/fig1.ipynb', SOURCE/'vizpub/fig2.ipynb']
    protected += list((ROOT/'img/fig1').glob('*.svg'))
    protected += [p for p in (SOURCE/'results/nmf').glob('*') if p.is_file()]
    protected += [p for p in PREVIOUS.rglob('*') if p.is_file()]
    before = {str(p):sha(p) for p in protected}
    model=json.loads((NMF/'nmf_manifest.json').read_text())
    assert model['figure1_notebook_sha256']==sha(ROOT/'vizpub/fig1.ipynb')
    assert nmf_chosen_k()==model['k']==3, 'Downstream rank must match fitted factors, not rank-selection maximum'
    assert model['fit_windows']=={'stimulus':[0.,.6], 'delay':[0.,1.], 'response':[-.5,1.]}
    comparison=json.loads((NMF/'response_window_comparison.json').read_text())
    assert comparison['common_raw_features_unchanged'] and comparison['site_waveforms_unchanged']
    assert comparison['cohort_and_site_order_unchanged'] and comparison['added_response_samples']==64
    factors=np.load(NMF/'factors.npz')
    inputs=np.load(NMF/'input_matrix.npz')
    sites=pd.read_csv(NMF/'channel_assignments.csv')
    expected=pd.read_csv(ROOT/'docs/evidence/subject_task_site_lists_2026-10-17/nmf_candidate_sites_not_final_inputs.csv')
    assert len(sites)==227 and sites.subject.nunique()==64 and sites.channel.is_unique
    assert set(sites.channel)==set(expected.site_id)
    assert np.array_equal(factors['channels'],inputs['channels'])
    assert sites.channel.tolist()==inputs['channels'].tolist()
    features=pd.read_csv(NMF/'input_features.csv')
    assert inputs['X'].shape==(227,399)
    assert features.groupby('phase').size().to_dict()=={'stimulus':77,'delay':129,'response':193}
    response_times=features.loc[features.phase.eq('response'),'time']
    assert np.isclose(response_times.min(),-.5) and np.isclose(response_times.max(),1.)
    assert np.allclose(np.linalg.norm(inputs['X'],axis=1),1)
    assert np.isfinite(inputs['X']).all() and (inputs['X']>=0).all()
    assert np.allclose(np.linalg.norm(factors['H'],axis=1),1)
    assert np.array_equal(sites.functional_cluster.to_numpy(),factors['component_names'][factors['W'].argmax(axis=1)])
    assert np.allclose(np.linalg.norm(inputs['X']-factors['W']@factors['H']),model['reconstruction_error'])
    starts=pd.read_csv(NMF/'initialization_metrics.csv')
    assert len(starts)==20 and starts.converged.all()
    assert np.isclose(starts.reconstruction_error.min(),model['reconstruction_error'])
    rank_meta=json.loads((NMF/'rank_selection_meta.json').read_text())
    assert rank_meta['input_matrix_sha256']==sha(NMF/'input_matrix.npz')
    rank=pd.read_csv(NMF/'rank_selection_metrics.csv')
    assert set(rank.k)==set(range(2,7)) and rank.n_boot.eq(200).all() and rank.row_frac.eq(.8).all()
    nb=nbformat.read(NOTEBOOK,as_version=4)
    for i,c in enumerate(nb.cells):
        if c.cell_type=='code': ast.parse(c.source,filename=f'fig2-cell-{i}')
    nbformat.validate(nb)
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    jupyter=OUT/'jupyter'
    spec=jupyter/'kernels/insula-paper-fig2'
    spec.mkdir(parents=True,exist_ok=True)
    (spec/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],display_name='Insula Figure 2 (ieeg)',language='python')))
    os.environ['JUPYTER_PATH']=str(jupyter)+os.pathsep+os.environ.get('JUPYTER_PATH','')
    def started(cell,cell_index,**kwargs):
        print(f'{datetime.now(timezone.utc).isoformat()} cell {cell_index}: '
              f'{cell.source.splitlines()[0][:100] if cell.source else "empty"}',flush=True)
    client=NotebookClient(nb,kernel_name='insula-paper-fig2',timeout=1800,iopub_timeout=180,
        raise_on_iopub_timeout=True,resources={'metadata':{'path':str(ROOT/'vizpub')}},
        on_cell_start=started,record_timing=True)
    try: client.execute()
    except Exception:
        nbformat.write(nb,OUT/'fig2_failed.ipynb')
        raise
    assert all(sha(Path(p))==h for p,h in before.items()), 'Protected Figure 1 or legacy result changed'
    errors=[(i,o.get('ename'),o.get('evalue')) for i,c in enumerate(nb.cells) for o in c.get('outputs',[]) if o.get('output_type')=='error']
    assert not errors
    assert all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')
    assert not [i for i,c in enumerate(nb.cells) for o in c.get('outputs',[]) if 'image/png' in o.get('data',{})]
    assert not list((ROOT/'img/fig2').glob('*.png'))
    figure_manifest=json.loads((OUT/'figure_manifest.json').read_text())
    assert figure_manifest['n_sites']==227 and figure_manifest['n_subjects']==64
    assert figure_manifest['assignments_sha256']==sha(NMF/'channel_assignments.csv')
    assert figure_manifest['fit_windows']==model['fit_windows']
    assert len(figure_manifest['figures'])>=15
    nb.metadata['insula_figure2']=dict(cohort_version='2026-10-17',n_sites=227,n_subjects=64,k=3,
        phases=['stimulus','delay','response'],fit_windows=model['fit_windows'],
        executed_utc=datetime.now(timezone.utc).isoformat(),svg_inline_only=True)
    temporary=NOTEBOOK.with_suffix('.executed.tmp.ipynb')
    nbformat.write(nb,temporary)
    temporary.replace(NOTEBOOK)
    evidence=EVIDENCE
    evidence.mkdir(parents=True,exist_ok=True)
    for name in ['input_manifest.json','input_matrix.npz','input_sites.csv','input_features.csv',
        'site_waveforms.csv.gz','waveform_source_membership.csv','channel_assignments.csv','factors.npz',
        'H_by_phase.csv','component_phase_profiles.csv','initialization_metrics.csv','nmf_manifest.json',
        'rank_selection_metrics.csv','rank_selection_meta.json','chosen_k.json','pc_scores.csv',
        'pc_clustering_meta.json','comparison_with_old_four_phase.csv',
        'comparison_with_response_0_to1.csv','response_window_label_transitions.csv',
        'response_window_comparison.json']:
        shutil.copy2(NMF/name,evidence/name)
    for path in OUT.glob('*.csv*'): shutil.copy2(path,evidence/path.name)
    shutil.copy2(OUT/'figure_manifest.json',evidence/'figure_manifest.json')
    for folder,source in [('figures',ROOT/'img/fig2'),('rank_figures',ROOT/'img/nmf')]:
        destination=evidence/folder
        destination.mkdir(exist_ok=True)
        for path in source.glob('*.svg'): shutil.copy2(path,destination/path.name)
    validation=dict(passed=True,code_cells=sum(c.cell_type=='code' for c in nb.cells),
        all_code_cells_executed=True,output_errors=errors,svg_panels=len(figure_manifest['figures']),
        svg_inline_only=True,n_sites=227,n_subjects=64,k=3,
        phases=['stimulus','delay','response'],H_components_unit_norm=True,
        assignments_match_normalized_W_argmax=True,best_of_20_verified=True,
        rank_same_input_verified=True,fitted_model_rank_used_by_downstream=True,figure1_and_legacy_results_unchanged=True,
        previous_response_window_archive_unchanged=True,window_only_change_verified=True,
        fit_windows=model['fit_windows'],n_features=399,response_time_samples=193,
        notebook_sha256=sha(NOTEBOOK),protected_file_hashes=before)
    (OUT/'execution_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    shutil.copy2(OUT/'execution_validation.json',evidence/'execution_validation.json')
    (OUT/'fig2_failed.ipynb').unlink(missing_ok=True)
    print(json.dumps({k:v for k,v in validation.items() if k!='protected_file_hashes'},indent=2),flush=True)


if __name__=='__main__': main()
