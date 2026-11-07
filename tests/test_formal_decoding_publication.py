"""Synthetic end-to-end checks of complete-family publication, not real results."""
from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

import src.decoding.figure4_collect as collector
from src.decoding.formal_cv import METHOD,inference,sha256_file
from src.decoding.figure4_formal import unit_identity


@pytest.fixture
def publication(tmp_path,monkeypatch):
    """Small synthetic cohort; real inference dimensions retain 30x5 and B=500."""
    spec=dict(max_iter=10000,SE='split variability only',
        phase_windows={p:[0,.6 if p=='Stimulus' else 1] for p in ['Stimulus','Delay','Response']},
        time_resolved=dict(window=.3,step=.03))
    source=tmp_path/'frozen_input';source.write_bytes(b'synthetic fixture only')
    digest=sha256_file(source)
    models=[]
    for roi in ['Sensory','Sustain','Motor']:
        for phase in ['Stimulus','Delay','Response']:
            channels=[f'D{j:04d}_A1-2' for j in range(1,15)] if (roi,phase)==('Sustain','Delay') else ['D0001_A1-2','D0002_B1-2']
            kept=channels[:-1] if len(channels)==14 else channels
            models.append(dict(index=len(models),roi=roi,phase=phase,role='primary',
                path=str(source),input_sha256=digest,channels=channels,kept_channels=kept,
                n_trials=20,n_electrodes=len(kept),n_subjects=len(kept),
                stage=dict(first=0,last=6),resolved=[dict(index=k,first=k,last=k+6,
                    start=k/128,end_exclusive=(k+6)/128,time=(k+2.5)/128) for k in range(2)]))
    models.append(dict(models[4],index=9,role='sensitivity_full14',kept_channels=models[4]['channels'],n_electrodes=14,n_subjects=14))
    plan=dict(models=models,output=str(tmp_path/'results/fig4/formal'),
        stage_units=10,resolved_units=18,windows_per_curve=2,specification_sha256='synthetic')
    audit=tmp_path/'audit';audit.mkdir()
    config=tmp_path/'config';config.mkdir()
    (config/'figure4_phonetic_decoding.json').write_text(json.dumps(dict(multiple_comparisons=dict(
        resolved_method='ieeg_time_cluster',
        resolved_function='src.decoding.run_decoding_resolved.cluster_correction',
        resolved_p_thresh=.05,resolved_tails=1))))
    monkeypatch.setattr(collector,'ROOT',tmp_path)
    monkeypatch.setattr(collector,'AUDIT',audit)
    monkeypatch.setattr(collector,'load_plan',lambda:(plan,spec))
    monkeypatch.setattr(collector,'verify_protected',lambda:0)
    def write(mode,index):
        model,bounds,path=unit_identity(mode,index,plan)
        path.parent.mkdir(parents=True,exist_ok=True)
        threshold=.998-.02*index
        scores=threshold+np.linspace(-.005,.005,30)[:,None]+np.zeros((30,5))
        null=np.broadcast_to(np.linspace(0,.999,500)[:,None,None],(500,30,5)).copy()
        result=inference(scores,null)
        st=h5py.string_dtype('utf-8')
        with h5py.File(path,'w') as f:
            f.attrs.update(complete=True,all_fits_converged=True,inference_method=METHOD,
                result_schema_version=3,config_sha256='synthetic',input_sha256=digest,
                n_repeats=30,n_folds=5,n_perm=500,max_iter=10000,n_trials=20,n_electrodes=model['n_electrodes'],
                n_subjects=model['n_subjects'],sample_first=bounds['first'],sample_last_exclusive=bounds['last'],
                observed_imputation_sha256='same full-reference fill')
            for k,v in result.items():f[k]=v
            f['accuracy_repeats']=scores;f['perm_scores']=null;f['classes']=np.arange(4)
            f['permutation_completed']=np.ones(500,dtype=bool)
            f['observed_iterations']=np.full((30,5),10)
            f['permutation_iterations']=np.full((500,30,5),20)
            f['observed_fold_assignment']=np.tile(np.arange(20)%5,(30,1)).astype('u1')
            f['permutation_fold_assignment']=np.tile(np.arange(20)%5,(500,30,1)).astype('u1')
            f['permutation_plan/labels']=np.tile(np.arange(20)%4,(500,1))
            f.create_dataset('permutation_imputation_sha256',data=np.full(500,b'paired',dtype='S64'))
            for k,v in dict(channel=model['kept_channels'],input_channel=model['channels'],trial=[f'item{i}' for i in range(20)]).items():
                f.create_dataset(k,data=np.asarray(v,dtype=st))
        return result['p_value']
    return plan,write


def test_stage_publication_corrects_all_nine_and_excludes_full14_from_family(publication):
    plan,write=publication
    p=[write('stage',i) for i in range(10)]
    collector.collect_stage()
    actual=collector.validate_stage()
    expected=multipletests(p[:9],method='fdr_bh')[1]
    np.testing.assert_array_equal([v['p_value_fdr'] for v in actual],expected)
    wrong=multipletests(p,method='fdr_bh')[1][:9]
    assert not np.array_equal(expected,wrong)
    with h5py.File(collector.primary_path(plan['models'][4]),'r+') as f:
        f['p_value_fdr'][()]=.9
    with pytest.raises(AssertionError):collector.validate_stage()


def test_resolved_publication_uses_original_timeperm_not_pointwise_FDR(publication,monkeypatch):
    plan,write=publication
    p=[write('resolved',i) for i in range(18)]
    def forbidden(*args,**kwargs):
        raise AssertionError('Pointwise multiple-comparison replacement is forbidden for temporal decoding')
    monkeypatch.setattr(collector,'multipletests',forbidden)
    collector.collect_resolved()
    table=pd.read_csv(Path(plan['output'])/'resolved/summary.csv')
    assert 'p_value_fdr' not in table and 'significant_fdr' not in table
    np.testing.assert_allclose(table.p_value,p,atol=1e-15)
    assert table.correction.eq('ieeg_time_cluster').all()
    first=Path(plan['output'])/'resolved/curves/Sensory_Stimulus.h5'
    with h5py.File(first,'r') as f:
        assert f.attrs['correction']=='ieeg_time_cluster'
        assert f.attrs['correction_family_size']==2
        assert f['accuracy_repeats'].shape==(2,30,5)
        assert f['perm_scores'].shape==(2,500,30,5)
        assert len(f['trial'])==20 and len(f['channel'])==2
        expected,raw=collector.cluster_correction(f['accuracy_repeats'][:].mean(axis=(1,2)),
            f['perm_scores'][:].mean(axis=(2,3)).T,p_thresh=.05,tails=1)
        np.testing.assert_array_equal(f['mask'][:],expected)
        np.testing.assert_array_equal(f['p_values'][:],raw)
        assert 'p_value_fdr' not in f
