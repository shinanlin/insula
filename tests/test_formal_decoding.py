"""Independent checks of the approved full-algorithm permutation test."""
import numpy as np
import pytest
from ieeg.calc.oversample import MinimumNaNSplit
from sklearn.metrics import balanced_accuracy_score

from src.decoding.decoder import sample_fold
from src.decoding.formal_cv import (
    ConvergenceFailure,evaluate_labels,inference,pipeline,sample_windows,unit_solver_specification,
)


def test_approved_solver_ceiling_applies_only_to_stage_motor_response(dataset):
    *_,spec=dataset
    amendment=dict(mode='stage',roi='Motor',phase='Response',max_iter=30000,
        approved_date='2026-10-20',reason='approved convergence check',
        base_specification_sha256='base',approval_file='approval.json',approval_sha256='frozen')
    plan=dict(specification_sha256='base',solver_amendments=[amendment])
    for mode in ['stage','resolved']:
        for roi in ['Sensory','Sustain','Motor']:
            for phase in ['Stimulus','Delay','Response']:
                effective,provenance=unit_solver_specification(spec,plan,dict(roi=roi,phase=phase),mode)
                if (mode,roi,phase)==('stage','Motor','Response'):
                    assert effective==dict(spec,max_iter=30000) and provenance==amendment
                else:
                    assert effective is spec and provenance is None
    assert spec['max_iter']==10000
    with pytest.raises(ValueError,match='other parameter changes'):
        unit_solver_specification(spec,dict(plan,solver_amendments=[dict(amendment,C=10)]),
            dict(roi='Motor',phase='Response'),'stage')
    with pytest.raises(ValueError,match='different base'):
        unit_solver_specification(spec,dict(plan,specification_sha256='changed'),
            dict(roi='Motor',phase='Response'),'stage')
from src.decoding.repeated_cv import whole_stimulus_permutations


@pytest.fixture
def dataset():
    rng=np.random.RandomState(17)
    y=np.repeat(np.arange(24)%4,2)
    groups=np.repeat([f"item{i:02d}" for i in range(24)],2)
    X=rng.normal(size=(48,4,6))
    X[np.arange(48),y]+=.3
    X[0,2,1]=np.nan;X[17,0]=np.nan
    spec=dict(n_repeats=2,n_folds=3,n_permutations=3,random_state=42,
        variance=.8,C=1.,max_iter=10000,tol=1e-4)
    return X,y,groups,spec


def test_restratified_draw_matches_independent_full_pipeline(dataset):
    X,y,groups,spec=dataset
    labels=whole_stimulus_permutations(y,groups,3)["labels"][1]
    result=evaluate_labels(X,labels,[0,2,3],spec)
    expected=np.empty((2,3))
    expected_folds=np.zeros((2,len(y)),dtype=np.uint8)
    for r in range(2):
        splitter=MinimumNaNSplit(n_splits=3,n_repeats=1,random_state=42+r)
        for fold,(tr,te) in enumerate(splitter.split(X,labels)):
            a,b,ya,yb=sample_fold(X,labels,tr,te,seed=42+r+fold)
            model=pipeline(spec)
            model.fit(a[:,[0,2,3]],ya)
            expected[r,fold]=balanced_accuracy_score(yb,model.predict(b[:,[0,2,3]]))
            expected_folds[r,te]=fold
    np.testing.assert_array_equal(result["scores"],expected)
    np.testing.assert_array_equal(result["fold_assignment"],expected_folds)
    actual=evaluate_labels(X,y,[0,2,3],spec)
    assert not np.array_equal(result["fold_assignment"],actual["fold_assignment"])


def test_channel_exclusion_preserves_reference_splits_and_numeric_imputation(dataset):
    X,y,groups,spec=dataset
    for labels in [y,whole_stimulus_permutations(y,groups,1)["labels"][0]]:
        full=evaluate_labels(X,labels,[0,1,2,3],spec)
        reduced=evaluate_labels(X,labels,[0,2,3],spec)
        np.testing.assert_array_equal(full["fold_assignment"],reduced["fold_assignment"])
        assert full["imputation_sha256"]==reduced["imputation_sha256"]


def test_matching_average_null_and_split_se():
    true=np.array([[.3,.4],[.5,.6],[.4,.5]])
    null=np.array([true-.1,true+.1,true-.05])
    result=inference(true,null)
    assert result["p_value"]==.5
    assert result["accuracy_stable"]==pytest.approx(true.mean())
    assert result["accuracy_se"]==pytest.approx(np.std(true.mean(1),ddof=1)/np.sqrt(3))
    null[0,0,0]=np.nan
    with pytest.raises(ValueError,match="incomplete/failed"):
        inference(true,null)


def test_sample_grid_has_complete_fixed_width_windows_and_actual_centers():
    windows=sample_windows(-.5,128,256,tmin=-.5,tmax=1.5,window=.3,step=.03)
    assert len(windows)==57
    for w in windows:
        assert w["last"]-w["first"]==38
        assert 0<=w["first"]<w["last"]<=256
        center=-.5+(w["first"]+w["last"]-1)/(2*128)
        assert w["time"]==center
    assert len({w["first"] for w in windows})==57
    with pytest.raises(ValueError,match="Incomplete"):
        sample_windows(-.5,128,250,tmin=-.5,tmax=1.5,window=.3,step=.03)


def test_nonconvergence_fails_the_unit_instead_of_dropping_a_draw(dataset):
    X,y,_,spec=dataset
    with pytest.raises(ConvergenceFailure,match="no draw was discarded"):
        evaluate_labels(X,y,[0,1,2,3],dict(spec,max_iter=1))


def test_partial_unit_resumes_and_reader_reconstructs_matching_inference(tmp_path,monkeypatch,dataset):
    import h5py
    import src.decoding.figure4_formal as workflow
    from src.decoding.formal_cv import METHOD,phase_window,sha256_file
    from src.decoding.windowed_results import load_windowed_result
    X,y,groups,spec=dataset
    path=tmp_path/'input.h5'
    st=h5py.string_dtype('utf-8')
    with h5py.File(path,'w') as f:
        f['X']=X;f['y']=y
        f.create_dataset('trial',data=np.asarray([f'{g}_{i%2}' for i,g in enumerate(groups)],dtype=st))
        f.create_dataset('condition',data=np.asarray(groups,dtype=st))
    channels=['D1_A1-2','D1_A2-3','D2_B1-2','D2_B2-3']
    model=dict(index=0,roi='Sensory',phase='Stimulus',role='primary',path=str(path),
        input_sha256=sha256_file(path),channels=channels,kept_channels=[channels[i] for i in [0,2,3]],
        excluded_channels=[channels[1]],n_trials=48,n_electrodes=3,n_subjects=2,fs=128,
        assignment_sha256='frozen',stage=phase_window(0,128,6,[0,6/128]))
    plan=dict(models=[model],output=str(tmp_path/'output'),specification_sha256='test-spec')
    monkeypatch.setattr(workflow,'load_plan',lambda:(plan,dict(spec,preflight_permutations=1)))
    workflow.fit_unit('stage',0,n_jobs=2,preflight_only=True)
    _,_,result_path=workflow.unit_identity('stage',0,plan)
    assert not result_path.exists()
    with h5py.File(result_path.with_suffix('.pending.h5'),'r') as f:
        assert f['permutation_completed'][:].sum()==1
        saved=f['perm_scores'][0]
    workflow.fit_unit('stage',0,n_jobs=2)
    value=load_windowed_result(result_path,expected_method=METHOD)
    with h5py.File(result_path,'r') as f:
        np.testing.assert_array_equal(f['perm_scores'][0],saved)
        assert f['permutation_completed'][:].all()
        expected=inference(f['accuracy_repeats'][:],f['perm_scores'][:])
    assert value['p_value']==expected['p_value']
    assert value['SE']==expected['accuracy_se']
