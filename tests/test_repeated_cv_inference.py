"""Independent checks of coherent relabeling, cached folds, and output semantics."""
import numpy as np
import pytest
from mne.decoding import Vectorizer
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.decoding.decoder import sample_fold
from src.decoding.repeated_cv import METHOD, repeated_cv_permutation, whole_stimulus_permutations
from src.decoding.run_decoding import save_window_result
from src.decoding.windowed_results import load_windowed_result


def test_whole_stimulus_draws_preserve_repetitions_and_trial_class_totals():
    groups = np.repeat(np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']), [2,2,2,2,3,3,3,3])
    labels = np.repeat(np.array([0,1,2,3,0,1,2,3]), [2,2,2,2,3,3,3,3])
    plan = whole_stimulus_permutations(labels, groups, 12)
    for draw, indices in zip(plan['labels'], plan['row_permutations']):
        np.testing.assert_array_equal(np.sort(indices), np.arange(len(labels)))
        np.testing.assert_array_equal(draw, labels[indices])
        np.testing.assert_array_equal(np.bincount(draw), np.bincount(labels))
        assert all(np.unique(draw[groups == name]).size == 1 for name in np.unique(groups))
    bad = labels.copy();bad[0] = 3
    with pytest.raises(ValueError, match='constant label'):
        whole_stimulus_permutations(bad, groups, 2)


@pytest.fixture(scope='module')
def fitted():
    rng = np.random.RandomState(8)
    labels = np.repeat(np.arange(24) % 4, 2)
    groups = np.repeat([f'item{i:02d}' for i in range(24)], 2)
    X = rng.normal(size=(48,4,6))
    X[np.arange(48),labels] += .6
    X[0] = np.nan;X[17,2,1] = np.nan
    transform = make_pipeline(Vectorizer(),StandardScaler(),PCA(.80,random_state=42))
    classifier = LinearSVC(C=1.,random_state=42,max_iter=5000)
    result = repeated_cv_permutation(X,labels,groups,transform,classifier,
        keep_channels=[0,2,3],n_splits=3,n_repeats=2,n_permutations=3,n_jobs=2)
    return X,labels,groups,transform,classifier,result


def test_cached_transform_matches_full_pipeline_refitted_for_each_global_draw(fitted):
    X,y,_,transform,classifier,result = fitted
    expected = np.empty((3,2,3))
    for pi,labels in enumerate(result['permutation_plan']['labels']):
        for fold in result['folds']:
            # Independent brute-force reference: impute with relabeled targets,
            # delete the analysis channel, then refit the complete pipeline.
            a,b,ya,yb = sample_fold(X,labels,fold.train,fold.test,seed=42+fold.repeat+fold.fold)
            pipe = make_pipeline(*[clone(step) for _,step in transform.steps],clone(classifier))
            pipe.fit(a[:,[0,2,3],:],ya)
            expected[pi,fold.repeat,fold.fold] = balanced_accuracy_score(yb,pipe.predict(b[:,[0,2,3],:]))
    np.testing.assert_array_equal(result['perm_scores'],expected)
    expected_mean = result['accuracy_repeats'].mean()
    expected_p = (1+np.count_nonzero(expected.mean(axis=(1,2))>=expected_mean))/4
    assert result['p_value'] == expected_p


def test_result_reader_requires_matching_mean_null_and_reports_repeat_SE(tmp_path,fitted):
    *_,result = fitted
    target = tmp_path/'result.h5'
    save_window_result(target,result,metadata=dict(n_repeats=2,n_folds=3,n_perm=3),
        input_metadata=dict(channel=['D1_A1-2','D1_A3-4','D1_A4-5'],input_channel=['D1_A1-2','D1_A2-3','D1_A3-4','D1_A4-5'],
            excluded_channel=['D1_A2-3'],trial=[f'trial{i}' for i in range(48)],condition=['item']*48),inference=METHOD)
    loaded = load_windowed_result(target,expected_method=METHOD)
    assert loaded['n_electrodes'] == 3
    np.testing.assert_array_equal(loaded['scores'],result['accuracy_repeats'].mean(axis=1))
    assert loaded['p_value'] == result['p_value']
    assert loaded['SE'] == result['accuracy_se']
    with pytest.raises(ValueError,match='expected'):
        load_windowed_result(target,expected_method='first_repeat')
