import numpy as np
import sys
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from mne.decoding import Vectorizer
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cross_decoder import (
    CrossDecoder,
    cross_domain_resolved_permutation_scores,
    cross_domain_generalized_permutation_scores,
)
import src.run_cross_roi_resolved as run_cross_roi_resolved
import src.run_cross_roi_generalized as run_cross_roi_generalized


def _make_data():
    rng = np.random.RandomState(0)
    n_epochs = 12
    n_times = 12
    y = np.array([0, 1] * 6)
    latent = rng.randn(n_epochs, 2, n_times)
    latent[y == 1, 0, 4:8] += 1.0
    w1 = rng.randn(2, 4)
    w2 = rng.randn(2, 5)
    X1 = np.einsum("ect,cf->eft", latent, w1) + 0.05 * rng.randn(n_epochs, 4, n_times)
    X2 = np.einsum("ect,cf->eft", latent, w2) + 0.05 * rng.randn(n_epochs, 5, n_times)
    return X1, y.copy(), X2, y.copy()


def _make_decoder():
    estimator = make_pipeline(
        Vectorizer(),
        StandardScaler(),
        PCA(n_components=0.80, random_state=42),
        LinearSVC(random_state=42, max_iter=10000),
    )
    return CrossDecoder(estimator=estimator, n_components=2, random_state=42)


def test_cross_roi_runner_modules_import():
    assert hasattr(run_cross_roi_resolved, "main")
    assert hasattr(run_cross_roi_generalized, "main")


def test_cross_roi_resolved_small_synthetic():
    X1, y1, X2, y2 = _make_data()
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    decoder = _make_decoder()

    obs_scores, perm_scores, pvals = cross_domain_resolved_permutation_scores(
        X1=X1,
        y1=y1,
        X2=X2,
        y2=y2,
        cv=cv,
        cross_decoder=decoder,
        n_permutations=2,
        n_jobs=1,
        random_state=42,
        window=0.25,
        step=0.25,
        fs=4,
        tmin=0.0,
        tmax=3.0,
    )

    assert obs_scores.ndim == 2
    assert perm_scores.ndim == 3
    assert pvals.ndim == 1
    assert obs_scores.shape[1] == 2
    assert perm_scores.shape[1:] == (2, 2)
    assert pvals.shape[0] == obs_scores.shape[0]


def test_cross_roi_generalized_small_synthetic():
    X1, y1, X2, y2 = _make_data()
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    decoder = _make_decoder()

    obs_scores, perm_scores, pvals = cross_domain_generalized_permutation_scores(
        X1=X1,
        y1=y1,
        X2=X2,
        y2=y2,
        cv=cv,
        cross_decoder=decoder,
        n_permutations=2,
        n_jobs=1,
        random_state=42,
        window=0.25,
        step=0.25,
        fs=4,
        train_tmin=0.0,
        train_tmax=3.0,
        test_tmin=0.0,
        test_tmax=3.0,
    )

    assert obs_scores.ndim == 3
    assert perm_scores.ndim == 4
    assert pvals.ndim == 2
    assert obs_scores.shape[2] == 2
    assert perm_scores.shape[2:] == (2, 2)
    assert pvals.shape == obs_scores.shape[:2]
