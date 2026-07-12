"""Tests for run_decoder_patterns_resolved.py.

Verifies the math of the Haufe (2014) pattern extraction and that the time-resolved
sliding window properly saves the (channels, times) patterns.
"""
import numpy as np
import pytest
import os
import sys
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from mne.decoding import Vectorizer
from sklearn.preprocessing import StandardScaler
from ieeg.calc.oversample import MinimumNaNSplit

# Add scripts dir to path for import
scripts_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "src"
)
sys.path.insert(0, scripts_dir)
from run_decoder_patterns_resolved import compute_haufe_pattern, compute_patterns_cv


def test_haufe_math_correctness():
    """Haufe pattern should perfectly recover a pure linear generative model."""
    np.random.seed(42)
    N, F = 100, 10
    X = np.random.randn(N, F)
    # create a true latent signal driven by feature 0 (pos) and 1 (neg)
    s = 2.0 * X[:, 0] - 1.5 * X[:, 1] + np.random.randn(N) * 0.1
    y = (s > 0).astype(int)
    
    clf = LinearSVC(random_state=42)
    clf.fit(X, y)
    y_pred = clf.decision_function(X)
    
    A = compute_haufe_pattern(X, y_pred)
    
    # Feature 0 should have the highest positive pattern weight
    # Feature 1 should have the lowest negative pattern weight
    assert A[0] > max(A[2:])
    assert A[1] < min(A[2:])
    

def test_compute_patterns_cv():
    """Testing integration with scikit-learn CV and Pipelines."""
    np.random.seed(42)
    n_trials = 20
    n_channels = 3
    n_times = 5  # tiny window
    X = np.random.randn(n_trials, n_channels, n_times)
    
    # Feature index [0, 2] tracks the class perfectly
    s = 3.0 * X[:, 0, 2] + np.random.randn(n_trials) * 0.1
    y = (s > 0).astype(int)
    
    pipeline = make_pipeline(
        Vectorizer(),
        StandardScaler(),
        LinearSVC(random_state=42)
    )
    cv = MinimumNaNSplit(n_splits=2, n_repeats=1)
    
    avg_pattern = compute_patterns_cv(X, y, cv, pipeline)
    
    # Output shape should match a single trial window: (channels, times)
    assert avg_pattern.shape == (n_channels, n_times)
    
    # The true generating feature (Channel 0, Time 2) should have a strong positive pattern
    assert avg_pattern[0, 2] > 0
    assert avg_pattern[0, 2] == np.max(avg_pattern)
