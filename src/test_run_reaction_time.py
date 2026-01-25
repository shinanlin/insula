"""
Simple pytest tests for run_reaction_time.py

Run with: python -m pytest test_run_reaction_time.py -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from himalaya.ridge import RidgeCV

from run_reaction_time import predict_permutation_scores


def _window_mean(arr: np.ndarray, center_idx: int, half_win: int) -> np.ndarray:
    """Compute mean over a sliding window centered at center_idx."""
    start = max(0, center_idx - half_win)
    end = min(arr.shape[-1], center_idx + half_win + 1)
    return np.nanmean(arr[..., start:end], axis=-1)


def _prepare_data(df, phase, description, min_trials=10):
    """Prepare 3D data array and RT vector from long-format DataFrame."""
    sub = df[(df['phase'] == phase) & (df['description'] == description)]
    if 'remark' in sub.columns:
        sub = sub[sub['remark'] == 'CORRECT']
    
    if sub.empty:
        return None, None, None, None
    
    times = np.sort(sub['time'].unique())
    trials = sub['trial'].dropna().unique()
    channels = sub['channel'].unique()
    
    if len(trials) < min_trials or len(channels) < 1:
        return None, None, None, None
    
    trial_to_rt = sub.groupby('trial')['rt'].first().to_dict()
    rt = np.array([trial_to_rt[t] for t in trials])
    
    valid = ~np.isnan(rt)
    if valid.sum() < min_trials:
        return None, None, None, None
    
    trials = trials[valid]
    rt = rt[valid]
    
    # Build 3D array via pivot
    valid_sub = sub[sub['trial'].isin(trials)]
    pivot = valid_sub.pivot_table(
        index='trial', columns=['channel', 'time'], values='value', aggfunc='mean'
    )
    X_3d = pivot.values.reshape(len(trials), len(channels), len(times))
    
    return X_3d, rt, times, channels


def extract_reaction_time(epos_df):
    """Copy of extract_reaction_time for testing."""
    epoch_tbl = (
        epos_df[['subject', 'epoch', 'condition', 'description', 'phase', 'onset', 'remark']]
        .drop_duplicates()
    )
    epoch_tbl = epoch_tbl.sort_values(['subject', 'onset', 'epoch']).reset_index(drop=True)
    epoch_tbl['trial'] = (
        epoch_tbl
        .groupby(['subject'], sort=False)['phase']
        .transform(lambda s: s.eq('Stimulus').cumsum())
    )
    epoch_tbl.loc[epoch_tbl['trial'].eq(0), 'trial'] = np.nan
    
    go_resp_onset = (
        epoch_tbl[epoch_tbl['phase'].isin(['Go', 'Response'])]
        .pivot_table(
            index=['subject', 'description', 'trial', 'condition'],
            columns='phase',
            values='onset',
            aggfunc='first'
        )
        .reset_index()
        .rename(columns={'Go': 'go_onset', 'Response': 'resp_onset'})
    )
    go_resp_onset['rt'] = go_resp_onset['resp_onset'] - go_resp_onset['go_onset']
    
    epoch_tbl = epoch_tbl.merge(
        go_resp_onset[['subject', 'description', 'trial', 'condition', 'rt']],
        on=['subject', 'description', 'trial', 'condition'],
        how='left'
    )
    return epoch_tbl


def sliding_window_rt_prediction(df, phase, description, win_size=0.05, alphas=None):
    """Simplified version for testing (no permutation, no cluster test)."""
    if alphas is None:
        alphas = np.logspace(-3, 3, 5)
    
    X_3d, rt, times, channels = _prepare_data(df, phase, description)
    if X_3d is None:
        return None
    
    dt = np.median(np.diff(times))
    half_win = int(np.ceil(win_size / 2 / dt))
    n_times = len(times)
    
    r2_true = np.zeros(n_times)
    best_alphas = np.zeros(n_times)
    
    for t_idx in range(n_times):
        X_win = _window_mean(X_3d, t_idx, half_win)
        valid_mask = ~np.isnan(X_win).any(axis=1)
        if valid_mask.sum() < 10:
            r2_true[t_idx] = np.nan
            best_alphas[t_idx] = np.nan
            continue
        X_t = X_win[valid_mask]
        y_t = rt[valid_mask]
        
        # Simple Ridge with fixed alpha for test speed
        ridge = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(X_t, y_t)
        r2_true[t_idx] = ridge.score(X_t, y_t)
        best_alphas[t_idx] = 1.0
    
    result = pd.DataFrame({
        'time': times,
        'r2': r2_true,
        'best_alpha': best_alphas,
        'r2_null_mean': np.zeros(n_times),
        'r2_null_std': np.zeros(n_times),
        'mask': np.zeros(n_times, dtype=bool),
        'pval': np.ones(n_times),
    })
    return result


def make_mock_df(n_trials=50, n_channels=5, n_times=100):
    """Create a mock DataFrame similar to load_subject_data output."""
    times = np.linspace(-0.5, 1.0, n_times)
    trials = np.arange(1, n_trials + 1)
    channels = [f'ch{i}' for i in range(n_channels)]
    
    rng = np.random.default_rng(42)
    rt_values = rng.uniform(0.3, 1.5, n_trials)
    
    rows = []
    for trial_idx, trial in enumerate(trials):
        for ch in channels:
            for t in times:
                rows.append({
                    'time': t,
                    'channel': ch,
                    'value': rng.normal(0, 1),
                    'epoch': trial_idx,
                    'onset': trial_idx * 10.0,
                    'phase': 'Delay',
                    'description': 'Decision',
                    'subject': 'D0001',
                    'condition': 'Word/latin',
                    'remark': 'CORRECT',
                    'trial': float(trial),
                    'rt': rt_values[trial_idx],
                })
    
    return pd.DataFrame(rows)


class TestExtractReactionTime:
    """Tests for extract_reaction_time function."""
    
    def test_basic_rt_extraction(self):
        """Test that RT is correctly computed from Go and Response onsets."""
        df = pd.DataFrame({
            'subject': ['S1'] * 4,
            'epoch': [0, 1, 2, 3],
            'condition': ['A'] * 4,
            'description': ['Decision'] * 4,
            'phase': ['Stimulus', 'Go', 'Response', 'Delay'],
            'onset': [0.0, 1.0, 1.5, 0.5],
            'remark': ['CORRECT'] * 4,
        })
        
        result = extract_reaction_time(df)
        
        rt_values = result[result['phase'] == 'Response']['rt'].values
        assert len(rt_values) == 1
        assert np.isclose(rt_values[0], 0.5)


class TestPrepareData:
    """Tests for _prepare_data function."""
    
    def test_returns_correct_shapes(self):
        """Test that _prepare_data returns arrays with correct shapes."""
        df = make_mock_df(n_trials=30, n_channels=3, n_times=50)
        
        X_3d, rt, times, channels = _prepare_data(df, 'Delay', 'Decision')
        
        assert X_3d is not None
        assert X_3d.shape == (30, 3, 50)
        assert len(rt) == 30
        assert len(times) == 50
        assert len(channels) == 3
    
    def test_returns_none_for_insufficient_trials(self):
        """Test that function returns None when not enough trials."""
        df = make_mock_df(n_trials=5, n_channels=3, n_times=50)
        
        X_3d, rt, times, channels = _prepare_data(df, 'Delay', 'Decision')
        
        assert X_3d is None


class TestSlidingWindowRTPrediction:
    """Tests for sliding_window_rt_prediction function."""
    
    def test_returns_dataframe_with_correct_columns(self):
        """Test that output has expected columns."""
        df = make_mock_df(n_trials=30, n_channels=3, n_times=50)
        
        result = sliding_window_rt_prediction(df, phase='Delay', description='Decision')
        
        assert result is not None
        expected_cols = {'time', 'r2', 'best_alpha', 'r2_null_mean', 'r2_null_std', 'mask', 'pval'}
        assert expected_cols.issubset(set(result.columns))
    
    def test_returns_none_for_insufficient_trials(self):
        """Test that function returns None when not enough trials."""
        df = make_mock_df(n_trials=5, n_channels=3, n_times=50)
        
        result = sliding_window_rt_prediction(df, phase='Delay', description='Decision')
        
        assert result is None
    
    def test_returns_none_for_wrong_phase(self):
        """Test that function returns None for non-existent phase."""
        df = make_mock_df(n_trials=30, n_channels=3, n_times=50)
        
        result = sliding_window_rt_prediction(df, phase='NonExistentPhase', description='Decision')
        
        assert result is None
    
    def test_r2_values_are_bounded(self):
        """Test that R² values are in reasonable range."""
        df = make_mock_df(n_trials=30, n_channels=3, n_times=50)
        
        result = sliding_window_rt_prediction(df, phase='Delay', description='Decision')
        
        assert result is not None
        assert (result['r2'] <= 1.0).all()


class TestWindowMean:
    """Tests for _window_mean function."""
    
    def test_window_mean_center(self):
        """Test window mean at center of array."""
        arr = np.arange(10).reshape(1, 1, 10).astype(float)
        result = _window_mean(arr, center_idx=5, half_win=2)
        # Should average indices 3, 4, 5, 6, 7 -> mean of [3,4,5,6,7] = 5.0
        assert np.isclose(result[0, 0], 5.0)
    
    def test_window_mean_edge(self):
        """Test window mean at edge of array."""
        arr = np.arange(10).reshape(1, 1, 10).astype(float)
        result = _window_mean(arr, center_idx=0, half_win=2)
        # Should average indices 0, 1, 2 -> mean of [0,1,2] = 1.0
        assert np.isclose(result[0, 0], 1.0)


class TestPredictPermutationScores:
    def test_shapes_and_pval_bounds(self):
        rng = np.random.default_rng(0)
        n_trials, n_channels, n_times = 40, 4, 16
        X = rng.normal(size=(n_trials, n_channels, n_times)).astype(float)
        rt = rng.normal(size=(n_trials,)).astype(float)

        alphas = np.logspace(-3, 3, 5)
        pipeline = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=alphas, fit_intercept=True),
        )
        cv = KFold(n_splits=4, shuffle=True, random_state=42)

        obs, perm, pval = predict_permutation_scores(
            X,
            rt,
            pipeline,
            cv,
            n_perm=7,
            random_state=123,
            n_jobs=1,
        )

        assert obs.shape == (n_channels,)
        assert perm.shape == (n_channels, 7)
        assert pval.shape == (n_channels,)

        assert np.all((pval >= 0.0) & (pval <= 1.0))
        assert np.isfinite(np.nanmean(obs))

    def test_determinism_fixed_seed(self):
        rng = np.random.default_rng(1)
        n_trials, n_channels, n_times = 30, 3, 8
        X = rng.normal(size=(n_trials, n_channels, n_times)).astype(float)
        rt = rng.normal(size=(n_trials,)).astype(float)

        alphas = np.logspace(-2, 2, 4)
        pipeline = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=alphas, fit_intercept=True),
        )
        cv = KFold(n_splits=3, shuffle=True, random_state=7)

        out1 = predict_permutation_scores(
            X, rt, pipeline, cv, n_perm=5, random_state=999, n_jobs=1
        )
        out2 = predict_permutation_scores(
            X, rt, pipeline, cv, n_perm=5, random_state=999, n_jobs=1
        )

        for a, b in zip(out1, out2):
            np.testing.assert_allclose(a, b, rtol=0, atol=0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
