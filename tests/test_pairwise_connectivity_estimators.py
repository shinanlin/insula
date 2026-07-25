import numpy as np
import pandas as pd

from src.connectivity.pairwise.config import ConnectivityConfig
from src.connectivity.pairwise.oaec import (
    compute_oaec,
    directional_orthogonalized_correlation_z,
    gaussian_analytic_filterbank,
)
from src.connectivity.pairwise.permutation import generate_derangements
from src.connectivity.pairwise.tf_dwpli import (
    _band_frequency_mask,
    debiased_wpli_from_imag,
    morlet_coefficients,
)
from src.connectivity.pairwise.config import wpli_frequencies
from src.connectivity.pairwise.xcorr import (
    compute_xcorr,
    lagged_cross_trial_pearson_z,
)


def _one_pair() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pair_id": ["S1-2__T1-2"],
            "source": ["S1-2"],
            "target": ["T1-2"],
            "source_index": [0],
            "target_index": [1],
        }
    )


def test_xcorr_detects_signed_hga_amplitude_lag():
    rng = np.random.default_rng(10)
    source = rng.normal(size=(36, 120))
    kernel = np.exp(-0.5 * (np.arange(-5, 6) / 2.0) ** 2)
    source = np.asarray(
        [np.convolve(trial, kernel, mode="same") for trial in source]
    )
    target = np.roll(source, 7, axis=1)
    values = lagged_cross_trial_pearson_z(
        source, target, np.arange(-12, 13)
    )
    observed = np.nanmean(
        values[:, np.arange(36), np.arange(36)], axis=1
    )
    assert np.arange(-12, 13)[np.nanargmax(observed)] == -7

    data = np.stack((source, target), axis=1).astype(np.float32)
    config = ConnectivityConfig(n_perm=30, max_lag_s=0.12)
    permutations = generate_derangements(30, 36, 7)
    result = compute_xcorr(
        data, 100.0, _one_pair(), permutations, config
    )
    assert result.pair_table.loc[0, "peak_lag_s"] == -0.07
    assert result.pair_table.loc[0, "peak_r"] > 0.9
    assert bool(result.pair_table.loc[0, "qc_pass"])


def test_oaec_operates_on_complex_signal_before_envelope():
    n_trials, n_time = 40, 96
    phase = np.linspace(0.0, 10.0 * np.pi, n_time)
    amplitude = (
        2.0
        + np.sin(np.linspace(0.0, 4.0 * np.pi, n_time))[None, :]
        + np.linspace(0.0, 0.4, n_trials)[:, None]
    )
    source = amplitude * np.exp(1j * phase[None, :])
    target = amplitude * np.exp(1j * (phase[None, :] + np.pi / 3.0))
    directional = directional_orthogonalized_correlation_z(source, target)
    assert np.nanmean(directional) > 0.99

    zero_phase = directional_orthogonalized_correlation_z(source, source)
    assert np.nanmax(np.abs(zero_phase)) < 1e-6


def test_oaec_shuffled_trials_are_reorthogonalized(monkeypatch):
    import src.connectivity.pairwise.oaec as oaec_module

    rng = np.random.default_rng(2)
    n_trials, n_time = 34, 80
    phase = np.linspace(0.0, 8.0 * np.pi, n_time)
    innovations = rng.normal(size=(n_trials, n_time))
    smoothed = np.asarray(
        [
            np.convolve(trial, np.ones(9) / 9.0, mode="same")
            for trial in innovations
        ]
    )
    amplitude = np.exp(0.5 * smoothed)
    coefficients = np.empty(
        (n_trials, 2, 1, n_time), dtype=np.complex64
    )
    coefficients[:, 0, 0] = amplitude * np.exp(1j * phase)
    coefficients[:, 1, 0] = amplitude * np.exp(
        1j * (phase + np.pi / 4.0)
    )

    def fake_filterbank(data, sfreq, centers, **kwargs):
        assert np.isrealobj(data)
        assert kwargs["target_sfreq"] == 128.0
        return coefficients

    monkeypatch.setattr(
        oaec_module, "hga_filterbank_centers", lambda: np.asarray([100.0])
    )
    monkeypatch.setattr(
        oaec_module, "gaussian_analytic_filterbank", fake_filterbank
    )
    raw = rng.normal(size=(n_trials, 2, 200)).astype(np.float32)
    times = np.linspace(-1.0, 1.0, 200, endpoint=False)
    config = ConnectivityConfig(
        n_perm=30, pair_block_size=1, permutation_chunk_size=7
    )
    permutations = generate_derangements(30, n_trials, 12)
    result = compute_oaec(
        raw,
        times,
        100.0,
        "Response",
        _one_pair(),
        permutations,
        config,
    )
    assert result.detail.attrs["orthogonalization"] == (
        "Hipp_pairwise_bidirectional"
    )
    assert result.pair_table.loc[0, "stat"] > 0.95
    assert result.pair_table.loc[0, "null_mean"] < 0.5


def test_gaussian_filterbank_returns_band_specific_complex_coefficients():
    sfreq = 512.0
    times = np.arange(1024) / sfreq
    raw = np.sin(2 * np.pi * 100.0 * times)[None, None].astype(np.float32)
    mask = (times >= 0.5) & (times < 1.5)
    coefficients = gaussian_analytic_filterbank(
        raw,
        sfreq,
        [100.0],
        time_mask=mask,
        target_sfreq=128.0,
    )
    assert np.iscomplexobj(coefficients)
    assert coefficients.shape == (1, 1, 1, 128)
    assert np.mean(np.abs(coefficients)) > 0.5


def test_dwpli_detects_nonzero_phase_and_suppresses_zero_phase():
    magnitudes = np.linspace(0.5, 2.0, 50)
    value, denominator = debiased_wpli_from_imag(magnitudes)
    assert value > 0.99
    assert denominator > 0

    zero, zero_denominator = debiased_wpli_from_imag(np.zeros(50))
    assert np.isnan(zero)
    assert zero_denominator == 0

    negative, _ = debiased_wpli_from_imag(
        np.asarray([1.0, 1.0, -1.0, -1.0])
    )
    assert negative < 0

    voltage_scale, _ = debiased_wpli_from_imag(magnitudes * 1e-12)
    assert voltage_scale > 0.99

    phase = np.linspace(0.0, 6.0 * np.pi, 50)
    source = np.exp(1j * phase)
    lagged_target = np.exp(1j * (phase - np.pi / 3.0))
    lagged_cross_spectrum = source * np.conj(lagged_target)
    phase_lag_value, _ = debiased_wpli_from_imag(
        np.imag(lagged_cross_spectrum)
    )
    assert phase_lag_value > 0.99


def test_broadband_frequency_mask_spans_full_wpli_grid():
    freqs = wpli_frequencies(step=1.0)
    broadband = _band_frequency_mask(freqs, "broadband")
    assert broadband[0]
    assert broadband[-1]
    assert np.all(broadband)
    assert np.sum(broadband) == len(freqs)


def test_wpli_morlet_input_is_complex_low_frequency_coefficients():
    sfreq = 128.0
    times = np.arange(256) / sfreq
    data = np.sin(2 * np.pi * 10.0 * times)[None, None]
    coefficients = morlet_coefficients(
        data.astype(np.float32),
        sfreq,
        np.asarray([8.0, 10.0, 12.0]),
        np.asarray([4.0, 4.0, 4.0]),
    )
    assert coefficients.dtype == np.complex64
    power = np.mean(np.abs(coefficients[0, 0]) ** 2, axis=-1)
    assert power[1] > power[0]
    assert power[1] > power[2]
