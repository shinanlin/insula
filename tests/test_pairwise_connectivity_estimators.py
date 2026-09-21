import numpy as np
import pandas as pd
import pytest

from src.connectivity.pairwise.config import ConnectivityConfig
from src.connectivity.pairwise.oaec import (
    compute_observed_lagged_oaec,
    compute_oaec,
    directional_orthogonalized_correlation_z,
    gaussian_analytic_filterbank,
    lagged_directional_orthogonalized_correlation_z,
)
from src.connectivity.pairwise.permutation import generate_derangements
from src.connectivity.pairwise.tf_dwpli import (
    _band_frequency_mask,
    _band_mean_dwpli_from_tf,
    compute_tf_dwpli,
    debiased_wpli_from_imag,
    morlet_coefficients,
)
from src.connectivity.pairwise.config import wpli_frequencies
from src.connectivity.pairwise.xcorr import (
    compute_xcorr,
    lagged_cross_trial_pearson_z,
    residualize_evoked_mean,
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


def test_xcorr_recovers_both_lag_directions():
    rng = np.random.default_rng(101)
    source = rng.normal(size=(28, 100))
    source = np.asarray(
        [np.convolve(trial, np.ones(7) / 7.0, mode="same") for trial in source]
    )
    lags = np.arange(-10, 11)
    identity = np.arange(len(source))
    for shift, expected in ((6, -6), (-5, 5)):
        target = np.roll(source, shift, axis=1)
        values = lagged_cross_trial_pearson_z(source, target, lags)
        observed = np.nanmean(values[:, identity, identity], axis=1)
        assert lags[np.nanargmax(observed)] == expected


def test_evoked_residualization_removes_common_mean_and_keeps_trial_lag():
    rng = np.random.default_rng(102)
    n_trials, n_time = 48, 140
    evoked = 4.0 * np.exp(-0.5 * ((np.arange(n_time) - 65) / 13.0) ** 2)
    innovations = rng.normal(size=(n_trials, n_time))
    trial_specific = np.asarray(
        [np.convolve(trial, np.ones(9) / 9.0, mode="same") for trial in innovations]
    )
    source = evoked[None, :] + trial_specific
    target = evoked[None, :] + np.roll(trial_specific, 7, axis=1)
    residual = residualize_evoked_mean(
        np.stack((source, target), axis=1).astype(np.float32)
    )
    np.testing.assert_allclose(residual.mean(axis=0), 0.0, atol=2e-7)

    lags = np.arange(-12, 13)
    identity = np.arange(n_trials)
    values = lagged_cross_trial_pearson_z(residual[:, 0], residual[:, 1], lags)
    observed = np.nanmean(values[:, identity, identity], axis=1)
    assert lags[np.nanargmax(observed)] == -7
    assert np.tanh(np.nanmax(observed)) > 0.8


def test_evoked_residualization_eliminates_common_waveform_correlation():
    rng = np.random.default_rng(104)
    n_trials, n_time = 80, 160
    evoked = 5.0 * np.sin(np.linspace(0, 3 * np.pi, n_time))
    source = evoked + rng.normal(scale=0.5, size=(n_trials, n_time))
    target = evoked + rng.normal(scale=0.5, size=(n_trials, n_time))
    data = np.stack((source, target), axis=1).astype(np.float32)
    residual = residualize_evoked_mean(data)
    identity = np.arange(n_trials)
    before = lagged_cross_trial_pearson_z(data[:, 0], data[:, 1], [0])
    after = lagged_cross_trial_pearson_z(
        residual[:, 0], residual[:, 1], [0]
    )
    before_r = np.tanh(np.nanmean(before[0, identity, identity]))
    after_r = np.tanh(np.nanmean(after[0, identity, identity]))
    assert before_r > 0.95
    assert abs(after_r) < 0.1


def test_xcorr_residualization_is_opt_in_and_default_is_unchanged(tmp_path):
    rng = np.random.default_rng(103)
    data = rng.normal(size=(24, 2, 80)).astype(np.float32)
    permutations = generate_derangements(20, 24, 9)
    config = ConnectivityConfig(n_perm=20, max_lag_s=0.1)
    default = compute_xcorr(
        data, 100.0, _one_pair(), permutations, config,
        scratch_dir=tmp_path / "default",
    )
    explicit = compute_xcorr(
        data, 100.0, _one_pair(), permutations, config,
        scratch_dir=tmp_path / "explicit", residualize_evoked=False,
    )
    pd.testing.assert_frame_equal(default.pair_table, explicit.pair_table)
    np.testing.assert_array_equal(
        default.detail["observed_fisher_z"],
        explicit.detail["observed_fisher_z"],
    )
    residual = compute_xcorr(
        data, 100.0, _one_pair(), permutations, config,
        scratch_dir=tmp_path / "residual", residualize_evoked=True,
    )
    assert residual.metric == "xcorr_resid"
    assert residual.pair_table.loc[0, "metric"] == "xcorr_resid"
    assert residual.detail.attrs["residualization"] == (
        "across_trial_mean_per_channel_time"
    )


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

    lags = np.arange(-5, 6)
    lagged, n_used = lagged_directional_orthogonalized_correlation_z(
        source, target, lags
    )
    lag0 = int(np.where(lags == 0)[0][0])
    np.testing.assert_allclose(
        lagged[lag0],
        np.nanmean(directional, axis=0),
        atol=1e-6,
    )
    assert n_used[lag0] == n_time
    assert n_used[0] == n_time - 5


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


def test_candidate_lagged_oaec_is_observed_only_and_signed(monkeypatch):
    import src.connectivity.pairwise.oaec as oaec_module

    rng = np.random.default_rng(22)
    n_trials, n_time = 32, 128
    phase = np.linspace(0.0, 12.0 * np.pi, n_time)
    innovation = rng.normal(size=(n_trials, n_time))
    amplitude = np.exp(
        np.asarray([
            np.convolve(trial, np.ones(9) / 9.0, mode="same")
            for trial in innovation
        ])
    )
    source = amplitude * np.exp(1j * phase[None, :])
    target = np.roll(amplitude, 5, axis=1) * np.exp(
        1j * (phase[None, :] + np.pi / 3.0)
    )
    coefficients = np.stack((source, target), axis=1)[:, :, None].astype(
        np.complex64
    )

    monkeypatch.setattr(
        oaec_module, "hga_filterbank_centers", lambda: np.asarray([100.0])
    )
    monkeypatch.setattr(
        oaec_module, "gaussian_analytic_filterbank",
        lambda *args, **kwargs: coefficients,
    )
    raw = rng.normal(size=(n_trials, 2, 256)).astype(np.float32)
    times = np.linspace(-1.0, 1.0, 256, endpoint=False)
    result = compute_observed_lagged_oaec(
        raw, times, 128.0, "Response", _one_pair(),
        ConnectivityConfig(max_lag_s=0.1, oaec_sfreq=128.0),
    )
    assert result.metric == "lagged_oaec_candidate"
    assert result.pair_table.loc[0, "peak_lag_s"] == pytest.approx(-5 / 128)
    assert result.detail.attrs["inference"] == (
        "observed_only_no_additional_permutation_test"
    )


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


def test_band_mean_dwpli_matches_average_of_tf_bins():
    rng = np.random.default_rng(0)
    n_trials, n_freq, n_time = 24, 3, 4
    phase = np.linspace(0.0, 6.0 * np.pi, n_time)
    # Trial-wise phase offsets: observed pairing is consistent; shuffle breaks it
    source_offset = rng.uniform(0.0, 2.0 * np.pi, size=n_trials)
    target_offset = source_offset + np.pi / 3.0  # fixed lag when correctly paired
    source = np.stack(
        [
            np.repeat(np.exp(1j * (phase + off))[None, :], n_freq, axis=0)
            for off in source_offset
        ],
        axis=0,
    )
    target = np.stack(
        [
            np.repeat(np.exp(1j * (phase + off))[None, :], n_freq, axis=0)
            for off in target_offset
        ],
        axis=0,
    )
    cross = source * np.conj(target)
    observed_tf, _ = debiased_wpli_from_imag(np.imag(cross), axis=0)
    permutations = np.stack(
        [rng.permutation(n_trials) for _ in range(40)], axis=0
    )
    frequency_mask = np.ones(n_freq, dtype=bool)
    observed_band, null = _band_mean_dwpli_from_tf(
        observed_tf,
        source,
        target,
        permutations,
        frequency_mask,
    )
    assert observed_band == pytest.approx(float(np.nanmean(observed_tf)))
    assert null.shape == (40,)
    assert np.all(np.isfinite(null))
    # Target-trial shuffle must produce a non-degenerate null
    assert float(np.std(null, ddof=1)) > 1e-3
    # Observed consistent lag should exceed typical shuffled null
    assert observed_band > float(np.mean(null))


def test_band_mean_dwpli_null_uses_float32_temporaries(monkeypatch):
    """Null cross-spectrum path must stay float32 (peak RAM)."""
    from src.connectivity.pairwise import tf_dwpli as mod

    seen: dict[str, object] = {}
    rng = np.random.default_rng(1)
    n_trials, n_freq, n_time = 16, 2, 4
    source = (
        rng.normal(size=(n_trials, n_freq, n_time))
        + 1j * rng.normal(size=(n_trials, n_freq, n_time))
    ).astype(np.complex64)
    target = (
        rng.normal(size=(n_trials, n_freq, n_time))
        + 1j * rng.normal(size=(n_trials, n_freq, n_time))
    ).astype(np.complex64)
    observed_tf = np.zeros((n_freq, n_time), dtype=np.float32)
    permutations = generate_derangements(10, n_trials, 3)
    frequency_mask = np.ones(n_freq, dtype=bool)

    real_fn = mod.debiased_wpli_from_imag

    def _tracking(imaginary_cross_spectrum, *, axis=0):
        arr = np.asarray(imaginary_cross_spectrum)
        seen["dtype"] = arr.dtype
        return real_fn(arr, axis=axis)

    monkeypatch.setattr(mod, "debiased_wpli_from_imag", _tracking)
    _band_mean_dwpli_from_tf(
        observed_tf, source, target, permutations, frequency_mask
    )
    assert seen["dtype"] == np.float32


def test_assign_band_inference_uses_local_offset():
    from src.connectivity.pairwise.tf_dwpli import _assign_band_inference

    n_pairs, n_bands = 2, 4
    output = {
        key: np.full(n_pairs * n_bands, np.nan)
        for key in (
            "null_mean",
            "null_std",
            "p_uncorrected",
            "q_fdr",
            "p_fwer_maxstat",
            "sig_fdr",
            "sig_fwer",
        )
    }
    # Flattened secondary family order: pair0 theta/alpha/beta, pair1 ...
    inference = {
        key: np.asarray([10.0, 20.0, 30.0, 11.0, 21.0, 31.0])
        for key in output
    }
    # band indices: theta=0, alpha=1, beta=2 (broadband=3 unused here)
    for local_offset, band_index in enumerate([0, 1, 2]):
        _assign_band_inference(
            output,
            band_index,
            n_pairs,
            n_bands,
            inference,
            band_stride=3,
            local_offset=local_offset,
        )
    assert output["null_mean"][0] == 10.0  # pair0 theta
    assert output["null_mean"][1] == 20.0  # pair0 alpha
    assert output["null_mean"][2] == 30.0  # pair0 beta
    assert output["null_mean"][4] == 11.0  # pair1 theta
    assert output["null_mean"][5] == 21.0  # pair1 alpha
    assert output["null_mean"][6] == 31.0  # pair1 beta


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


def _synthetic_wpli_pair_frame(n_pairs: int) -> pd.DataFrame:
    rows = []
    for pair_index in range(n_pairs):
        source_index = pair_index % 3
        target_index = (pair_index + 1) % 4
        rows.append(
            {
                "pair_id": f"S{source_index}-T{target_index}",
                "source": f"S{source_index}",
                "target": f"T{target_index}",
                "source_index": source_index,
                "target_index": target_index,
            }
        )
    return pd.DataFrame(rows)


def test_wpli_pair_parallel_matches_serial(monkeypatch):
    """Pair-level joblib threads must match the serial pair loop."""

    rng = np.random.default_rng(42)
    n_trials, n_channels, n_time = 32, 4, 16
    freq_step = 2.0
    n_freq = len(wpli_frequencies(step=freq_step))
    phase = np.linspace(0.0, 4.0 * np.pi, n_time)
    coefficients = np.empty(
        (n_trials, n_channels, n_freq, n_time), dtype=np.complex64
    )
    for trial in range(n_trials):
        for channel in range(n_channels):
            offset = rng.uniform(0.0, 2.0 * np.pi)
            coefficients[trial, channel] = np.exp(
                1j * (phase + offset + 0.2 * channel)
            )[None, :]

    def _fake_morlet(data, sfreq, freqs, n_cycles, *, n_jobs=1):
        del data, sfreq, freqs, n_cycles, n_jobs
        return coefficients

    monkeypatch.setattr(
        "src.connectivity.pairwise.tf_dwpli.morlet_coefficients",
        _fake_morlet,
    )

    pair_frame = _synthetic_wpli_pair_frame(n_pairs=6)
    permutations = generate_derangements(20, n_trials, 99)
    raw_data = rng.normal(size=(n_trials, n_channels, n_time)).astype(
        np.float32
    )
    raw_times = np.arange(n_time, dtype=np.float64) / 256.0

    serial = compute_tf_dwpli(
        raw_data,
        raw_times,
        256.0,
        "Stimulus",
        pair_frame,
        permutations,
        ConnectivityConfig(n_perm=20, n_jobs=1, wpli_freq_step=freq_step),
    )
    parallel = compute_tf_dwpli(
        raw_data,
        raw_times,
        256.0,
        "Stimulus",
        pair_frame,
        permutations,
        ConnectivityConfig(n_perm=20, n_jobs=4, wpli_freq_step=freq_step),
    )

    for key in (
        "observed_band_wpli2_debiased",
        "null_mean",
        "null_std",
        "valid_tf_bin",
    ):
        assert np.allclose(
            serial.detail[key].values,
            parallel.detail[key].values,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )

    compare_cols = [
        "stat",
        "null_mean",
        "null_std",
        "p_uncorrected",
        "q_fdr",
        "p_fwer_maxstat",
        "sig_fdr",
        "sig_fwer",
    ]
    for col in compare_cols:
        assert np.allclose(
            serial.pair_table[col].to_numpy(dtype=float),
            parallel.pair_table[col].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        ), f"mismatch in pair_table[{col!r}]"
