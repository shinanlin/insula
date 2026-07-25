"""Band-specific time-frequency debiased squared wPLI."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import xarray as xr

from .config import (
    ConnectivityConfig,
    PHASE_WINDOWS,
    WPLI_BANDS,
    phase_time_mask,
    wpli_frequencies,
    wpli_n_cycles,
)
from .permutation import scalar_permutation_inference
from .result import MetricResult


def debiased_wpli_from_imag(
    imaginary_cross_spectrum: np.ndarray,
    *,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute debiased squared wPLI and its denominator.

    The finite-sample estimator is intentionally not clipped to ``[0, 1]``.
    """

    values = np.asarray(imaginary_cross_spectrum, dtype=float)
    sum_imag = np.sum(values, axis=axis)
    sum_abs = np.sum(np.abs(values), axis=axis)
    sum_sq = np.sum(values**2, axis=axis)
    numerator = sum_imag**2 - sum_sq
    denominator = sum_abs**2 - sum_sq
    result = np.divide(
        numerator,
        denominator,
        out=np.full(np.shape(numerator), np.nan, dtype=float),
        # The denominator has signal-amplitude-to-the-fourth units. Raw MNE
        # voltage is in volts, so a valid denominator can be far below the
        # dimensionless machine epsilon.
        where=denominator > 0.0,
    )
    return result, denominator


def _resample_raw(
    data: np.ndarray,
    times: np.ndarray,
    sfreq: float,
    target_sfreq: float,
) -> tuple[np.ndarray, np.ndarray]:
    if np.isclose(sfreq, target_sfreq):
        return np.asarray(data, dtype=np.float32), np.asarray(times)
    ratio = Fraction(target_sfreq / sfreq).limit_denominator(2048)
    resampled = resample_poly(
        np.asarray(data, dtype=np.float32),
        up=ratio.numerator,
        down=ratio.denominator,
        axis=-1,
    ).astype(np.float32, copy=False)
    new_times = float(times[0]) + np.arange(resampled.shape[-1]) / target_sfreq
    return resampled, new_times


def morlet_coefficients(
    data: np.ndarray,
    sfreq: float,
    freqs: np.ndarray,
    n_cycles: np.ndarray,
    *,
    n_jobs: int = 1,
) -> np.ndarray:
    """Complex Morlet coefficients with no envelope/HGA input."""

    from mne.time_frequency import tfr_array_morlet

    coefficients = tfr_array_morlet(
        np.asarray(data, dtype=np.float32),
        sfreq=float(sfreq),
        freqs=np.asarray(freqs, dtype=float),
        n_cycles=np.asarray(n_cycles, dtype=float),
        output="complex",
        zero_mean=True,
        n_jobs=n_jobs,
        verbose=False,
    )
    return coefficients.astype(np.complex64, copy=False)


def _band_frequency_mask(
    freqs: np.ndarray, band_name: str
) -> np.ndarray:
    lower, upper = WPLI_BANDS[band_name]
    if band_name in {"beta", "broadband"}:
        return (freqs >= lower) & (freqs <= upper)
    return (freqs >= lower) & (freqs < upper)


def compute_tf_dwpli(
    raw_data: np.ndarray,
    raw_times: np.ndarray,
    sfreq: float,
    phase: str,
    pair_frame: pd.DataFrame,
    permutations: np.ndarray,
    config: ConnectivityConfig,
) -> MetricResult:
    """Compute exploratory TF-dwPLI and band-level permutation inference."""

    pair_channel_indices = sorted(
        set(pair_frame["source_index"].astype(int))
        | set(pair_frame["target_index"].astype(int))
    )
    compact_index = {
        original: compact
        for compact, original in enumerate(pair_channel_indices)
    }
    resampled, resampled_times = _resample_raw(
        np.asarray(raw_data)[:, pair_channel_indices, :],
        raw_times,
        sfreq,
        config.wpli_sfreq,
    )
    freqs = wpli_frequencies(config.wpli_freq_step)
    n_cycles = wpli_n_cycles(freqs)
    coefficients = morlet_coefficients(
        resampled,
        config.wpli_sfreq,
        freqs,
        n_cycles,
        n_jobs=config.n_jobs,
    )
    time_mask = phase_time_mask(resampled_times, phase)
    coefficients = coefficients[..., time_mask]
    phase_times = resampled_times[time_mask]

    n_trials = coefficients.shape[0]
    n_pairs = len(pair_frame)
    n_perm = permutations.shape[0]
    band_names = list(WPLI_BANDS)
    n_bands = len(band_names)
    identity = np.arange(n_trials)
    observed_tf = np.full(
        (n_pairs, freqs.size, phase_times.size),
        np.nan,
        dtype=np.float32,
    )
    valid_tf = np.zeros_like(observed_tf, dtype=bool)
    observed_band = np.full((n_pairs, n_bands), np.nan, dtype=np.float32)
    null = np.full(
        (n_perm, n_pairs, n_bands), np.nan, dtype=np.float32
    )
    source_power = np.full((n_pairs, n_bands), np.nan, dtype=np.float32)
    target_power = np.full((n_pairs, n_bands), np.nan, dtype=np.float32)
    valid_fraction = np.zeros((n_pairs, n_bands), dtype=np.float32)

    for pair_index, pair in pair_frame.iterrows():
        source_index = compact_index[int(pair["source_index"])]
        target_index = compact_index[int(pair["target_index"])]
        source = coefficients[:, source_index]
        target = coefficients[:, target_index]
        cross = source * np.conj(target)
        tf_value, tf_denominator = debiased_wpli_from_imag(
            np.imag(cross), axis=0
        )
        observed_tf[pair_index] = tf_value.astype(np.float32)
        valid_tf[pair_index] = np.isfinite(tf_value) & (
            tf_denominator > 0.0
        )

        for band_index, band_name in enumerate(band_names):
            frequency_mask = _band_frequency_mask(freqs, band_name)
            source_features = source[:, frequency_mask, :].reshape(
                n_trials, -1
            )
            target_features = target[:, frequency_mask, :].reshape(
                n_trials, -1
            )
            cross_trial_matrix = (
                source_features @ np.conj(target_features).T
            ) / source_features.shape[1]
            imaginary_matrix = np.imag(cross_trial_matrix)
            observed_vector = imaginary_matrix[identity, identity]
            observed_band[pair_index, band_index] = (
                debiased_wpli_from_imag(observed_vector, axis=0)[0]
            )
            selected = imaginary_matrix[identity[None, :], permutations]
            null[:, pair_index, band_index] = (
                debiased_wpli_from_imag(selected, axis=1)[0]
            ).astype(np.float32)
            source_power[pair_index, band_index] = float(
                np.mean(np.abs(source_features) ** 2)
            )
            target_power[pair_index, band_index] = float(
                np.mean(np.abs(target_features) ** 2)
            )
            valid_fraction[pair_index, band_index] = float(
                np.mean(valid_tf[pair_index, frequency_mask])
            )

    flat_observed = observed_band.reshape(-1)
    flat_null = null.reshape(n_perm, -1)
    inference = scalar_permutation_inference(
        flat_observed, flat_null, tail="greater", alpha=config.alpha
    )
    repeated = pair_frame.loc[
        pair_frame.index.repeat(n_bands)
    ].reset_index(drop=True)
    repeated["band"] = np.tile(np.asarray(band_names), n_pairs)
    repeated["metric"] = "wpli2_debiased_tf"
    repeated["stat"] = flat_observed
    repeated["band_power_source"] = source_power.reshape(-1)
    repeated["band_power_target"] = target_power.reshape(-1)
    repeated["valid_tf_fraction"] = valid_fraction.reshape(-1)
    duration = PHASE_WINDOWS[phase][1] - PHASE_WINDOWS[phase][0]
    repeated["wavelet_support_s"] = np.tile(
        [
            float(
                np.max(
                    5.0
                    * n_cycles[_band_frequency_mask(freqs, name)]
                    / (
                        np.pi
                        * freqs[_band_frequency_mask(freqs, name)]
                    )
                )
            )
            for name in band_names
        ],
        n_pairs,
    )
    repeated["short_window_flag"] = (
        duration < repeated["wavelet_support_s"].to_numpy()
    )
    repeated["exploratory_flag"] = np.tile(
        [
            name == "theta" and duration <= 0.5 + 1e-9
            for name in band_names
        ],
        n_pairs,
    )
    for key in (
        "null_mean",
        "null_std",
        "p_uncorrected",
        "q_fdr",
        "p_fwer_maxstat",
        "sig_fdr",
        "sig_fwer",
    ):
        repeated[key] = inference[key]
    repeated["qc_pass"] = (
        np.isfinite(flat_observed)
        & np.isfinite(inference["null_std"])
        & (valid_fraction.reshape(-1) >= 0.8)
    )

    variables: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "observed_tf_wpli2_debiased": (
            ("pair", "frequency", "time"),
            observed_tf,
        ),
        "valid_tf_bin": (("pair", "frequency", "time"), valid_tf),
        "observed_band_wpli2_debiased": (
            ("pair", "band"),
            observed_band,
        ),
        "null_mean": (
            ("pair", "band"),
            inference["null_mean"].reshape(n_pairs, n_bands).astype(np.float32),
        ),
        "null_std": (
            ("pair", "band"),
            inference["null_std"].reshape(n_pairs, n_bands).astype(np.float32),
        ),
    }
    if config.save_full_null:
        variables["null_band_stat"] = (
            ("permutation", "pair", "band"),
            null,
        )
    detail = xr.Dataset(
        data_vars=variables,
        coords={
            "pair": np.arange(n_pairs, dtype=np.int32),
            "frequency": freqs.astype(np.float32),
            "time": phase_times.astype(np.float32),
            "band": np.asarray(band_names, dtype=str),
            "pair_id": ("pair", pair_frame["pair_id"].astype(str).to_numpy()),
            "source": ("pair", pair_frame["source"].astype(str).to_numpy()),
            "target": ("pair", pair_frame["target"].astype(str).to_numpy()),
        },
        attrs={
            "metric": "wpli2_debiased_tf",
            "input": "raw_voltage_to_band_specific_complex_morlet",
            "band_statistic": "dwpli_of_trialwise_band_time_averaged_cross_spectrum",
            "wavelet_support": "full_10_sigma_support_5*n_cycles/(pi*f)",
            "tail": "greater",
            "n_perm": int(n_perm),
        },
    )
    return MetricResult(
        metric="wpli",
        pair_table=repeated,
        detail=detail,
        runtime_metadata={
            "n_pairs": n_pairs,
            "n_frequencies": int(freqs.size),
            "n_trials": n_trials,
            "n_times": int(phase_times.size),
        },
    )
