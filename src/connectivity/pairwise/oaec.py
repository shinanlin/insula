"""HGA filterbank orthogonalized amplitude-envelope correlation."""

from __future__ import annotations

from fractions import Fraction
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.fft import fft, ifft
from scipy.signal import resample_poly
import xarray as xr

from .config import ConnectivityConfig, phase_time_mask
from .permutation import scalar_permutation_inference
from .result import MetricResult


def hga_filterbank_centers(
    passband: tuple[float, float] = (70.0, 200.0),
    spacing: float = 1.0 / 7.0,
    f0: float = 0.018,
) -> np.ndarray:
    """Centers used by the lab's Gaussian HGA filterbank."""

    minimum, maximum = passband
    if minimum >= maximum:
        raise ValueError("passband lower edge must be below upper edge")
    centers = [f0]
    sigma = 0.39 * np.sqrt(f0)
    while np.log2(centers[-1] / f0) < np.log2(maximum / f0):
        if centers[-1] < 4.0:
            centers.append(centers[-1] + sigma)
        else:
            octave = np.log2(centers[-1] / f0) + spacing
            centers.append(f0 * (2.0**octave))
        sigma = 0.39 * np.sqrt(centers[-1])
    values = np.asarray(centers, dtype=np.float32)
    return values[(values >= minimum) & (values <= maximum)]


def gaussian_analytic_filterbank(
    data: np.ndarray,
    sfreq: float,
    centers: Iterable[float],
    *,
    time_mask: np.ndarray | None = None,
    target_sfreq: float | None = None,
) -> np.ndarray:
    """Return complex Gaussian-filterbank analytic signals.

    Filtering is performed over the full input time axis. ``time_mask`` is
    applied only after the inverse FFT, which avoids phase-window edge effects.
    """

    values = np.asarray(data, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("data must have shape (trial, channel, time)")
    centers = np.asarray(list(centers), dtype=np.float32)
    n_trials, n_channels, n_time = values.shape
    positive_freqs = (
        np.arange(0, n_time // 2 + 1, dtype=np.float32)
        * float(sfreq)
        / n_time
    )
    sigma = np.power(
        10.0, np.log10(0.39) + 0.5 * np.log10(centers)
    ) * np.sqrt(2.0)
    kernel_positive = np.exp(
        -0.5
        * (
            (positive_freqs[:, None] - centers[None, :])
            / sigma[None, :]
        )
        ** 2
    ).astype(np.float32)
    kernel = np.zeros((n_time, centers.size), dtype=np.complex64)
    upper = (n_time + 1) // 2
    kernel[1:upper] = 2.0 * kernel_positive[1:upper]
    if n_time % 2 == 0:
        kernel[n_time // 2] = kernel_positive[n_time // 2]

    if target_sfreq is not None and target_sfreq <= 0:
        raise ValueError("target_sfreq must be positive")
    if target_sfreq is None:
        selected = (
            np.arange(n_time)
            if time_mask is None
            else np.flatnonzero(np.asarray(time_mask, dtype=bool))
        )
        output_time = selected.size
    else:
        ratio = Fraction(target_sfreq / sfreq).limit_denominator(1024)
        n_resampled = int(np.ceil(n_time * ratio.numerator / ratio.denominator))
        if time_mask is None:
            selected = np.arange(n_resampled)
        else:
            indices = np.flatnonzero(np.asarray(time_mask, dtype=bool))
            if indices.size == 0:
                raise ValueError("time_mask selects no samples")
            start_s = indices[0] / float(sfreq)
            stop_s = (indices[-1] + 1) / float(sfreq)
            resampled_relative_times = (
                np.arange(n_resampled) / float(target_sfreq)
            )
            selected = np.flatnonzero(
                (resampled_relative_times >= start_s)
                & (resampled_relative_times < stop_s)
            )
        output_time = selected.size
    output = np.empty(
        (n_trials, n_channels, centers.size, output_time),
        dtype=np.complex64,
    )
    for trial in range(n_trials):
        spectrum = fft(values[trial], axis=-1).astype(np.complex64)
        filtered = ifft(
            spectrum[:, :, None] * kernel[None, :, :],
            axis=1,
        ).astype(np.complex64)
        filtered = filtered.transpose(0, 2, 1)
        if target_sfreq is not None:
            # Shift every narrowband analytic signal to complex baseband
            # before resampling. Directly downsampling a 70--200 Hz carrier
            # to 128 Hz would (correctly) anti-alias away the signal. The same
            # unit-magnitude rotation is applied to both channels and thus
            # leaves their relative phase and Hipp orthogonalization invariant.
            relative_times = np.arange(n_time, dtype=float) / float(sfreq)
            carrier = np.exp(
                -2j
                * np.pi
                * centers[:, None].astype(float)
                * relative_times[None, :]
            ).astype(np.complex64)
            filtered = filtered * carrier[None, :, :]
            filtered = resample_poly(
                filtered,
                up=ratio.numerator,
                down=ratio.denominator,
                axis=-1,
            )
        output[trial] = filtered[..., selected]
    return output


def _resample_envelope(
    envelope: np.ndarray, sfreq: float, target_sfreq: float
) -> np.ndarray:
    if np.isclose(sfreq, target_sfreq):
        return envelope
    ratio = Fraction(target_sfreq / sfreq).limit_denominator(1024)
    return resample_poly(
        envelope,
        up=ratio.numerator,
        down=ratio.denominator,
        axis=-1,
    )


def cross_trial_correlation_z(
    first: np.ndarray, second: np.ndarray
) -> np.ndarray:
    """All source-trial × target-trial correlations in Fisher-z units."""

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("envelopes must have shape (trial, time)")
    if first.shape != second.shape:
        raise ValueError("envelope arrays must have matching shape")
    first_centered = first - first.mean(axis=1, keepdims=True)
    second_centered = second - second.mean(axis=1, keepdims=True)
    denominator = (
        np.linalg.norm(first_centered, axis=1)[:, None]
        * np.linalg.norm(second_centered, axis=1)[None, :]
    )
    correlation = np.divide(
        first_centered @ second_centered.T,
        denominator,
        out=np.full(denominator.shape, np.nan, dtype=float),
        where=denominator > np.finfo(float).eps,
    )
    return np.arctanh(
        np.clip(correlation, -1.0 + 1e-7, 1.0 - 1e-7)
    )


def orthogonalized_log_power_envelopes(
    source: np.ndarray,
    target: np.ndarray,
    *,
    sfreq: float,
    target_sfreq: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Matched-trial bidirectional Hipp orthogonalized log-power envelopes."""

    source = np.asarray(source)
    target = np.asarray(target)
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("source and target must have shape (trial, time)")
    epsilon = np.finfo(np.float32).eps
    source_abs = np.maximum(np.abs(source), epsilon)
    target_abs = np.maximum(np.abs(target), epsilon)
    target_orth_source = np.abs(
        np.imag(target * np.conj(source) / source_abs)
    )
    source_orth_target = np.abs(
        np.imag(source * np.conj(target) / target_abs)
    )
    source_power = np.log(source_abs**2 + epsilon)
    target_power = np.log(target_abs**2 + epsilon)
    target_orth_power = np.log(target_orth_source**2 + epsilon)
    source_orth_power = np.log(source_orth_target**2 + epsilon)
    return tuple(
        _resample_envelope(array, sfreq, target_sfreq).astype(
            np.float32, copy=False
        )
        for array in (
            source_power,
            target_orth_power,
            target_power,
            source_orth_power,
        )
    )


def _hipp_log_power_envelopes(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Contemporaneous Hipp log-power envelopes; time is the last axis."""

    epsilon = np.finfo(np.float32).eps
    source_abs = np.maximum(np.abs(source), epsilon)
    target_abs = np.maximum(np.abs(target), epsilon)
    target_orth_source = np.abs(
        np.imag(target * np.conj(source) / source_abs)
    )
    source_orth_target = np.abs(
        np.imag(source * np.conj(target) / target_abs)
    )
    return (
        np.log(source_abs**2 + epsilon),
        np.log(target_orth_source**2 + epsilon),
        np.log(target_abs**2 + epsilon),
        np.log(source_orth_target**2 + epsilon),
    )


def _pearson_z_time_last(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_centered = first - np.mean(first, axis=-1, keepdims=True)
    second_centered = second - np.mean(second, axis=-1, keepdims=True)
    denominator = np.linalg.norm(first_centered, axis=-1) * np.linalg.norm(
        second_centered, axis=-1
    )
    numerator = np.sum(first_centered * second_centered, axis=-1)
    correlation = np.divide(
        numerator,
        denominator,
        out=np.full(denominator.shape, np.nan, dtype=float),
        where=denominator > np.finfo(float).eps,
    )
    return np.arctanh(np.clip(correlation, -1.0 + 1e-7, 1.0 - 1e-7))


def directional_orthogonalized_correlation_z(
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Hipp OAEC for broadcastable complex arrays with time last.

    Orthogonalization and envelope extraction happen inside this function, so
    shuffled target trials are re-orthogonalized against their surrogate
    source trial instead of shuffling a previously corrected envelope.
    """

    source = np.asarray(source)
    target = np.asarray(target)
    if source.shape != target.shape:
        try:
            source, target = np.broadcast_arrays(source, target)
        except ValueError as error:
            raise ValueError("source and target are not broadcastable") from error
    envelopes = _hipp_log_power_envelopes(source, target)
    return np.stack(
        (
            _pearson_z_time_last(envelopes[0], envelopes[1]),
            _pearson_z_time_last(envelopes[2], envelopes[3]),
        ),
        axis=-1,
    )


def lagged_directional_orthogonalized_correlation_z(
    source: np.ndarray,
    target: np.ndarray,
    lags: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Trial-mean OAEC vs lag; orthogonalize contemporaneously, then lag envelopes.

    ``source`` and ``target`` are complex arrays with shape ``(trial, time)``.
    The returned array has shape ``(n_lags, 2)`` (source→target, target→source).
    Negative lag means the source envelope leads, matching HGA xcorr.
    """

    source = np.asarray(source)
    target = np.asarray(target)
    if source.ndim != 2 or source.shape != target.shape:
        raise ValueError("source and target must have shape (trial, time)")
    envelopes = _hipp_log_power_envelopes(source, target)
    lag_values = np.asarray(list(lags), dtype=int)
    n_time = source.shape[-1]
    output = np.full((lag_values.size, 2), np.nan, dtype=np.float64)
    n_used = np.zeros(lag_values.size, dtype=int)
    for lag_index, lag in enumerate(lag_values):
        lag = int(lag)
        if abs(lag) >= n_time - 2:
            continue
        if lag < 0:
            first_slice, second_slice = slice(None, n_time + lag), slice(-lag, None)
        elif lag > 0:
            first_slice, second_slice = slice(lag, None), slice(None, n_time - lag)
        else:
            first_slice = second_slice = slice(None)
        n_used[lag_index] = envelopes[0][:, first_slice].shape[-1]
        output[lag_index, 0] = np.nanmean(
            _pearson_z_time_last(
                envelopes[0][:, first_slice], envelopes[1][:, second_slice]
            )
        )
        output[lag_index, 1] = np.nanmean(
            _pearson_z_time_last(
                envelopes[2][:, first_slice], envelopes[3][:, second_slice]
            )
        )
    return output, n_used


def compute_observed_lagged_oaec(
    raw_data: np.ndarray,
    raw_times: np.ndarray,
    sfreq: float,
    phase: str,
    pair_frame: pd.DataFrame,
    config: ConnectivityConfig,
) -> MetricResult:
    """Compute observed bidirectional lagged OAEC for preselected pairs.

    This estimator deliberately performs no second permutation test.  It is a
    descriptive consistency check for pairs selected independently by both
    original and evoked-mean-residualized xcorr.
    """

    if pair_frame.empty:
        raise ValueError("lagged OAEC requires at least one candidate pair")
    phase_mask = phase_time_mask(raw_times, phase)
    centers = hga_filterbank_centers()
    pair_channel_indices = sorted(
        set(pair_frame["source_index"].astype(int))
        | set(pair_frame["target_index"].astype(int))
    )
    compact_index = {
        original: compact
        for compact, original in enumerate(pair_channel_indices)
    }
    coefficients = gaussian_analytic_filterbank(
        np.asarray(raw_data)[:, pair_channel_indices, :],
        sfreq,
        centers,
        time_mask=phase_mask,
        target_sfreq=config.oaec_sfreq,
    )
    max_lag = int(round(config.max_lag_s * config.oaec_sfreq))
    if max_lag * 2 > coefficients.shape[-1]:
        raise ValueError("max_lag_s must not exceed half of the OAEC window")
    lag_samples = np.arange(-max_lag, max_lag + 1, dtype=int)
    lag_times = lag_samples.astype(float) / float(config.oaec_sfreq)
    curves = np.full(
        (len(pair_frame), len(lag_samples), 2), np.nan, dtype=np.float32
    )
    overlap = np.zeros(len(lag_samples), dtype=np.int32)
    for pair_position, (_, pair) in enumerate(pair_frame.iterrows()):
        source_index = compact_index[int(pair["source_index"])]
        target_index = compact_index[int(pair["target_index"])]
        frequency_curves = np.full(
            (len(centers), len(lag_samples), 2), np.nan, dtype=np.float32
        )
        for frequency_index in range(len(centers)):
            frequency_curves[frequency_index], overlap = (
                lagged_directional_orthogonalized_correlation_z(
                    coefficients[:, source_index, frequency_index],
                    coefficients[:, target_index, frequency_index],
                    lag_samples,
                )
            )
        curves[pair_position] = np.nanmean(frequency_curves, axis=0)

    # Source is always the Insula seed in this analysis.  Use the
    # source-to-target orthogonalization for the signed-lag summary; averaging
    # the two directional curves can create equal peaks at +/-lag by symmetry.
    source_curve = curves[:, :, 0]
    mean_curve = np.nanmean(curves, axis=2)
    finite = np.where(np.isfinite(source_curve), np.abs(source_curve), -np.inf)
    peak_index = np.argmax(finite, axis=1)
    no_peak = np.all(~np.isfinite(mean_curve), axis=1)
    output = pair_frame.copy().reset_index(drop=True)
    output["metric"] = "lagged_oaec_candidate"
    output["stat"] = np.abs(source_curve[np.arange(len(output)), peak_index])
    output["peak_lag_s"] = lag_times[peak_index]
    output["peak_fisher_z"] = source_curve[
        np.arange(len(output)), peak_index
    ]
    bidirectional_peak_index = np.argmax(np.abs(mean_curve), axis=1)
    output["bidirectional_mean_peak_lag_s"] = lag_times[
        bidirectional_peak_index
    ]
    output["source_to_target_peak_lag_s"] = lag_times[
        np.argmax(np.abs(curves[:, :, 0]), axis=1)
    ]
    output["target_to_source_peak_lag_s"] = lag_times[
        np.argmax(np.abs(curves[:, :, 1]), axis=1)
    ]
    output.loc[
        no_peak,
        [
            "stat", "peak_lag_s", "peak_fisher_z",
            "source_to_target_peak_lag_s", "target_to_source_peak_lag_s",
        ],
    ] = np.nan
    output["qc_pass"] = ~no_peak
    output["lag_convention"] = "negative_lag_source_leads"
    detail = xr.Dataset(
        data_vars={
            "observed_fisher_z": (
                ("pair", "lag", "direction"), curves
            ),
            "n_overlap_samples": (("lag",), overlap.astype(np.int32)),
        },
        coords={
            "pair": np.arange(len(output), dtype=np.int32),
            "lag": lag_times.astype(np.float32),
            "direction": np.asarray(
                ["source_to_target", "target_to_source"], dtype=str
            ),
            "pair_id": ("pair", output["pair_id"].astype(str).to_numpy()),
            "source": ("pair", output["source"].astype(str).to_numpy()),
            "target": ("pair", output["target"].astype(str).to_numpy()),
        },
        attrs={
            "metric": "lagged_oaec_candidate",
            "lag_convention": "negative_lag_source_leads",
            "selection": "xcorr_and_xcorr_resid_fdr_strict_same_direction",
            "inference": "observed_only_no_additional_permutation_test",
            "orthogonalization": "Hipp_pairwise_bidirectional_at_zero_lag",
            "n_hga_frequencies": int(len(centers)),
        },
    )
    return MetricResult(
        metric="lagged_oaec_candidate",
        pair_table=output,
        detail=detail,
        runtime_metadata={
            "n_pairs": int(len(output)),
            "n_lags": int(len(lag_samples)),
            "n_hga_frequencies": int(len(centers)),
            "n_trials": int(coefficients.shape[0]),
        },
    )


def compute_oaec(
    raw_data: np.ndarray,
    raw_times: np.ndarray,
    sfreq: float,
    phase: str,
    pair_frame: pd.DataFrame,
    permutations: np.ndarray,
    config: ConnectivityConfig,
) -> MetricResult:
    """Compute symmetric HGA filterbank OAEC for one analysis entity."""

    phase_mask = phase_time_mask(raw_times, phase)
    centers = hga_filterbank_centers()
    pair_channel_indices = sorted(
        set(pair_frame["source_index"].astype(int))
        | set(pair_frame["target_index"].astype(int))
    )
    compact_index = {
        original: compact
        for compact, original in enumerate(pair_channel_indices)
    }
    coefficients = gaussian_analytic_filterbank(
        np.asarray(raw_data)[:, pair_channel_indices, :],
        sfreq,
        centers,
        time_mask=phase_mask,
        target_sfreq=config.oaec_sfreq,
    )
    n_trials = coefficients.shape[0]
    n_pairs = len(pair_frame)
    n_perm = permutations.shape[0]
    observed_directional = np.full(
        (n_pairs, centers.size, 2), np.nan, dtype=np.float32
    )
    null = np.zeros((n_perm, n_pairs), dtype=np.float32)

    source_indices = np.asarray(
        [
            compact_index[int(value)]
            for value in pair_frame["source_index"]
        ],
        dtype=int,
    )
    target_indices = np.asarray(
        [
            compact_index[int(value)]
            for value in pair_frame["target_index"]
        ],
        dtype=int,
    )
    identity = np.arange(n_trials)
    # A full trial×trial OAEC matrix is cheaper than recomputing envelopes for
    # every permutation once n_perm exceeds n_trials. Limit the pair block to
    # keep the broadcast complex arrays comfortably within the 32 GB job.
    exact_pair_block_size = min(config.pair_block_size, 8)
    for frequency_index in range(centers.size):
        for pair_start in range(0, n_pairs, exact_pair_block_size):
            pair_stop = min(pair_start + exact_pair_block_size, n_pairs)
            source = coefficients[
                :, source_indices[pair_start:pair_stop], frequency_index, :
            ]
            target = coefficients[
                :, target_indices[pair_start:pair_stop], frequency_index, :
            ]
            cross_trial_directional = (
                directional_orthogonalized_correlation_z(
                    source[:, None, ...], target[None, ...]
                )
            )
            observed_directional[
                pair_start:pair_stop, frequency_index
            ] = np.nanmean(
                cross_trial_directional[identity, identity],
                axis=0,
            )
            for permutation_start in range(
                0, n_perm, config.permutation_chunk_size
            ):
                permutation_stop = min(
                    permutation_start + config.permutation_chunk_size,
                    n_perm,
                )
                selected = cross_trial_directional[
                    identity[None, :],
                    permutations[permutation_start:permutation_stop],
                ]
                null[
                    permutation_start:permutation_stop,
                    pair_start:pair_stop,
                ] += np.nanmean(
                    selected, axis=(1, 3)
                ).astype(np.float32)

    null /= float(centers.size)
    observed = np.nanmean(observed_directional, axis=(1, 2))
    inference = scalar_permutation_inference(
        observed, null, tail="two-sided", alpha=config.alpha
    )
    output = pair_frame.copy()
    output["metric"] = "oaec"
    output["stat"] = observed
    for key in (
        "null_mean",
        "null_std",
        "p_uncorrected",
        "q_fdr",
        "p_fwer_maxstat",
        "sig_fdr",
        "sig_fwer",
    ):
        output[key] = inference[key]
    output["qc_pass"] = np.isfinite(observed) & (
        np.isfinite(inference["null_std"])
    )
    variables: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "observed_fisher_z": (
            ("pair", "hga_frequency", "direction"),
            observed_directional,
        ),
        "null_mean": (("pair",), inference["null_mean"].astype(np.float32)),
        "null_std": (("pair",), inference["null_std"].astype(np.float32)),
    }
    if config.save_full_null:
        variables["null_stat"] = (("permutation", "pair"), null)
    detail = xr.Dataset(
        data_vars=variables,
        coords={
            "pair": np.arange(n_pairs, dtype=np.int32),
            "hga_frequency": centers,
            "direction": np.asarray(
                ["source_to_target", "target_to_source"], dtype=str
            ),
            "pair_id": ("pair", pair_frame["pair_id"].astype(str).to_numpy()),
            "source": ("pair", pair_frame["source"].astype(str).to_numpy()),
            "target": ("pair", pair_frame["target"].astype(str).to_numpy()),
        },
        attrs={
            "metric": "oaec",
            "tail": "two-sided",
            "orthogonalization": "Hipp_pairwise_bidirectional",
            "input": "raw_voltage_to_70_200Hz_complex_filterbank",
            "resampling": "frequency_center_demodulation_to_complex_baseband",
            "n_perm": int(n_perm),
        },
    )
    return MetricResult(
        metric="oaec",
        pair_table=output,
        detail=detail,
        runtime_metadata={
            "n_pairs": n_pairs,
            "n_hga_frequencies": int(centers.size),
            "n_trials": n_trials,
        },
    )
