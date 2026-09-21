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
    WPLI_PRIMARY_BAND,
    WPLI_SECONDARY_BANDS,
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
    Float32 inputs stay float32 to keep null temporary arrays small; other
    floating inputs keep their dtype, and non-floats are cast to float32.
    """

    values = np.asarray(imaginary_cross_spectrum)
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float32, copy=False)
    sum_imag = np.sum(values, axis=axis)
    sum_abs = np.sum(np.abs(values), axis=axis)
    sum_sq = np.sum(values**2, axis=axis)
    numerator = sum_imag**2 - sum_sq
    denominator = sum_abs**2 - sum_sq
    result = np.divide(
        numerator,
        denominator,
        out=np.full(np.shape(numerator), np.nan, dtype=values.dtype),
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


def _band_mean_dwpli_from_tf(
    observed_tf: np.ndarray,
    source_coeff: np.ndarray,
    target_coeff: np.ndarray,
    permutations: np.ndarray,
    frequency_mask: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Average per-bin dwPLI within a band for observed and permuted nulls.

    The null re-pairs each source trial ``t`` with target trial
    ``permutations[p, t]`` before forming the imaginary cross-spectrum.
    Shuffling already-paired ``imag(source*conj(target))`` is invariant for
    dwPLI and must not be used.
    """

    band_tf = observed_tf[frequency_mask, :]
    observed = float(np.nanmean(band_tf))

    n_trials = source_coeff.shape[0]
    source_band = source_coeff[:, frequency_mask, :].reshape(n_trials, -1)
    target_band = target_coeff[:, frequency_mask, :].reshape(n_trials, -1)
    target_shuf = target_band[permutations]
    # imag(source * conj(target_shuf)); keep float32 to halve peak null RAM
    cross_imag = np.subtract(
        source_band.imag[None, :, :] * target_shuf.real,
        source_band.real[None, :, :] * target_shuf.imag,
        dtype=np.float32,
    )
    null_per_bin = debiased_wpli_from_imag(cross_imag, axis=1)[0]
    null = np.nanmean(null_per_bin, axis=1).astype(np.float32)
    return observed, null


def _compute_one_pair_dwpli(
    source: np.ndarray,
    target: np.ndarray,
    permutations: np.ndarray,
    freqs: np.ndarray,
    band_names: list[str],
) -> dict[str, np.ndarray]:
    """Observed TF maps, band stats, and null for one electrode pair."""

    n_perm = permutations.shape[0]
    n_bands = len(band_names)
    cross = source * np.conj(target)
    tf_value, tf_denominator = debiased_wpli_from_imag(
        np.imag(cross), axis=0
    )
    observed_tf_row = tf_value.astype(np.float32)
    valid_tf_row = np.isfinite(tf_value) & (tf_denominator > 0.0)

    observed_band_row = np.full(n_bands, np.nan, dtype=np.float32)
    null_pair = np.full((n_perm, n_bands), np.nan, dtype=np.float32)
    source_power_row = np.full(n_bands, np.nan, dtype=np.float32)
    target_power_row = np.full(n_bands, np.nan, dtype=np.float32)
    valid_fraction_row = np.zeros(n_bands, dtype=np.float32)

    for band_index, band_name in enumerate(band_names):
        frequency_mask = _band_frequency_mask(freqs, band_name)
        observed_band_row[band_index], null_pair[:, band_index] = (
            _band_mean_dwpli_from_tf(
                observed_tf_row,
                source,
                target,
                permutations,
                frequency_mask,
            )
        )
        source_power_row[band_index] = float(
            np.mean(np.abs(source[:, frequency_mask, :]) ** 2)
        )
        target_power_row[band_index] = float(
            np.mean(np.abs(target[:, frequency_mask, :]) ** 2)
        )
        valid_fraction_row[band_index] = float(
            np.mean(valid_tf_row[frequency_mask])
        )

    return {
        "observed_tf": observed_tf_row,
        "valid_tf": valid_tf_row,
        "observed_band": observed_band_row,
        "null": null_pair,
        "source_power": source_power_row,
        "target_power": target_power_row,
        "valid_fraction": valid_fraction_row,
    }


def _assign_band_inference(
    output: dict[str, np.ndarray],
    band_index: int,
    n_pairs: int,
    n_bands: int,
    inference: dict[str, np.ndarray],
    *,
    band_stride: int,
    local_offset: int = 0,
) -> None:
    for key in (
        "null_mean",
        "null_std",
        "p_uncorrected",
        "q_fdr",
        "p_fwer_maxstat",
        "sig_fdr",
        "sig_fwer",
    ):
        flat_rows = band_index + n_bands * np.arange(n_pairs)
        output[key][flat_rows] = inference[key][local_offset::band_stride]


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
    primary_index = band_names.index(WPLI_PRIMARY_BAND)
    secondary_indices = [
        band_names.index(name) for name in WPLI_SECONDARY_BANDS
    ]
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

    pair_specs = [
        (
            pair_pos,
            compact_index[int(pair_frame.iloc[pair_pos]["source_index"])],
            compact_index[int(pair_frame.iloc[pair_pos]["target_index"])],
        )
        for pair_pos in range(n_pairs)
    ]

    def _process_pair(
        pair_pos: int, source_index: int, target_index: int
    ) -> tuple[int, dict[str, np.ndarray]]:
        source = coefficients[:, source_index]
        target = coefficients[:, target_index]
        return pair_pos, _compute_one_pair_dwpli(
            source,
            target,
            permutations,
            freqs,
            band_names,
        )

    if config.n_jobs == 1:
        pair_results = [
            _process_pair(pair_pos, source_index, target_index)
            for pair_pos, source_index, target_index in pair_specs
        ]
    else:
        from joblib import Parallel, delayed

        pair_results = Parallel(
            n_jobs=config.n_jobs,
            prefer="threads",
            batch_size=1,
        )(
            delayed(_process_pair)(pair_pos, source_index, target_index)
            for pair_pos, source_index, target_index in pair_specs
        )

    for pair_pos, out in pair_results:
        observed_tf[pair_pos] = out["observed_tf"]
        valid_tf[pair_pos] = out["valid_tf"]
        observed_band[pair_pos] = out["observed_band"]
        null[:, pair_pos, :] = out["null"]
        source_power[pair_pos] = out["source_power"]
        target_power[pair_pos] = out["target_power"]
        valid_fraction[pair_pos] = out["valid_fraction"]

    inference_output: dict[str, np.ndarray] = {
        key: np.full(n_pairs * n_bands, np.nan, dtype=float)
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
    primary_inference = scalar_permutation_inference(
        observed_band[:, primary_index],
        null[:, :, primary_index],
        tail="greater",
        alpha=config.alpha,
    )
    _assign_band_inference(
        inference_output,
        primary_index,
        n_pairs,
        n_bands,
        primary_inference,
        band_stride=1,
    )
    secondary_observed = observed_band[:, secondary_indices].reshape(-1)
    secondary_null = null[:, :, secondary_indices].reshape(n_perm, -1)
    secondary_inference = scalar_permutation_inference(
        secondary_observed,
        secondary_null,
        tail="greater",
        alpha=config.alpha,
    )
    for local_index, band_index in enumerate(secondary_indices):
        _assign_band_inference(
            inference_output,
            band_index,
            n_pairs,
            n_bands,
            secondary_inference,
            band_stride=len(secondary_indices),
            local_offset=local_index,
        )

    flat_observed = observed_band.reshape(-1)
    repeated = pair_frame.loc[
        pair_frame.index.repeat(n_bands)
    ].reset_index(drop=True)
    repeated["band"] = np.tile(np.asarray(band_names), n_pairs)
    repeated["metric"] = "wpli2_debiased_tf"
    repeated["inference_family"] = np.where(
        repeated["band"] == WPLI_PRIMARY_BAND,
        "broadband_primary",
        "subband_secondary",
    )
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
    repeated["exploratory_flag"] = repeated["band"] != WPLI_PRIMARY_BAND
    repeated.loc[
        (repeated["band"] == "theta")
        & (duration <= 0.5 + 1e-9),
        "exploratory_flag",
    ] = True
    for key in inference_output:
        repeated[key] = inference_output[key]
    repeated["sig_fdr"] = repeated["sig_fdr"].astype(bool)
    repeated["sig_fwer"] = repeated["sig_fwer"].astype(bool)
    repeated["qc_pass"] = (
        np.isfinite(flat_observed)
        & np.isfinite(inference_output["null_std"])
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
            inference_output["null_mean"]
            .reshape(n_pairs, n_bands)
            .astype(np.float32),
        ),
        "null_std": (
            ("pair", "band"),
            inference_output["null_std"]
            .reshape(n_pairs, n_bands)
            .astype(np.float32),
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
            "band_statistic": (
                "mean_of_per_frequency_time_dwpli_within_band"
            ),
            "inference": (
                "broadband_primary_and_subband_secondary_families"
            ),
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
