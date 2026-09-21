#!/usr/bin/env python3
"""Loaders for pooled-ROI decoding datasets that also expose stimulus identity.

``run_decoding.load_roi_data`` returns only ``X``/``y`` and is consumed by
callers that unpack a fixed three-tuple, so it cannot grow a grouping vector
without breaking them. This module reads the same HDF5 files and additionally
returns the per-trial ``condition`` (the stimulus word) and ``trial`` identity
when present, which item-grouped CV and cross-condition pairing need.
"""

import os

import h5py
import numpy as np
from mne_bids import BIDSPath


def load_pooled_roi(bids_root, ref, roi, datatype, description, phase, band, tmin, tmax):
    """Load one pooled-ROI file plus its per-trial stimulus identities.

    Parameters
    ----------
    bids_root : str or Path
        Root of the BIDS dataset holding ``derivatives/decoding({ref})``.
    ref : str
        Reference scheme, e.g. ``'bipolar'``.
    roi : str
        Pooled-ROI identifier used as the BIDS subject, e.g. ``'STGl'``.
    datatype : str
        Decoding target, e.g. ``'lexicality'``.
    description : str
        Trial description, e.g. ``'Repeat'``.
    phase : str
        Epoch alignment, e.g. ``'Stimulus'``.
    band : str
        Frequency band suffix, e.g. ``'highgamma'``.
    tmin, tmax : float
        Temporal crop bounds in seconds relative to the epoch lock.

    Returns
    -------
    X : ndarray, shape (n_trials, n_channels, n_times)
    y : ndarray, shape (n_trials,)
    conditions : ndarray of str, shape (n_trials,)
        Per-trial stimulus word, used as the CV grouping vector. Each word
        appears twice (once per presentation), so grouping on this vector is
        word-level grouping.
    labels : ndarray of str
    channels : ndarray of str
    trials : ndarray of str or None
        Per-trial identity when the file stores a ``trial`` dataset
        (functional pools). Anatomical ROI files may omit it; callers that
        need a pair key should then synthesise one from ``conditions``.
    src_path : BIDSPath
        The matched source file.
    """
    root = BIDSPath(
        root=os.path.join(bids_root, "derivatives", f"decoding({ref})"),
        datatype=datatype,
        description=description,
        suffix=band,
        processing=phase,
        extension=".h5",
        check=False,
    )
    matches = root.copy().update(subject=roi).match()
    if not matches:
        raise FileNotFoundError(f"No {datatype} file for ROI={roi} phase={phase} desc={description}")
    if len(matches) > 1:
        raise ValueError(f"Expected one file, found {len(matches)}: {[str(m) for m in matches]}")

    with h5py.File(matches[0], "r") as stream:
        X = stream["X"][:]
        y = stream["y"][:]
        conditions = _decode_str_array(stream["condition"][:])
        labels = _decode_str_array(stream["label"][:])
        channels = _decode_str_array(stream["channel"][:])
        trials = (
            _decode_str_array(stream["trial"][:]) if "trial" in stream else None
        )
        t_start = float(stream.attrs["tmin"])
        fs = float(stream.attrs["fs"])

    start_idx = int(fs * (tmin - t_start))
    end_idx = int(fs * (tmax - t_start))
    X = X[:, :, start_idx:end_idx]
    return X, y, conditions, labels, channels, trials, matches[0]


def _decode_str_array(values):
    decoded = []
    for value in np.asarray(values):
        if isinstance(value, bytes):
            decoded.append(value.decode())
        else:
            decoded.append(str(value))
    return np.asarray(decoded)


def pair_keys(trials, conditions):
    """Stable per-trial keys for aligning two pooled-ROI loads.

    Functional files store ``trial`` as ``{word}_{presentation}``. Anatomical
    ROI files often omit that dataset; synthesise the same scheme from the
    within-file occurrence of each word so Repeat and Decision still pair.
    """
    if trials is not None:
        return _decode_str_array(trials)
    counts = {}
    keys = []
    for condition in _decode_str_array(conditions):
        counts[condition] = counts.get(condition, 0) + 1
        keys.append(f"{condition}_{counts[condition]}")
    return np.asarray(keys)


def pair_pooled_conditions(
    X_a,
    y_a,
    conditions_a,
    labels_a,
    channels_a,
    trials_a,
    X_b,
    y_b,
    conditions_b,
    labels_b,
    channels_b,
    trials_b,
):
    """Align two pooled-ROI loads by trial identity, then by channel name.

    Trial order follows ``a``. Labels must agree on the shared trials.
    Channels are the name intersection in ``a``'s order. ``y`` is taken from
    ``a`` after the label check, so integer codes do not have to match if the
    string labels already do.

    Returns
    -------
    X_a, y, conditions, labels, channels, keys, X_b
    """
    keys_a = pair_keys(trials_a, conditions_a)
    keys_b = pair_keys(trials_b, conditions_b)
    index_b = {key: index for index, key in enumerate(keys_b)}
    common = [key for key in keys_a if key in index_b]
    if not common:
        raise ValueError("Pooled conditions have no common trial identities")
    index_a = {key: index for index, key in enumerate(keys_a)}
    idx_a = np.asarray([index_a[key] for key in common], dtype=int)
    idx_b = np.asarray([index_b[key] for key in common], dtype=int)

    labels_common_a = np.asarray(labels_a)[idx_a]
    labels_common_b = np.asarray(labels_b)[idx_b]
    mismatched = [
        key
        for key, left, right in zip(common, labels_common_a, labels_common_b)
        if left != right
    ]
    if mismatched:
        raise ValueError(
            f"Cross-condition labels differ for trials: {mismatched[:10]}"
        )

    channels_a = _decode_str_array(channels_a)
    channels_b = _decode_str_array(channels_b)
    keep_channels = [channel for channel in channels_a if channel in set(channels_b)]
    if not keep_channels:
        raise ValueError("Pooled conditions have no common channels")
    ch_a = np.asarray([int(np.flatnonzero(channels_a == channel)[0]) for channel in keep_channels])
    ch_b = np.asarray([int(np.flatnonzero(channels_b == channel)[0]) for channel in keep_channels])

    X_a = np.asarray(X_a)[idx_a][:, ch_a]
    X_b = np.asarray(X_b)[idx_b][:, ch_b]
    y = np.asarray(y_a)[idx_a]
    conditions = np.asarray(conditions_a)[idx_a]
    labels = labels_common_a
    return X_a, y, conditions, labels, np.asarray(keep_channels), np.asarray(common), X_b
