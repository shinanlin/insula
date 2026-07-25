"""Configuration shared by the pairwise connectivity estimators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Mapping

import numpy as np


SCHEMA_VERSION = "1.1.0"

# Half-open intervals are enforced with explicit time masks, not Epochs.crop.
PHASE_WINDOWS: Mapping[str, tuple[float, float]] = {
    "Stimulus": (0.0, 0.5),
    "Delay": (0.0, 1.0),
    "Go": (0.0, 0.5),
    "Response": (-0.5, 0.5),
}

WPLI_BANDS: Mapping[str, tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "broadband": (4.0, 30.0),
}

DEFAULT_DATASETS: Mapping[str, str] = {
    "LexicalDelay": "/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS",
    "LexicalNoDelay": "/cwork/ns458/BIDS-1.0_LexicalDecRepNoDelay/BIDS",
    "PhonemeSequence": "/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS",
    "PictureNaming": "/cwork/ns458/BIDS-1.3_PictureNaming/BIDS",
    "SentenceRep": "/cwork/ns458/BIDS-1.4_SentenceRep/BIDS",
}


@dataclass(frozen=True)
class ConnectivityConfig:
    """Numerical and inferential settings for one analysis entity."""

    n_perm: int = 1_000
    random_state: int = 42
    alpha: float = 0.05
    min_trials: int = 30
    max_lag_s: float = 0.25
    permutation_chunk_size: int = 100
    pair_block_size: int = 32
    n_jobs: int = 1
    oaec_sfreq: float = 128.0
    wpli_sfreq: float = 256.0
    wpli_freq_step: float = 1.0
    save_full_null: bool = False

    def validate(self) -> None:
        if self.n_perm < 1:
            raise ValueError("n_perm must be positive")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        if self.min_trials < 3:
            raise ValueError("min_trials must be at least 3")
        if self.max_lag_s <= 0:
            raise ValueError("max_lag_s must be positive")
        if self.permutation_chunk_size < 1 or self.pair_block_size < 1:
            raise ValueError("chunk sizes must be positive")
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be positive")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def stable_hash(self, extra: Mapping[str, object] | None = None) -> str:
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            **self.as_dict(),
        }
        if extra:
            payload["extra"] = dict(extra)
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        return sha256(encoded).hexdigest()[:16]


def phase_time_mask(times: np.ndarray, phase: str) -> np.ndarray:
    """Return the prescribed half-open time mask for ``phase``."""

    if phase not in PHASE_WINDOWS:
        raise ValueError(
            f"Unknown phase {phase!r}; expected one of {tuple(PHASE_WINDOWS)}"
        )
    start, stop = PHASE_WINDOWS[phase]
    mask = (np.asarray(times) >= start) & (np.asarray(times) < stop)
    if not np.any(mask):
        raise ValueError(
            f"No samples in phase={phase!r} interval [{start}, {stop})"
        )
    return mask


def wpli_frequencies(step: float = 1.0) -> np.ndarray:
    """Return the inclusive 4--30 Hz frequency grid."""

    if step <= 0:
        raise ValueError("frequency step must be positive")
    return np.arange(4.0, 30.0 + step / 2.0, step, dtype=float)


def wpli_n_cycles(freqs: np.ndarray) -> np.ndarray:
    """Frequency-dependent Morlet cycles from the approved analysis plan."""

    return np.clip(np.asarray(freqs, dtype=float) / 2.0, 3.0, 7.0)
