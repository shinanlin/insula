"""Load token-level stimulus features from stimulus_features.h5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURES_H5 = PACKAGE_DIR / "features" / "stimulus_features.h5"


def _decode_str_array(arr: np.ndarray) -> np.ndarray:
    out = np.empty(len(arr), dtype=object)
    for i, x in enumerate(arr):
        out[i] = x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
    return out


@dataclass(frozen=True)
class StimulusFeatures:
    """Token-aligned acoustic and phonological features."""

    tokens: np.ndarray  # (n,)
    lexicality: np.ndarray  # (n,)
    wav_path: np.ndarray  # (n,)
    duration_s: np.ndarray  # (n,)

    mel_logmean: np.ndarray  # (n, n_mels)
    mel_pca: np.ndarray  # (n, n_pca)

    phones_str: np.ndarray  # (n,)
    n_phones: np.ndarray  # (n,)
    phone_types: np.ndarray  # (P,)
    pos_phone: np.ndarray  # (n, n_pos, P)
    pos_phone_flat: np.ndarray  # (n, n_pos * P)
    pos_phone_pca: np.ndarray  # (n, n_pca)

    attrs: dict

    @property
    def n_tokens(self) -> int:
        return int(len(self.tokens))

    @property
    def word_mask(self) -> np.ndarray:
        return self.lexicality == "Word"

    @property
    def nonword_mask(self) -> np.ndarray:
        return self.lexicality == "Nonword"

    def token_index(self) -> dict[str, int]:
        return {str(t): i for i, t in enumerate(self.tokens)}

    def by_token(self, token: str) -> int:
        idx = self.token_index().get(token.lower())
        if idx is None:
            raise KeyError(f"Token {token!r} not in stimulus_features.h5")
        return idx

    def subset_words(self) -> StimulusFeatures:
        mask = self.word_mask
        return StimulusFeatures(
            tokens=self.tokens[mask],
            lexicality=self.lexicality[mask],
            wav_path=self.wav_path[mask],
            duration_s=self.duration_s[mask],
            mel_logmean=self.mel_logmean[mask],
            mel_pca=self.mel_pca[mask],
            phones_str=self.phones_str[mask],
            n_phones=self.n_phones[mask],
            phone_types=self.phone_types,
            pos_phone=self.pos_phone[mask],
            pos_phone_flat=self.pos_phone_flat[mask],
            pos_phone_pca=self.pos_phone_pca[mask],
            attrs=dict(self.attrs),
        )


def align_stimulus_features(
    trial_tokens: list[str] | np.ndarray,
    features: StimulusFeatures | None = None,
    h5_path: Path | str = DEFAULT_FEATURES_H5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (phon, acoustic) matrices aligned to trial token order.

    phon: ``(n_trials, n_pos * n_phones)`` from ``pos_phone_flat``
    acoustic: ``(n_trials, n_mels)`` from ``mel_logmean``
    """
    feat = features or load_stimulus_features(h5_path)
    lookup = feat.token_index()
    phon_rows = []
    acous_rows = []
    for tok in trial_tokens:
        key = str(tok).lower()
        if key not in lookup:
            raise KeyError(f"Token {key!r} not found in stimulus_features.h5")
        idx = lookup[key]
        phon_rows.append(feat.pos_phone_flat[idx])
        acous_rows.append(feat.mel_logmean[idx])
    return (
        np.asarray(phon_rows, dtype=np.float64),
        np.asarray(acous_rows, dtype=np.float64),
    )


def load_stimulus_features(
    h5_path: Path | str = DEFAULT_FEATURES_H5,
) -> StimulusFeatures:
    """Load all token-level features from H5."""
    h5_path = Path(h5_path)
    if not h5_path.is_file():
        raise FileNotFoundError(
            f"Stimulus features not found: {h5_path}. "
            "Run: python src/semantic/build_stimulus_features.py"
        )

    with h5py.File(h5_path, "r") as hf:
        attrs = dict(hf.attrs)
        acous = hf["acoustic"]
        phon = hf["phonology"]
        return StimulusFeatures(
            tokens=_decode_str_array(hf["tokens"][:]),
            lexicality=_decode_str_array(hf["lexicality"][:]),
            wav_path=_decode_str_array(hf["wav_path"][:]),
            duration_s=np.asarray(hf["duration_s"], dtype=np.float32),
            mel_logmean=np.asarray(acous["mel_logmean"], dtype=np.float32),
            mel_pca=np.asarray(acous["mel_pca"], dtype=np.float32),
            phones_str=_decode_str_array(phon["phones_str"][:]),
            n_phones=np.asarray(phon["n_phones"], dtype=np.int16),
            phone_types=_decode_str_array(phon["phone_types"][:]),
            pos_phone=np.asarray(phon["pos_phone"], dtype=np.uint8),
            pos_phone_flat=np.asarray(phon["pos_phone_flat"], dtype=np.uint8),
            pos_phone_pca=np.asarray(phon["pos_phone_pca"], dtype=np.float32),
            attrs=attrs,
        )
