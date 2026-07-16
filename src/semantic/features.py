"""Stimulus feature table for Lexical Delay semantic analyses.

Builds a per-token table with lexicality, phonological codes, optional
frequency, and embedding vectors. Semantic RSA uses Word rows only.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_BIDS_ROOT = Path("/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS")

ARTICULATORY_PHONE_TO_LABEL = {
    "M": "sonorant",
    "N": "sonorant",
    "NG": "sonorant",
    "L": "sonorant",
    "R": "sonorant",
    "W": "sonorant",
    "Y": "sonorant",
    "B": "labial_obstruent",
    "P": "labial_obstruent",
    "F": "labial_obstruent",
    "V": "labial_obstruent",
    "D": "coronal_obstruent",
    "T": "coronal_obstruent",
    "S": "coronal_obstruent",
    "Z": "coronal_obstruent",
    "JH": "coronal_obstruent",
    "CH": "coronal_obstruent",
    "SH": "coronal_obstruent",
    "ZH": "coronal_obstruent",
    "TH": "coronal_obstruent",
    "DH": "coronal_obstruent",
    "G": "posterior_obstruent",
    "K": "posterior_obstruent",
    "HH": "posterior_obstruent",
}

_G2P = None


@lru_cache(maxsize=4096)
def word_to_phonemes(word: str) -> tuple[str, ...]:
    """Grapheme-to-phoneme via g2p_en (same stack as prepare_decoding_dataset)."""
    global _G2P
    if _G2P is None:
        from g2p_en import G2p

        _G2P = G2p()
    phones = [p for p in _G2P(word) if p not in {" ", ""}]
    cleaned = []
    for p in phones:
        if isinstance(p, str):
            cleaned.append(p.rstrip("012"))
    return tuple(p for p in cleaned if p != "")


def articulator_of(word: str) -> Optional[str]:
    phones = word_to_phonemes(word)
    if not phones:
        return None
    return ARTICULATORY_PHONE_TO_LABEL.get(phones[0].upper())


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]


def phoneme_edit_distance(a: str, b: str) -> int:
    return levenshtein(" ".join(word_to_phonemes(a)), " ".join(word_to_phonemes(b)))


def tokens_from_events(
    bids_root: Path | str = DEFAULT_BIDS_ROOT,
    subjects: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Parse unique tokens and lexicality from BIDS events.tsv files."""
    bids_root = Path(bids_root)
    pattern = "sub-*/ieeg/*_events.tsv"
    rows = []
    for path in sorted(bids_root.glob(pattern)):
        subject = path.parts[-3].replace("sub-", "")
        if subjects is not None and subject not in set(subjects):
            continue
        df = pd.read_csv(path, sep="\t")
        if "trial_type" not in df.columns:
            continue
        for tt in df["trial_type"].astype(str):
            parts = tt.split("/")
            if len(parts) < 4:
                continue
            if parts[0] not in {"Auditory_stim", "Cue", "Delay", "Go", "Resp"}:
                continue
            rows.append(
                {
                    "subject": subject,
                    "description": parts[1],  # Yes_No / Repeat
                    "lexicality": parts[2],  # Word / Nonword
                    "token": parts[3].lower(),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No events parsed under {bids_root}")
    return pd.DataFrame(rows)


def build_stimulus_table(
    bids_root: Path | str = DEFAULT_BIDS_ROOT,
    subjects: Optional[Iterable[str]] = None,
    frequency_csv: Optional[Path | str] = None,
    embedding_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """One row per unique token with lexicality and control features.

    Parameters
    ----------
    frequency_csv
        Optional CSV with columns ``token,log_freq`` (or ``word,freq``).
    embedding_path
        Optional ``.npy`` / ``.npz`` with aligned embeddings; if omitted,
        ``embedding`` column is left empty for a later fill step.
    """
    events = tokens_from_events(bids_root, subjects=subjects)
    tokens = (
        events.groupby("token", as_index=False)
        .agg(
            lexicality=("lexicality", lambda s: s.mode().iloc[0]),
            n_event_rows=("token", "size"),
            n_subjects=("subject", "nunique"),
        )
        .sort_values("token")
        .reset_index(drop=True)
    )
    tokens["n_letters"] = tokens["token"].str.len()
    phones = tokens["token"].map(lambda w: " ".join(word_to_phonemes(w)))
    tokens["phonemes"] = phones
    tokens["n_phones"] = tokens["phonemes"].str.split().map(len)
    tokens["articulator"] = tokens["token"].map(articulator_of)

    tokens["log_freq"] = np.nan
    if frequency_csv is not None:
        freq = pd.read_csv(frequency_csv)
        rename = {}
        if "word" in freq.columns and "token" not in freq.columns:
            rename["word"] = "token"
        if "freq" in freq.columns and "log_freq" not in freq.columns:
            freq["log_freq"] = np.log10(freq["freq"].astype(float) + 1e-6)
        freq = freq.rename(columns=rename)
        tokens = tokens.merge(freq[["token", "log_freq"]], on="token", how="left")

    tokens["embedding"] = None
    if embedding_path is not None:
        logger.warning(
            "embedding_path provided but vector alignment is left to the caller; "
            "store arrays alongside this table in results/semantic/"
        )

    return tokens


def phonological_rdm(tokens: Iterable[str]) -> np.ndarray:
    """Pairwise phoneme-sequence edit distance RDM."""
    toks = list(tokens)
    n = len(toks)
    rdm = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(phoneme_edit_distance(toks[i], toks[j]))
            rdm[i, j] = rdm[j, i] = d
    return rdm


def orthographic_rdm(tokens: Iterable[str]) -> np.ndarray:
    toks = list(tokens)
    n = len(toks)
    rdm = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(levenshtein(toks[i], toks[j]))
            rdm[i, j] = rdm[j, i] = d
    return rdm


def scalar_diff_rdm(values: np.ndarray) -> np.ndarray:
    """Absolute pairwise difference RDM for a 1-D feature (e.g. log frequency)."""
    v = np.asarray(values, dtype=float).reshape(-1)
    return np.abs(v[:, None] - v[None, :])
