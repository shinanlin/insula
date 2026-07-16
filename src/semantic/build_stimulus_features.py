#!/usr/bin/env python3
"""Build token-level acoustic and phonological features under src/semantic/features/.

Reads Lexical Delay BIDS events (84 tokens: 42 Word + 42 Nonword), extracts
mel log-mean vectors from ``BIDS/stimuli/{token}.wav`` and position×phone
one-hot matrices via g2p_en, then writes a single H5 cache:

  features/stimulus_features.h5

See ``load_stimulus_features.py`` for reading.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import h5py
import librosa
import numpy as np
import rootutils
from sklearn.decomposition import PCA

rootutils.setup_root(
    __file__,
    indicator=".project-root",
    pythonpath=True,
    cwd=True,
)

from src.semantic.features import DEFAULT_BIDS_ROOT, build_stimulus_table, word_to_phonemes
from src.semantic.load_embeddings import DEFAULT_EMBEDDING_DIR, load_embedding_table

logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = PACKAGE_DIR / "features"
DEFAULT_OUT_H5 = DEFAULT_OUT_DIR / "stimulus_features.h5"

SR_TARGET = 16_000
N_MELS = 16
FMIN = 50.0
FMAX = 8_000.0
HOP_LENGTH = 160  # 10 ms at 16 kHz
N_FFT = 512
N_POS = 5
N_PCA = 8
LOG_EPS = 1e-10


def _str_dtype():
    return h5py.string_dtype(encoding="utf-8")


def _resolve_wav(stimuli_dir: Path, token: str) -> Path:
    path = stimuli_dir / f"{token.lower()}.wav"
    if not path.is_file():
        raise FileNotFoundError(f"Missing wav for token {token!r}: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Empty wav for token {token!r}: {path}")
    return path


def extract_mel_logmean(
    wav_path: Path,
    *,
    sr_target: int = SR_TARGET,
    n_mels: int = N_MELS,
    fmin: float = FMIN,
    fmax: float = FMAX,
    hop_length: int = HOP_LENGTH,
    n_fft: int = N_FFT,
) -> tuple[np.ndarray, float]:
    """Return (n_mels,) time-averaged log-mel and duration in seconds."""
    y, sr = librosa.load(str(wav_path), sr=sr_target, mono=True)
    duration_s = float(len(y) / sr)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        hop_length=hop_length,
        n_fft=n_fft,
    )
    log_mel = np.log(mel + LOG_EPS)
    return log_mel.mean(axis=1).astype(np.float32), duration_s


def phones_to_pos_matrix(
    phones: tuple[str, ...],
    phone_index: dict[str, int],
    n_pos: int = N_POS,
) -> np.ndarray:
    """Position × phone one-hot matrix, shape (n_pos, n_phone_types)."""
    n_phones = len(phone_index)
    mat = np.zeros((n_pos, n_phones), dtype=np.uint8)
    seq = list(phones[:n_pos])
    if len(seq) < n_pos:
        seq = seq + [""] * (n_pos - len(seq))
    for pos, phone in enumerate(seq):
        if not phone:
            continue
        idx = phone_index.get(phone.upper())
        if idx is None:
            raise KeyError(f"Phone {phone!r} not in inventory")
        mat[pos, idx] = 1
    return mat


def build_phone_inventory(tokens: list[str]) -> tuple[list[str], list[tuple[str, ...]]]:
    """Collect sorted phone inventory and per-token phone sequences."""
    seqs = [word_to_phonemes(t) for t in tokens]
    inventory = sorted({p.upper() for seq in seqs for p in seq})
    return inventory, seqs


def _token_inventory(
    bids_root: Path,
    embedding_dir: Path,
) -> tuple[list[str], list[str]]:
    """Union of events tokens and GloVe Word list (older subjects use expanded set)."""
    table = build_stimulus_table(bids_root)
    lex_map = {str(t): str(lex) for t, lex in zip(table["token"], table["lexicality"])}
    try:
        emb = load_embedding_table(embedding_dir)
        for tok in emb.tokens:
            key = str(tok).lower()
            lex_map.setdefault(key, "Word")
    except FileNotFoundError:
        logger.warning("GloVe embedding table not found; using events tokens only")

    tokens = sorted(lex_map)
    lexicality = [lex_map[t] for t in tokens]
    return tokens, lexicality


def build(
    bids_root: Path,
    stimuli_dir: Path,
    out_h5: Path,
    *,
    n_mels: int = N_MELS,
    n_pca: int = N_PCA,
    n_pos: int = N_POS,
    embedding_dir: Path = DEFAULT_EMBEDDING_DIR,
) -> Path:
    tokens, lexicality = _token_inventory(bids_root, embedding_dir)
    n_tokens = len(tokens)
    n_word = sum(lex == "Word" for lex in lexicality)
    n_nw = sum(lex == "Nonword" for lex in lexicality)
    logger.info("Token inventory: %d total (%d Word, %d Nonword)", n_tokens, n_word, n_nw)

    wav_paths: list[str] = []
    durations = np.zeros(n_tokens, dtype=np.float32)
    mel_logmean = np.zeros((n_tokens, n_mels), dtype=np.float32)

    for i, token in enumerate(tokens):
        wav = _resolve_wav(stimuli_dir, token)
        wav_paths.append(str(wav.resolve()))
        mel_logmean[i], durations[i] = extract_mel_logmean(wav, n_mels=n_mels)
        logger.info(
            "acoustic %s (%s): dur=%.3fs mel_range=[%.2f, %.2f]",
            token,
            lexicality[i],
            durations[i],
            float(mel_logmean[i].min()),
            float(mel_logmean[i].max()),
        )

    phone_types, phone_seqs = build_phone_inventory(tokens)
    phone_index = {p: j for j, p in enumerate(phone_types)}
    n_phone_types = len(phone_types)

    phones_str = []
    n_phones_arr = np.zeros(n_tokens, dtype=np.int16)
    pos_phone = np.zeros((n_tokens, n_pos, n_phone_types), dtype=np.uint8)

    for i, (token, seq) in enumerate(zip(tokens, phone_seqs)):
        phones_str.append(" ".join(seq))
        n_phones_arr[i] = len(seq)
        if len(seq) != n_pos:
            logger.warning(
                "token %s has %d phones (expected %d); pad/trim for position matrix",
                token,
                len(seq),
                n_pos,
            )
        pos_phone[i] = phones_to_pos_matrix(seq, phone_index, n_pos=n_pos)

    pos_phone_flat = pos_phone.reshape(n_tokens, n_pos * n_phone_types)

    mel_pca_model = PCA(n_components=n_pca, random_state=0).fit(mel_logmean)
    mel_pca = mel_pca_model.transform(mel_logmean)
    phon_pca_model = PCA(n_components=n_pca, random_state=0).fit(
        pos_phone_flat.astype(np.float32)
    )
    phon_pca = phon_pca_model.transform(pos_phone_flat.astype(np.float32))

    out_h5 = Path(out_h5)
    out_h5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_h5, "w") as hf:
        hf.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        hf.attrs["bids_root"] = str(bids_root)
        hf.attrs["stimuli_dir"] = str(stimuli_dir)
        hf.attrs["n_tokens"] = n_tokens
        hf.attrs["n_mels"] = n_mels
        hf.attrs["n_pos"] = n_pos
        hf.attrs["n_phone_types"] = n_phone_types
        hf.attrs["n_pca"] = n_pca
        hf.attrs["sr_target"] = SR_TARGET
        hf.attrs["fmin"] = FMIN
        hf.attrs["fmax"] = FMAX
        hf.attrs["hop_length"] = HOP_LENGTH
        hf.attrs["n_fft"] = N_FFT

        hf.create_dataset("tokens", data=np.asarray(tokens, dtype=object), dtype=_str_dtype())
        hf.create_dataset(
            "lexicality", data=np.asarray(lexicality, dtype=object), dtype=_str_dtype()
        )
        hf.create_dataset("wav_path", data=np.asarray(wav_paths, dtype=object), dtype=_str_dtype())
        hf.create_dataset("duration_s", data=durations, dtype=np.float32)

        acous = hf.create_group("acoustic")
        acous.attrs["spectrum"] = "mel"
        acous.attrs["pca_explained_variance_ratio"] = mel_pca_model.explained_variance_ratio_
        acous.create_dataset("mel_logmean", data=mel_logmean, dtype=np.float32)
        acous.create_dataset("mel_pca", data=mel_pca.astype(np.float32), dtype=np.float32)

        phon = hf.create_group("phonology")
        phon.attrs["g2p"] = "g2p_en"
        phon.attrs["stress_stripped"] = True
        phon.attrs["pca_explained_variance_ratio"] = phon_pca_model.explained_variance_ratio_
        phon.create_dataset(
            "phones_str", data=np.asarray(phones_str, dtype=object), dtype=_str_dtype()
        )
        phon.create_dataset("n_phones", data=n_phones_arr, dtype=np.int16)
        phon.create_dataset(
            "phone_types",
            data=np.asarray(phone_types, dtype=object),
            dtype=_str_dtype(),
        )
        phon.create_dataset("pos_phone", data=pos_phone, dtype=np.uint8)
        phon.create_dataset("pos_phone_flat", data=pos_phone_flat, dtype=np.uint8)
        phon.create_dataset("pos_phone_pca", data=phon_pca.astype(np.float32), dtype=np.float32)

    logger.info(
        "Wrote %d tokens to %s (Word=%d, Nonword=%d)",
        n_tokens,
        out_h5,
        sum(lex == "Word" for lex in lexicality),
        sum(lex == "Nonword" for lex in lexicality),
    )
    return out_h5


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids_root", type=Path, default=DEFAULT_BIDS_ROOT)
    parser.add_argument(
        "--stimuli_dir",
        type=Path,
        default=None,
        help="Directory with {token}.wav (default: bids_root/stimuli)",
    )
    parser.add_argument("--out_h5", type=Path, default=DEFAULT_OUT_H5)
    parser.add_argument("--embedding_dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--n_mels", type=int, default=N_MELS)
    parser.add_argument("--n_pca", type=int, default=N_PCA)
    parser.add_argument("--n_pos", type=int, default=N_POS)
    args = parser.parse_args()

    stimuli_dir = args.stimuli_dir or (args.bids_root / "stimuli")
    build(
        bids_root=args.bids_root,
        stimuli_dir=stimuli_dir,
        out_h5=args.out_h5,
        n_mels=args.n_mels,
        n_pca=args.n_pca,
        n_pos=args.n_pos,
        embedding_dir=args.embedding_dir,
    )


if __name__ == "__main__":
    main()
