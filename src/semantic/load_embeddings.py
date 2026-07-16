"""Load Word-token GloVe embeddings from src/semantic/embedding/."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_EMBEDDING_DIR = PACKAGE_DIR / "embedding"


@dataclass(frozen=True)
class EmbeddingTable:
    """Word tokens and aligned GloVe vectors."""

    tokens: np.ndarray  # (n_word,) object
    vectors: np.ndarray  # (n_word, dim) float

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def token_to_index(self) -> dict[str, int]:
        return {str(t): i for i, t in enumerate(self.tokens)}


def load_embedding_table(embedding_dir: Path | str = DEFAULT_EMBEDDING_DIR) -> EmbeddingTable:
    """Load stimulus_tokens_word.npy and embeddings_glove300.npy."""
    embedding_dir = Path(embedding_dir)
    tokens = np.load(embedding_dir / "stimulus_tokens_word.npy", allow_pickle=True)
    vectors = np.load(embedding_dir / "embeddings_glove300.npy")
    if tokens.shape[0] != vectors.shape[0]:
        raise ValueError(
            f"Token/vector count mismatch: {tokens.shape[0]} vs {vectors.shape[0]}"
        )
    return EmbeddingTable(tokens=tokens, vectors=vectors)


def align_embeddings(
    trial_tokens: list[str] | np.ndarray,
    table: EmbeddingTable,
) -> np.ndarray:
    """Return (n_trials, dim) GloVe matrix for trial token order."""
    lookup = table.token_to_index
    rows = []
    for tok in trial_tokens:
        key = str(tok).lower()
        if key not in lookup:
            raise KeyError(f"Token {key!r} not found in embedding table")
        rows.append(table.vectors[lookup[key]])
    return np.asarray(rows, dtype=np.float64)
