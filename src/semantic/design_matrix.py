"""Build trial-level design matrices for semantic ridge encoding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
from mne_bids import BIDSPath

from src.semantic.load_embeddings import (
    DEFAULT_EMBEDDING_DIR,
    EmbeddingTable,
    align_embeddings,
    load_embedding_table,
)
from src.semantic.load_stimulus_features import (
    DEFAULT_FEATURES_H5,
    StimulusFeatures,
    align_stimulus_features,
    load_stimulus_features,
)

DEFAULT_BIDS_ROOT = Path("/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS")


@dataclass(frozen=True)
class FeatureBlocks:
    """Token-aligned feature blocks for multi-block encoding."""

    semantic: np.ndarray  # (n_trials, n_sem)
    phon: np.ndarray  # (n_trials, n_phon)
    acoustic: np.ndarray  # (n_trials, n_acous)

    @property
    def block_names(self) -> tuple[str, ...]:
        return ("semantic", "phon", "acoustic")

    def get(self, name: str) -> np.ndarray:
        if name == "semantic":
            return self.semantic
        if name == "phon":
            return self.phon
        if name == "acoustic":
            return self.acoustic
        raise KeyError(f"Unknown feature block: {name!r}")


@dataclass(frozen=True)
class TrialDesign:
    """Trial-level encoding design for one subject × phase × description."""

    X: np.ndarray  # (n_trials, n_features) GloVe
    Y: np.ndarray  # (n_trials, n_channels, n_times)
    groups: np.ndarray  # (n_trials,) token labels for GroupKFold
    tokens: np.ndarray  # (n_trials,) token per trial
    ch_names: list[str]
    times: np.ndarray  # (n_times,)
    subject: str
    phase: str
    description: str
    tmin: float
    tmax: float

    @property
    def n_channels(self) -> int:
        return int(self.Y.shape[1])

    @property
    def n_times(self) -> int:
        return int(self.Y.shape[2])

    @property
    def n_unique_tokens(self) -> int:
        return int(len(np.unique(self.groups)))


@dataclass(frozen=True)
class MultiBlockTrialDesign:
    """Trial-level design with separate semantic / phon / acoustic blocks."""

    blocks: FeatureBlocks
    Y: np.ndarray  # (n_trials, n_channels, n_times)
    groups: np.ndarray
    tokens: np.ndarray
    ch_names: list[str]
    times: np.ndarray
    subject: str
    phase: str
    description: str
    tmin: float
    tmax: float

    @property
    def n_channels(self) -> int:
        return int(self.Y.shape[1])

    @property
    def n_times(self) -> int:
        return int(self.Y.shape[2])

    @property
    def n_unique_tokens(self) -> int:
        return int(len(np.unique(self.groups)))

    @property
    def X(self) -> np.ndarray:
        """Concatenated full design (sem | phon | acous)."""
        return np.hstack(
            [self.blocks.semantic, self.blocks.phon, self.blocks.acoustic]
        )

    def block_slices(self) -> dict[str, slice]:
        n_sem = self.blocks.semantic.shape[1]
        n_phon = self.blocks.phon.shape[1]
        n_acous = self.blocks.acoustic.shape[1]
        i0 = 0
        i1 = n_sem
        i2 = i1 + n_phon
        i3 = i2 + n_acous
        return {
            "semantic": slice(i0, i1),
            "phon": slice(i1, i2),
            "acoustic": slice(i2, i3),
        }


def parse_condition(name: str) -> dict[str, str]:
    """Parse BIDS condition string into lexicality, token, remark."""
    parts = str(name).split("/")
    if len(parts) < 4:
        raise ValueError(f"Cannot parse condition: {name!r}")
    return {
        "description": parts[1],
        "lexicality": parts[2],
        "token": parts[3].lower(),
        "remark": parts[4] if len(parts) > 4 else "",
    }


def is_word_correct(meta: dict[str, str]) -> bool:
    return meta["lexicality"] == "Word" and meta["remark"] == "CORRECT"


def vectorize_y(y: np.ndarray) -> np.ndarray:
    """(n_trials, n_channels, n_times) -> (n_trials, n_channels * n_times)."""
    if y.ndim != 3:
        raise ValueError(f"Y must be 3-D, got shape {y.shape}")
    n_trials, n_channels, n_times = y.shape
    return y.reshape(n_trials, n_channels * n_times)


def reshape_r(
    r_flat: np.ndarray,
    n_channels: int,
    n_times: int,
) -> np.ndarray:
    """(n_channels * n_times,) -> (n_channels, n_times)."""
    r_flat = np.asarray(r_flat, dtype=float).reshape(-1)
    expected = n_channels * n_times
    if r_flat.size != expected:
        raise ValueError(f"Expected {expected} values, got {r_flat.size}")
    return r_flat.reshape(n_channels, n_times)


def _load_word_correct_epochs(
    subject: str,
    phase: str,
    description: str,
    tmin: float,
    tmax: float,
    bids_root: Path,
    band: str,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Return Y, times, ch_names, tokens, groups for Word/CORRECT trials."""
    subject = subject.replace("sub-", "")
    bids_path = BIDSPath(
        root=str(bids_root / "derivatives" / "epoch(bipolar)"),
        subject=subject,
        suffix=band,
        description=description,
        processing=phase,
        datatype="epoch(band)(sig)(effective)",
        extension=".h5",
        check=False,
    )
    matches = bids_path.match()
    if not matches:
        raise FileNotFoundError(
            f"No epoch file for sub-{subject} phase={phase} desc={description}"
        )

    epochs = mne.read_epochs(matches[0], preload=True, verbose="error")
    epochs.crop(tmin=tmin, tmax=tmax)

    id_to_name = {v: k for k, v in epochs.event_id.items()}
    keep_idx = []
    tokens = []
    groups = []

    for i, code in enumerate(epochs.events[:, 2]):
        name = id_to_name[int(code)]
        meta = parse_condition(name)
        if not is_word_correct(meta):
            continue
        keep_idx.append(i)
        tokens.append(meta["token"])
        groups.append(meta["token"])

    if not keep_idx:
        raise RuntimeError(
            f"No Word/CORRECT trials for sub-{subject} {phase}/{description}"
        )

    epochs = epochs[keep_idx]
    return (
        epochs.get_data(),
        epochs.times,
        list(epochs.ch_names),
        np.asarray(tokens, dtype=object),
        np.asarray(groups, dtype=object),
    )


def load_trial_design(
    subject: str,
    phase: str = "Delay",
    description: str = "Decision",
    tmin: float = -0.5,
    tmax: float = 1.0,
    bids_root: Path | str = DEFAULT_BIDS_ROOT,
    embedding_dir: Path | str = DEFAULT_EMBEDDING_DIR,
    band: str = "highgamma",
    embedding_table: EmbeddingTable | None = None,
) -> TrialDesign:
    """Load epochs, filter Word/CORRECT trials, crop, and align GloVe features."""
    subject = subject.replace("sub-", "")
    bids_root = Path(bids_root)
    Y, times, ch_names, tokens_arr, groups_arr = _load_word_correct_epochs(
        subject, phase, description, tmin, tmax, bids_root, band
    )
    table = embedding_table or load_embedding_table(embedding_dir)
    X = align_embeddings(tokens_arr, table)

    return TrialDesign(
        X=X,
        Y=Y,
        groups=groups_arr,
        tokens=tokens_arr,
        ch_names=ch_names,
        times=times,
        subject=subject,
        phase=phase,
        description=description,
        tmin=tmin,
        tmax=tmax,
    )


def load_trial_design_multi(
    subject: str,
    phase: str = "Delay",
    description: str = "Decision",
    tmin: float = -0.5,
    tmax: float = 1.0,
    bids_root: Path | str = DEFAULT_BIDS_ROOT,
    embedding_dir: Path | str = DEFAULT_EMBEDDING_DIR,
    features_h5: Path | str = DEFAULT_FEATURES_H5,
    band: str = "highgamma",
    embedding_table: EmbeddingTable | None = None,
    stimulus_features: StimulusFeatures | None = None,
) -> MultiBlockTrialDesign:
    """Load Word/CORRECT epochs with semantic, phonology, and acoustic blocks."""
    subject = subject.replace("sub-", "")
    bids_root = Path(bids_root)
    Y, times, ch_names, tokens_arr, groups_arr = _load_word_correct_epochs(
        subject, phase, description, tmin, tmax, bids_root, band
    )
    table = embedding_table or load_embedding_table(embedding_dir)
    X_sem = align_embeddings(tokens_arr, table)
    feat = stimulus_features or load_stimulus_features(features_h5)
    X_phon, X_acous = align_stimulus_features(tokens_arr, features=feat)

    return MultiBlockTrialDesign(
        blocks=FeatureBlocks(
            semantic=np.asarray(X_sem, dtype=np.float64),
            phon=np.asarray(X_phon, dtype=np.float64),
            acoustic=np.asarray(X_acous, dtype=np.float64),
        ),
        Y=Y,
        groups=groups_arr,
        tokens=tokens_arr,
        ch_names=ch_names,
        times=times,
        subject=subject,
        phase=phase,
        description=description,
        tmin=tmin,
        tmax=tmax,
    )
