"""Unit tests for concat-NMF waveform PCA helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from src.nmf.waveform_pca import (
    align_matrix_to_assignments,
    fit_waveform_pca,
    plot_waveform_pca,
    write_pca_outputs,
)


def _block_waveforms(n_per_block: int = 12, n_features: int = 24, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_features)
    templates = [
        np.exp(-((t - 0.2) / 0.08) ** 2),
        np.exp(-((t - 0.5) / 0.12) ** 2),
        np.exp(-((t - 0.8) / 0.10) ** 2),
    ]
    names = ("sustain", "motor", "sensory")
    rows = []
    labels = []
    channels = []
    for c, tmpl in enumerate(templates):
        for i in range(n_per_block):
            row = np.clip(tmpl + 0.05 * rng.random(n_features), 0, None)
            row = row / np.linalg.norm(row)
            rows.append(row)
            labels.append(names[c])
            channels.append(f"ch_{names[c]}_{i}")
    return np.asarray(rows), np.asarray(labels), np.asarray(channels)


def test_align_matrix_to_assignments_keeps_shared_order():
    meta = pd.DataFrame({"channel": ["a", "b", "c", "d"]})
    assignments = pd.DataFrame(
        {
            "channel": ["d", "b", "z"],
            "functional_cluster": ["sensory", "motor", "sustain"],
        }
    )
    idx, aligned = align_matrix_to_assignments(meta, assignments)
    np.testing.assert_array_equal(idx, [1, 3])
    assert aligned["channel"].tolist() == ["b", "d"]
    assert aligned["functional_cluster"].tolist() == ["motor", "sensory"]


def test_fit_waveform_pca_separates_block_waveforms():
    X, labels, _channels = _block_waveforms()
    scores, explained = fit_waveform_pca(X, n_components=3)
    assert scores.shape == (len(X), 3)
    assert explained.shape == (3,)
    assert explained[0] > explained[1] > 0
    sil = silhouette_score(scores[:, :2], labels, metric="euclidean")
    assert sil > 0.5


def test_plot_waveform_pca_writes_svg(tmp_path: Path):
    X, labels, _channels = _block_waveforms(n_per_block=6)
    scores, explained = fit_waveform_pca(X, n_components=3)
    path = tmp_path / "waveform_pca.svg"
    plot_waveform_pca(scores, labels, explained, path=path)
    assert path.is_file()
    assert path.read_text(encoding="utf-8").lstrip().startswith("<?xml")


def test_write_pca_outputs_includes_scores(tmp_path: Path):
    _X, labels, channels = _block_waveforms(n_per_block=3)
    aligned = pd.DataFrame(
        {
            "channel": channels,
            "subject": ["S1"] * len(channels),
            "functional_cluster": labels,
            "dominance": np.linspace(0.4, 0.9, len(channels)),
        }
    )
    scores = np.arange(len(channels) * 3, dtype=float).reshape(len(channels), 3)
    explained = np.array([0.5, 0.3, 0.1])
    scores_path = tmp_path / "scores.csv"
    meta_path = tmp_path / "meta.json"
    write_pca_outputs(
        aligned=aligned,
        scores=scores,
        explained=explained,
        meta={"n_shared": int(len(aligned))},
        scores_path=scores_path,
        meta_path=meta_path,
    )
    frame = pd.read_csv(scores_path)
    assert {"channel", "subject", "functional_cluster", "PC1", "PC2", "PC3"} <= set(
        frame.columns
    )
    assert len(frame) == len(aligned)
    payload = meta_path.read_text(encoding="utf-8")
    assert "0.5" in payload
