from __future__ import annotations

import numpy as np
import pandas as pd

from src.nmf.oaec_soft_embedding import (
    attach_soft_target_weights,
    collapse_insula_seeds,
    condition_network_embedding,
    prepare_oaec_edges,
    prepare_soft_projection,
    subject_embedding,
)


def test_prepare_projection_retains_mixed_weights_and_downweights_bad_fit():
    raw = pd.DataFrame(
        {
            "subject": ["D1", "D1", "D1"],
            "channel": ["x", "y", "insula"],
            "roi": ["STG", "MFG", "AIC"],
            "explained_energy": [0.5, 1.0, 1.0],
            "proportion_sensory": [0.34, 0.9, 1.0],
            "proportion_sustain": [0.33, 0.05, 0.0],
            "proportion_motor": [0.33, 0.05, 0.0],
            "best_component": ["sensory", "sensory", "sensory"],
            "dominance": [0.34, 0.9, 1.0],
            "in_discovery": [False, False, False],
        }
    )
    out = prepare_soft_projection(raw)
    assert out["channel"].tolist() == ["x", "y"]
    mixed = out.set_index("channel").loc["x"]
    assert np.isclose(mixed["soft_weight_sensory"], 0.17)
    assert np.isclose(mixed["soft_weight_sustain"], 0.165)
    assert np.isclose(mixed["soft_weight_motor"], 0.165)


def test_pipeline_uses_excess_null_and_collapses_seeds_before_soft_weighting():
    projection = prepare_soft_projection(
        pd.DataFrame(
            {
                "subject": ["D1", "D1"],
                "channel": ["t1", "t2"],
                "roi": ["STG", "PrG"],
                "explained_energy": [1.0, 1.0],
                "proportion_sensory": [0.8, 0.2],
                "proportion_sustain": [0.1, 0.2],
                "proportion_motor": [0.1, 0.6],
                "in_discovery": [False, False],
            }
        )
    )
    raw_edges = pd.DataFrame(
        {
            "source": ["s1", "s2", "s1", "s2"],
            "target": ["t1", "t1", "t2", "t2"],
            "stat": [0.4, 0.6, 0.2, 0.4],
            "null_mean": [0.3, 0.3, 0.1, 0.1],
            "subject": ["D1"] * 4,
            "phase": ["Delay"] * 4,
            "task": ["Task"] * 4,
            "dataset": ["Task"] * 4,
            "description": ["Repeat"] * 4,
            "metric": ["oaec"] * 4,
            "qc_pass": [True] * 4,
            "source_is_seed": [True] * 4,
            "target_is_seed": [False] * 4,
        }
    )
    edges = prepare_oaec_edges(raw_edges)
    attached = attach_soft_target_weights(edges, projection)
    targets = collapse_insula_seeds(attached)
    target_effect = targets.set_index("target_channel")["oaec_excess"]
    assert np.isclose(target_effect["t1"], 0.2)
    assert np.isclose(target_effect["t2"], 0.2)
    cells = condition_network_embedding(targets)
    assert np.allclose(cells["embedding"], 0.2)
    subjects = subject_embedding(cells, keep_phase=False)
    assert set(subjects["component"]) == {"sensory", "sustain", "motor"}


def test_prepare_oaec_edges_does_not_threshold_on_pair_p_value():
    pairs = pd.DataFrame(
        {
            "source": ["s1", "s1"],
            "target": ["t1", "t2"],
            "stat": [0.2, 0.4],
            "null_mean": [0.1, 0.1],
            "p_uncorrected": [0.9, 0.001],
            "subject": ["sub-D1", "sub-D1"],
            "phase": ["Stimulus", "Stimulus"],
            "description": ["Repeat", "Repeat"],
            "source_is_seed": [True, True],
            "target_is_seed": [False, False],
            "qc_pass": [True, True],
        }
    )
    out = prepare_oaec_edges(pairs)
    assert len(out) == 2
    assert out["subject"].unique().tolist() == ["D1"]
    assert np.allclose(out["oaec_excess"], [0.1, 0.3])
