from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.hga.package_ave_cord import (
    _apply_aparc_roi_merge,
    build_coord_dataframe,
)
from src.hga.package_highgamma import parcellation_subset
from src.paths import hga_results_dir


class _FakeEpochs:
    def __init__(self, ch_names, tmin=-1.0, tmax=2.0, data=None):
        self.ch_names = list(ch_names)
        self.tmin = tmin
        self.tmax = tmax
        n_times = 10
        if data is None:
            data = np.ones((1, len(ch_names), n_times))
        self._data = data

    def crop(self, tmin, tmax):
        return self

    def get_data(self):
        return self._data


def test_build_coord_dataframe_uses_template_coords_and_significant():
    parc = pd.DataFrame(
        {
            "name": ["D0094_LAI1-2", "D0094_LFO1-2"],
            "center": ["insula", "ctx_lh_G_orbital"],
            "roi": ["AIC", "OFC"],
            "hemi": ["L", "L"],
            "mix": [False, False],
            "x_t": [-8.0, 1.0],
            "y_t": [30.0, 2.0],
            "z_t": [-27.0, 3.0],
        }
    )
    parc_sub = parcellation_subset(parc)
    epochs = _FakeEpochs(["D0094_LAI1-2", "D0094_LFO1-2"])

    df = build_coord_dataframe(
        epochs,
        parc_sub,
        {"D0094_LAI1-2"},
        band="highgamma",
        subject="D0094",
        task="LexicalDelay",
        description="Repeat",
        phase="Audio",
        modality="sound",
        atlas="hammers",
    )

    assert list(df["channel"]) == ["D0094_LAI1-2", "D0094_LFO1-2"]
    assert df.loc[0, "x"] == -8.0
    assert df.loc[0, "y"] == 30.0
    assert df.loc[0, "z"] == -27.0
    assert bool(df.loc[0, "significant"]) is True
    assert bool(df.loc[1, "significant"]) is False
    assert df.loc[0, "roi"] == "AIC"


def test_aparc_roi_merge_only_for_aparc_atlas():
    parc = pd.DataFrame(
        {
            "name": ["D0024_LOF1-2"],
            "center": ["ctx_lh_G_orbital"],
            "roi": ["PrG"],
            "hemi": ["L"],
            "x_t": [1.0],
            "y_t": [2.0],
            "z_t": [3.0],
        }
    )
    parc_sub = parcellation_subset(parc)
    epochs = _FakeEpochs(["D0024_LOF1-2"])

    hammers_df = build_coord_dataframe(
        epochs,
        parc_sub,
        set(),
        band="highgamma",
        subject="D0024",
        task="LexicalDelay",
        description="Repeat",
        phase="Audio",
        modality="sound",
        atlas="hammers",
    )
    aparc_df = build_coord_dataframe(
        epochs,
        parc_sub,
        set(),
        band="highgamma",
        subject="D0024",
        task="LexicalDelay",
        description="Repeat",
        phase="Audio",
        modality="sound",
        atlas="aparc2009s",
    )

    assert hammers_df.loc[0, "roi"] == "PrG"
    assert aparc_df.loc[0, "roi"] == "SMC"


def test_apply_aparc_roi_merge():
    df = pd.DataFrame({"roi": ["PrG", "PoG", "AIC"]})
    merged = _apply_aparc_roi_merge(df)
    assert list(merged["roi"]) == ["SMC", "SMC", "AIC"]


def test_coord_results_dir_has_atlas_suffix():
    path = hga_results_dir("LexicalDelay", "bipolar", "hammers")
    assert path.name == "LexicalDelay(bipolar)(hammers)"
