from __future__ import annotations

import pandas as pd
import pytest

from src.hga.package_highgamma import load_parcellation, parcellation_subset, results_dir


def test_parcellation_subset_new_schema_maps_template_native_and_endpoints():
    parc = pd.DataFrame(
        {
            "name": ["D0094_LFO1-2"],
            "center": ["ctx_lh_G_orbital"],
            "roi": ["OFC"],
            "hemi": ["L"],
            "x": [-5.0],
            "y": [40.0],
            "z": [38.0],
            "x1": [-5.5],
            "y1": [39.5],
            "z1": [37.5],
            "x2": [-4.5],
            "y2": [40.5],
            "z2": [38.5],
            "contact_1": ["LFO1"],
            "contact_1_label": ["ctx_lh_G_orbital"],
            "contact_2": ["LFO2"],
            "contact_2_label": ["Left-Cerebral-White-Matter"],
            "x_t": [-8.0],
            "y_t": [30.0],
            "z_t": [-27.0],
            "x1_t": [-8.5],
            "y1_t": [29.5],
            "z1_t": [-27.5],
            "x2_t": [-7.5],
            "y2_t": [30.5],
            "z2_t": [-26.5],
        }
    )

    subset = parcellation_subset(parc)

    assert subset.loc[0, "channel"] == "D0094_LFO1-2"
    assert subset.loc[0, "label"] == "ctx_lh_G_orbital"
    assert subset.loc[0, "x"] == -8.0
    assert subset.loc[0, "x_native"] == -5.0
    assert subset.loc[0, "x1_native"] == -5.5
    assert subset.loc[0, "x1_template"] == -8.5
    assert subset.loc[0, "contact_1"] == "LFO1"
    assert subset.loc[0, "contact_2_label"] == "Left-Cerebral-White-Matter"


def test_parcellation_subset_legacy_schema_keeps_midpoint_and_nan_endpoints():
    parc = pd.DataFrame(
        {
            "name": ["D0024_LOF1-2"],
            "center": ["ctx_lh_G_orbital"],
            "roi": ["OFC"],
            "hemi": ["L"],
            "x": [-8.0],
            "y": [30.0],
            "z": [-27.0],
        }
    )

    subset = parcellation_subset(parc)

    assert subset.loc[0, "x_native"] == -8.0
    assert pd.isna(subset.loc[0, "x"])
    assert pd.isna(subset.loc[0, "x1_native"])
    assert pd.isna(subset.loc[0, "x1_template"])
    assert pd.isna(subset.loc[0, "contact_1"])


def test_parcellation_subset_preserves_mix_when_present():
    parc = pd.DataFrame(
        {
            "name": ["D0035_LAI1-2"],
            "center": ["insula"],
            "roi": ["PIC–AIC"],
            "hemi": ["L"],
            "mix": [True],
            "x_t": [1.0],
            "y_t": [2.0],
            "z_t": [3.0],
        }
    )
    subset = parcellation_subset(parc)
    assert bool(subset.loc[0, "mix"])


def test_results_dir_atlas_suffix():
    aparc = results_dir("LexicalDelay", "bipolar", "aparc2009s")
    hammers = results_dir("LexicalDelay", "bipolar", "hammers")
    assert aparc.name == "LexicalDelay(bipolar)(aparc2009s)"
    assert hammers.name == "LexicalDelay(bipolar)(hammers)"


def test_load_parcellation_uses_atlas_suffix(tmp_path):
    aparc_csv = tmp_path / "sub-D0079_aparc2009s.csv"
    hammers_csv = tmp_path / "sub-D0079_hammers.csv"
    aparc_csv.write_text("name,center,roi,hemi\n")
    hammers_csv.write_text("name,center,roi,hemi,mix\n")

    class _FakePath:
        subject = "D0079"
        root = str(tmp_path / "epoch(bipolar)")

        def copy(self):
            return self

        def update(self, **kwargs):
            self._kwargs = kwargs
            return self

        def match(self):
            suffix = self._kwargs.get("suffix")
            if suffix == "aparc2009s":
                return [str(aparc_csv)]
            if suffix == "hammers":
                return [str(hammers_csv)]
            return []

    assert load_parcellation(_FakePath(), "bipolar", "aparc2009s").empty
    assert load_parcellation(_FakePath(), "bipolar", "hammers").empty


def test_load_parcellation_invalid_atlas():
    class _FakePath:
        def copy(self):
            return self

    with pytest.raises(ValueError, match="atlas must be one of"):
        load_parcellation(_FakePath(), "bipolar", "invalid")
