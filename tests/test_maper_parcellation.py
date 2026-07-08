from __future__ import annotations

import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "pipeline"
    / "04_extract_labels"
    / "extract_maper_parcellation.py"
)
SPEC = importlib.util.spec_from_file_location("extract_maper_parcellation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

MANIFEST_SCRIPT = SCRIPT.with_name("build_extraction_manifest.py")
MANIFEST_SPEC = importlib.util.spec_from_file_location(
    "build_extraction_manifest", MANIFEST_SCRIPT)
MANIFEST_MODULE = importlib.util.module_from_spec(MANIFEST_SPEC)
assert MANIFEST_SPEC.loader is not None
MANIFEST_SPEC.loader.exec_module(MANIFEST_MODULE)


def point(name, tissue="GM", label_id=1, region6="", ap=""):
    valid = bool(label_id and tissue == "GM")
    return {
        "name": name,
        "tissue": tissue,
        "id": label_id,
        "valid": valid,
        "region6": region6,
        "ap": ap,
        "is_insula": bool(valid and label_id in MODULE.INSULA_IDS),
    }


def test_consensus_ignores_white_matter_but_retains_point_count():
    result = MODULE.consensus([
        point("WM", tissue="WM", label_id=0),
        point("insula anterior short gyrus L", label_id=86, region6="ASG", ap="Anterior"),
        point("insula middle short gyrus L", label_id=88, region6="MSG", ap="Anterior"),
    ])
    assert result["maper_roi"] == (
        "insula anterior short gyrus L–insula middle short gyrus L"
    )
    assert result["maper_mix"] is True
    assert result["maper_valid_points"] == 2
    assert result["maper_insula_points"] == 2
    assert result["maper_insula_status"] == "partial"
    assert result["maper_ap_consensus"] == "Anterior"
    assert result["maper_ap_mix"] is False


def test_consensus_single_insula_among_non_tissue():
    result = MODULE.consensus([
        point("WM", tissue="WM", label_id=0),
        point("insula posterior short gyrus L", label_id=90, region6="PSG", ap="Anterior"),
        point("WM", tissue="WM", label_id=0),
    ])
    assert result["maper_roi"] == "insula posterior short gyrus L"
    assert result["maper_mix"] is False
    assert result["maper_valid_points"] == 1
    assert result["maper_insula_points"] == 1
    assert result["maper_insula_status"] == "partial"


def test_consensus_cross_region_and_ap_boundary():
    cross_region = MODULE.consensus([
        point("FL inferior frontal gyrus L", label_id=56),
        point("insula posterior short gyrus L", label_id=90, region6="PSG", ap="Anterior"),
        point("WM", tissue="WM", label_id=0),
    ])
    assert cross_region["maper_roi"] == (
        "FL inferior frontal gyrus L–insula posterior short gyrus L"
    )
    assert cross_region["maper_mix"] is True

    ap_boundary = MODULE.consensus([
        point("insula posterior short gyrus L", label_id=90, region6="PSG", ap="Anterior"),
        point("insula anterior long gyrus L", label_id=94, region6="ALG", ap="Posterior"),
        point("insula posterior long gyrus L", label_id=20, region6="PLG", ap="Posterior"),
    ])
    assert ap_boundary["maper_insula_points"] == 3
    assert ap_boundary["maper_insula_status"] == "core"
    assert ap_boundary["maper_ap_consensus"] == "Anterior–Posterior"
    assert ap_boundary["maper_ap_mix"] is True

    reverse_boundary = MODULE.consensus([
        point("insula anterior long gyrus L", label_id=94, region6="ALG", ap="Posterior"),
        point("insula posterior short gyrus L", label_id=90, region6="PSG", ap="Anterior"),
        point("insula posterior long gyrus L", label_id=20, region6="PLG", ap="Posterior"),
    ])
    assert reverse_boundary["maper_ap_consensus"] == "Anterior–Posterior"


def test_consensus_all_non_tissue_and_agreement_labels():
    result = MODULE.consensus([
        point("WM", tissue="WM", label_id=0),
        point("CSF", tissue="CSF", label_id=0),
        point("Outside", tissue="Outside", label_id=0),
    ])
    assert result["maper_roi"] == "WM"
    assert result["maper_valid_points"] == 0
    assert result["maper_insula_points"] == 0
    assert result["maper_insula_status"] == "none"
    assert MODULE.agreement_label(True, 3) == "concordant_insula"
    assert MODULE.agreement_label(False, 3) == "maper_only"
    assert MODULE.agreement_label(True, 2) == "concordant_insula"
    assert MODULE.agreement_label(False, 1) == "maper_only"
    assert MODULE.agreement_label(True, 0) == "aparc_only"
    assert MODULE.agreement_label(False, 0) == "concordant_noninsula"


def test_consensus_all_csf_or_outside_falls_back_to_unknown():
    result = MODULE.consensus([
        point("CSF", tissue="CSF", label_id=0),
        point("Outside", tissue="Outside", label_id=0),
        point("Unclassified-GM", tissue="Unclassified-GM", label_id=0),
    ])
    assert result["maper_roi"] == "Unknown"
    assert result["maper_mix"] is False
    assert result["maper_insula_status"] == "none"


def test_tissue_overrides_for_callosum_ventricles_and_unclassified_gm():
    assert MODULE.tissue_name(44, 2) == "WM"
    assert MODULE.tissue_name(45, 2) == "CSF"
    assert MODULE.tissue_name(0, 2) == "Unclassified-GM"
    assert MODULE.tissue_name(86, 3) == "WM"


def test_coordinate_units_are_inferred_once_per_table():
    millimetres = np.array([[9.0, -1.0, 0.5], [30.0, 20.0, -15.0]])
    metres = millimetres / 1000.0
    assert MODULE.coordinate_scale_to_mm(millimetres) == 1.0
    assert MODULE.coordinate_scale_to_mm(metres) == 1000.0


def test_split_bipolar_name_matches_general_parcellation_rule():
    assert MODULE.split_bipolar_name("D0019_ROG1-2", "D0019") == ("ROG1", "ROG2")
    assert MODULE.split_bipolar_name("LA1-LB2", "D0019") == ("LA1", "LB2")


def test_manifest_prefers_canonical_parcellation_over_qc_variant(tmp_path):
    subject = "D0019"
    canonical = tmp_path / f"sub-{subject}_aparc2009s.csv"
    legacy = tmp_path / f"sub-{subject}_proc-3mm_aparc2009s.csv"
    canonical.touch()
    legacy.touch()
    assert MANIFEST_MODULE.choose_parcellation(tmp_path, subject) == canonical
    canonical.unlink()
    assert MANIFEST_MODULE.choose_parcellation(tmp_path, subject) == legacy


def test_sphere_statistics_use_whole_sphere_denominator():
    segmentation = np.zeros((7, 7, 7), dtype=int)
    tissue = np.full((7, 7, 7), 2, dtype=int)
    segmentation[3, 3, 3] = 86
    point_data = {"voxel": np.array([3.0, 3.0, 3.0])}
    offsets = MODULE.sphere_offsets(np.eye(4), 2.0)
    result = MODULE.sphere_summary(point_data, segmentation, tissue, offsets)
    assert result["sphere_total_voxels"] == 33
    assert result["sphere_insula_voxels"] == 1
    assert result["sphere_insula_fraction"] == 1 / 33
    assert result["sphere_winner_region6"] == "ASG"
    assert result["sphere_winner_fraction_within_insula"] == 1.0


def test_vote_summary_uses_all_30_atlases(tmp_path):
    labels = [86] * 20 + [94] * 5 + [0] * 5
    paths = []
    for index, label in enumerate(labels, start=1):
        path = tmp_path / f"a{index}.nii.gz"
        data = np.zeros((3, 3, 3), dtype=np.int16)
        data[1, 1, 1] = label
        nib.save(nib.Nifti1Image(data, np.eye(4)), path)
        paths.append(path)
    lut = {86: "ASG L", 94: "ALG L"}
    result = MODULE.vote_summaries(paths, [(1, 1, 1)], (3, 3, 3), lut)[0]
    assert result["insula_vote_fraction"] == 25 / 30
    assert result["anterior_vote_fraction"] == 20 / 30
    assert result["posterior_vote_fraction"] == 5 / 30
    assert result["winner_id"] == 86
    assert result["winner_vote_fraction"] == 20 / 30
