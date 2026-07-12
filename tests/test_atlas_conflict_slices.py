import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).parents[1] / "pipeline" / "05_visual_qc" / "plot_atlas_conflict_slices.py"
SPEC = importlib.util.spec_from_file_location("plot_atlas_conflict_slices", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_pure_conflict_row_excludes_aparc_mixed_and_maper_partial():
    assert MODULE.pure_conflict_row(pd.Series({
        "maper_atlas_agreement": "aparc_only",
        "roi": "INS",
        "maper_insula_status": "none",
    }))
    assert not MODULE.pure_conflict_row(pd.Series({
        "maper_atlas_agreement": "aparc_only",
        "roi": "INS–IFG",
        "maper_insula_status": "none",
    }))
    assert MODULE.pure_conflict_row(pd.Series({
        "maper_atlas_agreement": "maper_only",
        "roi": "IFG",
        "maper_insula_status": "core",
    }))
    assert not MODULE.pure_conflict_row(pd.Series({
        "maper_atlas_agreement": "maper_only",
        "roi": "IFG",
        "maper_insula_status": "partial",
    }))


def test_pure_conflict_row_accepts_concordant_core_insula():
    assert MODULE.pure_conflict_row(pd.Series({
        "maper_atlas_agreement": "concordant_insula",
        "roi": "INS",
        "maper_insula_status": "core",
    }))
    assert MODULE.pure_conflict_row(pd.Series({
        "maper_atlas_agreement": "concordant_insula",
        "roi": "INS–STG",
        "maper_insula_status": "core",
    }))
    assert not MODULE.pure_conflict_row(pd.Series({
        "maper_atlas_agreement": "concordant_insula",
        "roi": "INS",
        "maper_insula_status": "partial",
    }))
    assert not MODULE.pure_conflict_row(pd.Series({
        "maper_atlas_agreement": "concordant_insula",
        "roi": "STG",
        "maper_insula_status": "core",
    }))


def _row(task="TaskA", agreement="aparc_only", x=1.0):
    return {
        "subject": "D0001",
        "name": "D0001_A1-2",
        "task": task,
        "maper_atlas_agreement": agreement,
        "roi": "INS",
        "center": "ctx_lh_G_insular_short",
        "maper_insula_status": "none" if agreement == "aparc_only" else "core",
        "maper_region6_consensus": "" if agreement == "aparc_only" else "ASG",
        "maper_ap_consensus": "" if agreement == "aparc_only" else "Anterior",
        "maper_center_winner_vote_fraction": 1.0,
        "contact_1_clean": "A1",
        "contact_2_clean": "A2",
        "contact_1_x": 0.0,
        "contact_1_y": 0.0,
        "contact_1_z": 0.0,
        "center_mm_x": x,
        "center_mm_y": 0.0,
        "center_mm_z": 0.0,
        "contact_2_x": 2.0,
        "contact_2_y": 0.0,
        "contact_2_z": 0.0,
        "orig": "/tmp/orig.mgz",
        "fused": "/tmp/fused.nii.gz",
    }


def test_build_qc_cases_deduplicates_matching_task_rows():
    rows = pd.DataFrame([_row("TaskA"), _row("TaskB")])
    cases = MODULE.build_qc_cases(rows)
    assert len(cases) == 1
    assert cases[0].tasks == ("TaskA", "TaskB")
    assert cases[0].warning == ""


def test_build_qc_cases_splits_task_specific_coordinate_mismatch():
    rows = pd.DataFrame([_row("TaskA", x=1.0), _row("TaskB", x=1.5)])
    cases = MODULE.build_qc_cases(rows)
    assert len(cases) == 2
    assert {case.tasks for case in cases} == {("TaskA",), ("TaskB",)}
    assert all(case.warning == "task_specific_coordinate_or_label_mismatch" for case in cases)


def test_view_limits_stay_inside_image_bounds():
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    xlim, ylim = MODULE.view_limits(np.array([2.0, 2.0, 2.0]), (20, 20, 20), affine, (0, 1), 80.0)
    assert xlim == (-0.5, 19.5)
    assert ylim == (-0.5, 19.5)


def test_lut_insula_ids_are_selected_by_label_name(tmp_path):
    lut = tmp_path / "FreeSurferColorLUT.txt"
    lut.write_text(
        "1 Left-Cerebral-White-Matter 0 0 0 0\n"
        "11117 ctx_lh_G_insular_short 0 0 0 0\n"
        "12150 ctx_rh_S_circular_insula_sup 0 0 0 0\n"
    )
    assert MODULE.load_freesurfer_insula_ids(lut) == frozenset({11117, 12150})


def test_lut_circular_insula_ids_are_selected_separately(tmp_path):
    lut = tmp_path / "FreeSurferColorLUT.txt"
    lut.write_text(
        "11117 ctx_lh_G_insular_short 0 0 0 0\n"
        "11148 ctx_lh_S_circular_insula_ant 0 0 0 0\n"
        "12149 ctx_rh_S_circular_insula_inf 0 0 0 0\n"
        "11112 ctx_lh_G_front_inf-Opercular 0 0 0 0\n"
    )
    assert MODULE.load_freesurfer_circular_insula_ids(lut) == frozenset({11148, 12149})
