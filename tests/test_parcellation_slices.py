import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO = Path(__file__).parents[1]
PLOT_PATH = REPO / "pipeline" / "05_visual_qc" / "plot_parcellation_slices.py"
RENDER_PATH = REPO / "pipeline" / "05_visual_qc" / "slice_render.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PLOT = _load_module(PLOT_PATH, "plot_parcellation_slices")
RENDER = _load_module(RENDER_PATH, "slice_render")


def _row(roi="AIC", mix=False):
    return {
        "subject": "D0094",
        "name": "D0094_LAI1-2",
        "roi": roi,
        "mix": mix,
        "center": "ctx_lh_G_insular_short",
        "contact_1": "LAI1",
        "contact_2": "LAI2",
        "x": 1.0, "y": 2.0, "z": 3.0,
        "x1": 0.0, "y1": 2.0, "z1": 3.0,
        "x2": 2.0, "y2": 2.0, "z2": 3.0,
    }


def test_insula_row_mask_hammers_pure_aic_pic_only():
    df = pd.DataFrame([
        _row("AIC", False),
        _row("PIC", False),
        _row("AIC", True),
        _row("PIC–AIC", False),
        _row("IFG", False),
    ])
    mask = PLOT.insula_row_mask(df, "hammers")
    assert mask.tolist() == [True, True, False, False, False]


def test_insula_row_mask_aparc_pure_ins_only():
    df = pd.DataFrame([
        _row("INS", False),
        _row("Insula", False),
        _row("INS", True),
        _row("INS–IFG", False),
        _row("IFG", False),
    ])
    mask = PLOT.insula_row_mask(df, "aparc2009s")
    assert mask.tolist() == [True, True, False, False, False]


def test_recon_subject_id_strips_leading_zeros():
    assert PLOT.recon_subject_id("D0023") == "D23"
    assert PLOT.recon_subject_id("D0103") == "D103"


def test_view_limits_stay_inside_image_bounds():
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    xlim, ylim = RENDER.view_limits(
        np.array([2.0, 2.0, 2.0]), (20, 20, 20), affine, (0, 1), 80.0,
    )
    assert xlim == (-0.5, 19.5)
    assert ylim == (-0.5, 19.5)


def test_parcellation_qc_dir():
    sys.path.insert(0, str(REPO))
    from src.paths import parcellation_qc_dir

    path = parcellation_qc_dir("hammers", "D0094")
    assert path.as_posix().endswith("results/qc/hammers/sub-D0094")
