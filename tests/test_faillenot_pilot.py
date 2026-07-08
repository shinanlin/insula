import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "src" / "faillenot_pilot.py"
SPEC = importlib.util.spec_from_file_location("faillenot_pilot", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_probability_summary_anterior():
    values = {key: 0.0 for key in MODULE.REGION_KEYS}
    values.update(asg=0.4, msg=0.2, plg=0.1)
    result = MODULE.probability_summary(values)
    assert result["label"] == "ASG"
    assert result["ap"] == "Anterior"
    assert np.isclose(result["p_anterior"], 0.6)
    assert np.isclose(result["p_posterior"], 0.1)


def test_probability_summary_unclassified():
    result = MODULE.probability_summary({key: 0.0 for key in MODULE.REGION_KEYS})
    assert result["label"] == "Unclassified"
    assert result["ap"] == "Unclassified"
    assert np.isnan(result["ap_anterior_fraction"])


def test_consensus_ignores_unclassified_and_preserves_mix():
    assert MODULE.consensus_ap(["Unclassified", "Anterior"]) == ("Anterior", False)
    assert MODULE.consensus_ap(["Anterior", "Posterior", "Anterior"]) == (
        "Anterior–Posterior", True)
    assert MODULE.consensus_ap(["Unclassified"]) == ("Unclassified", False)


def test_two_mm_sphere_offsets_are_physical():
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    offsets = MODULE.sphere_offsets(2.0, affine)
    distances = np.linalg.norm(offsets, axis=1)
    assert len(offsets) == 33
    assert distances.max() <= 2.0
    assert any(np.all(offset == 0) for offset in offsets)
