from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.univariate import viz_mean as vm


def test_phase_processing_roundtrip():
    assert vm._phase_processing("delay") == "Delay"
    assert vm._phase_processing("Response") == "Response"
    with pytest.raises(ValueError):
        vm._phase_processing("unknown")


def test_task_results_dir_sibling_hammers_layout():
    root = Path("/tmp/results")
    path = vm.task_results_dir(root, "LexicalDelay")
    assert path == root / "LexicalDelay(bipolar)(hammers)"


def test_task_results_dir_packaged_local():
    from src.paths import RESULTS_ROOT, hga_results_dir

    assert vm.task_results_dir(RESULTS_ROOT, "LexicalDelay") == hga_results_dir(
        "LexicalDelay"
    )


def test_has_mean_contrasts_false_for_empty_tree(tmp_path: Path):
    assert not vm.has_mean_contrasts(tmp_path)


def test_resolve_mean_results_root_falls_back_to_sibling():
    root = vm.resolve_mean_results_root(vm.RESULTS_ROOT)
    assert root == vm.FALLBACK_MEAN_RESULTS
    assert vm.has_mean_contrasts(root)


def test_electrode_point_sizes_scale_with_pattern():
    from src.decoding.viz_insula_patterns import SIG_SIZE, _electrode_point_sizes

    spatial = pd.DataFrame(
        {
            "significant": [True, True, True],
            "pattern": [0.1, 0.5, 1.0],
        }
    )
    side = spatial.copy()
    fixed = _electrode_point_sizes(
        side, spatial, size_by_pattern=False, size_range=(8.0, 64.0)
    )
    assert fixed == [SIG_SIZE, SIG_SIZE, SIG_SIZE]
    linear = _electrode_point_sizes(
        side, spatial, size_by_pattern=True, size_range=(8.0, 64.0), size_gamma=1.0
    )
    scaled = _electrode_point_sizes(
        side, spatial, size_by_pattern=True, size_range=(8.0, 64.0), size_gamma=2.0
    )
    assert scaled[0] < scaled[1] < scaled[2]
    assert scaled[0] >= 8.0
    assert scaled[2] <= 64.0
    # gamma>1 pulls small effects closer to vmin than linear
    assert scaled[0] < linear[0]


def test_attach_metadata_and_filter_qc():
    contrasts = pd.DataFrame(
        {
            "channel": ["D0001_A1-2", "D0001_B1-2", "D0001_C1-2"],
            "significant": [True, True, True],
            "mean_diff": [0.5, -0.2, 0.1],
            "contrast": ["DecisionVsRepeatMean"] * 3,
            "phase": ["delay"] * 3,
            "task": ["LexicalDelay"] * 3,
        }
    )
    coords = pd.DataFrame(
        {
            "channel": ["D0001_A1-2", "D0001_B1-2", "D0001_C1-2"],
            "roi": ["AIC", "Thal", "STG"],
            "hemi": ["L", "R", "L"],
            "label": ["insula anterior pole L", 0, "ctx_lh_G_temp_sup"],
            "x": [-10.0, 10.0, -20.0],
            "y": [5.0, 5.0, 40.0],
            "z": [2.0, 2.0, 10.0],
            "mix": [False, False, True],
        }
    )
    merged = vm.attach_metadata(contrasts, coords)
    filtered = vm.filter_qc(merged)
    assert set(filtered["channel"]) == {"D0001_A1-2"}


def test_select_significant_direction_labels():
    df = pd.DataFrame(
        {
            "significant": [True, True, False],
            "phase": ["delay", "delay", "delay"],
            "contrast": ["DecisionVsRepeatMean"] * 3,
            "task": ["LexicalDelay"] * 3,
            "mean_diff": [0.3, -0.1, 0.2],
            "channel": ["a", "b", "c"],
        }
    )
    sig = vm.select_significant(df, phase="delay", contrast="DecisionVsRepeatMean")
    assert len(sig) == 2
    assert set(sig["direction"]) == {"Decision", "Repeat"}


def test_select_significant_repeat_vs_passive_direction():
    df = pd.DataFrame(
        {
            "significant": [True, True],
            "phase": ["stimulus", "stimulus"],
            "contrast": ["RepeatVsPassiveMean", "RepeatVsPassiveMean"],
            "task": ["LexicalNoDelay"] * 2,
            "mean_diff": [0.4, -0.2],
            "channel": ["a", "b"],
        }
    )
    sig = vm.select_significant(df, contrast="RepeatVsPassiveMean")
    assert list(sig["direction"]) == ["Repeat", "Passive"]


def test_select_significant_word_nonword_direction():
    df = pd.DataFrame(
        {
            "significant": [True, True],
            "phase": ["response", "response"],
            "contrast": ["WordVsNonwordDecisionMean", "WordVsNonwordDecisionMean"],
            "task": ["LexicalNoDelay"] * 2,
            "mean_diff": [0.2, -0.4],
            "channel": ["a", "b"],
        }
    )
    sig = vm.select_significant(df, contrast="WordVsNonwordDecisionMean")
    assert list(sig["direction"]) == ["Word", "Nonword"]


def test_filter_insula_electrodes():
    df = pd.DataFrame({"roi": ["AIC", "PIC", "STG", "IFG"], "channel": list("abcd")})
    insula = vm.filter_insula_electrodes(df)
    assert set(insula["channel"]) == {"a", "b"}


def test_roi_counts():
    df = pd.DataFrame({"roi": ["AIC", "AIC", "STG"], "channel": ["a", "b", "c"]})
    counts = vm.roi_counts(df)
    assert counts["AIC"] == 2
    assert counts["STG"] == 1


def test_discover_mean_paths_on_repo_results():
    root = Path(__file__).resolve().parent.parent / "results"
    paths = vm.discover_mean_paths(
        root,
        "LexicalDelay",
        "delay",
        "DecisionVsRepeatMean",
    )
    if not paths:
        pytest.skip("Mean univariate outputs not present in workspace")
    assert all(p.description == "DecisionVsRepeatMean" for p in paths)
    assert all(p.processing == "Delay" for p in paths)


def test_plot_signed_electrodes_brain_empty():
    import matplotlib

    matplotlib.use("Agg")
    fig = vm.plot_signed_electrodes_brain(pd.DataFrame(), title="empty")
    assert fig is not None
