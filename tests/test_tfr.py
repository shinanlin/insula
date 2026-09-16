"""Insula TFR workspace paths and per-electrode figures (SHI-87)."""
import numpy as np

from src.tfr.plot import plot_channel_phases
from src.tfr.run import (
    channel_fig_path,
    select_insula_channels,
    subset_epochs,
    tfr_paths,
)


def test_tfr_workspace_layout():
    paths = tfr_paths("D0088", "PhonemeSequence", "Stimulus", "Repeat")
    assert paths["tfr"].as_posix().endswith(
        "results/tfr/PhonemeSequence/sub-D0088/tfr/"
        "sub-D0088_task-PhonemeSequence_proc-Stimulus_desc-Repeat_tfr.h5"
    )
    assert paths["stats"].as_posix().endswith(
        "results/tfr/PhonemeSequence/sub-D0088/tfr/"
        "sub-D0088_task-PhonemeSequence_proc-Stimulus_desc-Repeat_stats.h5"
    )
    fig = channel_fig_path("D0088", "PhonemeSequence", "D0088_LAI3-4")
    assert fig.as_posix().endswith(
        "img/tfr/PhonemeSequence/sub-D0088/"
        "sub-D0088_task-PhonemeSequence_desc-Repeat_chan-D0088_LAI3-4_tfr.svg"
    )
    assert "BIDS" not in paths["tfr"].as_posix()
    assert "derivatives/tfr" not in paths["tfr"].as_posix()


def test_smoke_output_root_is_separate():
    paths = tfr_paths(
        "D0088",
        "PhonemeSequence",
        "Stimulus",
        output_root="/tmp/tfr_smoke",
    )
    assert "/tmp/tfr_smoke/PhonemeSequence/" in paths["tfr"].as_posix()
    assert "results/tfr/PhonemeSequence" not in paths["tfr"].as_posix()


def test_select_insula_channels_preserves_anatomical_order():
    anatomical = ["D1_A1-2", "D1_B1-2", "D1_C1-2"]
    assert select_insula_channels(["D1_C1-2", "D1_X", "D1_A1-2"], anatomical) == [
        "D1_A1-2",
        "D1_C1-2",
    ]


class _FakeEpochs:
    def __init__(self, names, preload=False):
        self.ch_names = list(names)
        self.preload = preload

    def load_data(self):
        self.preload = True
        return self

    def pick(self, names):
        if not self.preload:
            raise RuntimeError(
                "adding, dropping, or reordering channels requires epochs data to be loaded"
            )
        self.ch_names = list(names)
        return self


def test_subset_epochs_loads_before_pick():
    epochs = _FakeEpochs(["D1_C1-2", "D1_X", "D1_A1-2"])
    out = subset_epochs(epochs, ["D1_A1-2", "D1_B1-2", "D1_C1-2"])
    assert out is epochs
    assert epochs.preload
    assert epochs.ch_names == ["D1_A1-2", "D1_C1-2"]


def test_subset_epochs_returns_none_when_no_insula_channels():
    assert subset_epochs(_FakeEpochs(["D1_X"]), ["D1_A1-2"]) is None


def test_n_jobs_is_never_negative():
    from src.tfr.run import positive_jobs

    assert positive_jobs(-1) == 1
    assert positive_jobs(0) == 1
    assert positive_jobs(8) == 8


def test_cluster_is_serial():
    from src.tfr.run import CLUSTER_N_JOBS

    assert CLUSTER_N_JOBS == 1


def test_comp_by_sort_matches_numpy_1_26():
    from src.tfr.run import _comp_by_sort

    rng = np.random.default_rng(0)
    diff = rng.normal(size=(6, 4, 5))
    out = _comp_by_sort(diff, axis=0)
    assert out.shape == diff.shape
    assert np.all((out >= 0) & (out <= 1))


def _synthetic_panels():
    rng = np.random.default_rng(0)
    times = np.linspace(-0.5, 1.5, 40)
    freqs = np.arange(4, 40, 3)
    panels = {}
    for i, phase in enumerate(("Stimulus", "Delay", "Go", "Response")):
        data = rng.normal(size=(len(freqs), len(times)))
        mask = np.zeros_like(data)
        pvals = np.ones_like(data)
        if i == 0:
            mask[3:8, 10:20] = 1
            pvals[3:8, 10:20] = 0.01
        panels[phase] = {"times": times, "freqs": freqs, "data": data, "mask": mask, "pvals": pvals}
    return panels


def test_channel_phases_svg_has_four_alignments(tmp_path):
    dest = tmp_path / "sub-D0088_task-PhonemeSequence_desc-Repeat_chan-D0088_LAI3-4_tfr.svg"
    plot_channel_phases(
        _synthetic_panels(),
        dest,
        subject="D0088",
        channel="D0088_LAI3-4",
        task="PhonemeSequence",
        cluster="Sensory",
    )
    text = dest.read_text()
    assert dest.is_file() and dest.stat().st_size > 500
    for label in ("Stimulus", "Delay", "Go", "Response", "LAI3-4", "Sensory"):
        assert label in text


def test_channel_phases_can_drop_go(tmp_path):
    dest = tmp_path / "no_go.svg"
    panels = _synthetic_panels()
    plot_channel_phases(
        panels,
        dest,
        subject="D0088",
        channel="D0088_LAI3-4",
        task="PhonemeSequence",
        phases=("Stimulus", "Delay", "Response"),
    )
    text = dest.read_text()
    assert "Stimulus" in text and "Delay" in text and "Response" in text
    assert ">Go<" not in text


def test_shared_channels_keeps_stimulus_order():
    from src.tfr.run import shared_channels

    products = {
        "Stimulus": {"ch_names": ["A", "B", "C"]},
        "Delay": {"ch_names": ["C", "A"]},
        "Go": {"ch_names": ["A", "C", "D"]},
        "Response": {"ch_names": ["C", "A"]},
    }
    assert shared_channels(products) == ["A", "C"]
