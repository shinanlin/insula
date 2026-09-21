"""Tests for subject×modality sig-union helpers in package_highgamma."""

from __future__ import annotations

from types import SimpleNamespace

from src.hga.package_highgamma import (
    pick_union_channels,
    sig_union_by_subject_modality,
)


class _FakeEpochs:
    def __init__(self, ch_names: list[str]):
        self.ch_names = ch_names


def _sig_path(subject: str, modality: str | None, channels: list[str]):
    return SimpleNamespace(
        subject=subject,
        recording=modality,
        description="Repeat",
        processing="Stimulus",
        ch_names=channels,
    )


def test_sig_union_merges_channels_across_phases(monkeypatch):
    paths = [
        _sig_path("D0001", None, ["ch_a", "ch_b"]),
        _sig_path("D0001", None, ["ch_b", "ch_c"]),
    ]

    def fake_read(path, preload=False, verbose=False):
        return _FakeEpochs(path.ch_names)

    import mne

    monkeypatch.setattr(mne, "read_epochs", fake_read)

    union = sig_union_by_subject_modality(paths)
    assert union[("sub-D0001", "sound")] == {"ch_a", "ch_b", "ch_c"}


def test_sig_union_separates_modalities(monkeypatch):
    paths = [
        _sig_path("D0001", "sound", ["ch_a"]),
        _sig_path("D0001", "image", ["ch_img"]),
    ]

    def fake_read(path, preload=False, verbose=False):
        return _FakeEpochs(path.ch_names)

    import mne

    monkeypatch.setattr(mne, "read_epochs", fake_read)

    union = sig_union_by_subject_modality(paths)
    assert union[("sub-D0001", "sound")] == {"ch_a"}
    assert union[("sub-D0001", "image")] == {"ch_img"}


def test_pick_union_channels_intersects_available():
    union = {("sub-D0001", "sound"): {"ch_a", "ch_b", "ch_c"}}
    picks = pick_union_channels(union, "D0001", "sound", ["ch_b", "ch_d", "ch_a"])
    assert picks == ["ch_a", "ch_b"]


def test_pick_union_channels_empty_when_no_overlap():
    union = {("sub-D0001", "sound"): {"ch_a"}}
    picks = pick_union_channels(union, "D0001", "sound", ["ch_x"])
    assert picks == []


def test_sig_union_skips_baseline(monkeypatch):
    paths = [
        _sig_path("D0001", None, ["ch_a"]),
        SimpleNamespace(
            subject="D0001",
            recording=None,
            description="baseline",
            processing="baseline",
            ch_names=["ch_base"],
        ),
    ]

    def fake_read(path, preload=False, verbose=False):
        return _FakeEpochs(path.ch_names)

    import mne

    monkeypatch.setattr(mne, "read_epochs", fake_read)

    union = sig_union_by_subject_modality(paths)
    assert union[("sub-D0001", "sound")] == {"ch_a"}
    assert "ch_base" not in union[("sub-D0001", "sound")]
