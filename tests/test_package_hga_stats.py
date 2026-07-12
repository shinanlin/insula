from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from package_HGA import stats_path_candidates


class _FakeBIDSPath:
    def __init__(self, *, root: str, task: str, subject: str = "D0019"):
        self.root = root
        self.task = task
        self.subject = subject
        self.description = "Repeat"
        self.processing = "Stimulus"
        self.suffix = "highgamma"
        self.datatype = "epoch(band)(sig)(effective)"
        self.extension = ".h5"

    def copy(self):
        return _FakeBIDSPath(
            root=self.root,
            task=self.task,
            subject=self.subject,
        )

    def update(self, **kwargs):
        updated = self.copy()
        for key, value in kwargs.items():
            setattr(updated, key, value)
        return updated

    def __str__(self) -> str:
        return (
            f"{self.root}/sub-{self.subject}/bipolar/"
            f"sub-{self.subject}_task-{self.task}_proc-{self.processing}"
            f"_desc-{self.description}_{self.suffix}{self.extension}"
        )


def test_stats_path_candidates_include_phoneme_alias():
    epoch_root = "/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/derivatives/epoch(bipolar)"
    epoch_path = _FakeBIDSPath(root=epoch_root, task="PhonemeSequencing")
    candidates = [str(path) for path in stats_path_candidates(epoch_path, "bipolar")]

    assert len(candidates) == 2
    assert "task-PhonemeSequencing_" in candidates[0]
    assert "task-PhonemeSequence_" in candidates[1]
    assert candidates[0].replace("epoch(bipolar)", "statistics") == candidates[0]
    assert "/statistics/" in candidates[1]


def test_stats_path_candidates_keep_lexical_single_path():
    epoch_root = "/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/derivatives/epoch(bipolar)"
    epoch_path = _FakeBIDSPath(root=epoch_root, task="LexicalDelay")
    candidates = [str(path) for path in stats_path_candidates(epoch_path, "bipolar")]

    assert candidates == [
        (
            f"{epoch_root.replace('epoch(bipolar)', 'statistics')}/sub-D0019/bipolar/"
            "sub-D0019_task-LexicalDelay_proc-Stimulus_desc-Repeat_highgamma.h5"
        )
    ]
