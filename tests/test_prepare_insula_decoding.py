from pathlib import Path

import pandas as pd
import pytest
from mne_bids import BIDSPath

from src.decoding.prepare_insula_decoding_dataset import (
    LEXICAL_NO_DELAY_PHASES,
    PSEUDO_SUBJECTS,
    load_assignments,
    significance_union,
    task_descriptions,
    task_features,
    task_phases,
)


def assignment_rows():
    return pd.DataFrame(
        {
            "subject": ["D1", "D2", "D3", "D4", "D5", "D6"],
            "channel": [
                "D1_L1-2",
                "D2_R1-2",
                "D3_L1-2",
                "D4_R1-2",
                "D5_L2-3",
                "D6_R2-3",
            ],
            "hemi": ["L", "R", "L", "R", "L", "R"],
            "functional_cluster": [
                "sensory",
                "sensory",
                "sustain",
                "sustain",
                "sensory",
                "sustain",
            ],
            "roi": ["AIC", "PIC", "AIC", "PIC", "PIC", "AIC"],
        }
    )


def test_load_assignments_maps_ins_by_hemisphere(tmp_path: Path):
    path = tmp_path / "assignments.csv"
    assignment_rows().to_csv(path, index=False)
    loaded = load_assignments(path)
    assert set(loaded["pseudo_subject"]) == {"INSl", "INSr"}
    assert set(PSEUDO_SUBJECTS) == {"INSl", "INSr"}
    left = loaded[loaded["pseudo_subject"].eq("INSl")]
    right = loaded[loaded["pseudo_subject"].eq("INSr")]
    assert len(left) == 3
    assert len(right) == 3
    assert set(left["functional_cluster"]) == {
        "sensory",
        "sustain",
    }


def test_lexical_nodelay_significance_union_matches_delay_policy():
    channels = {
        "Decision": {"D1_L1-2", "D2_R1-2"},
        "Repeat": {"D3_L1-2"},
    }
    assert significance_union("LexicalNoDelay", channels) == {
        "D1_L1-2",
        "D2_R1-2",
        "D3_L1-2",
    }
    assert significance_union("LexicalDelay", channels) == significance_union(
        "LexicalNoDelay", channels
    )


def test_lexical_delay_task_scope():
    assert task_phases("LexicalDelay") == (
        "Stimulus",
        "Delay",
        "Go",
        "Response",
    )
    assert task_descriptions("LexicalDelay") == ("Decision", "Repeat")
    assert task_features("LexicalDelay") == ("phoneme", "articulator", "lexicality")


def test_lexical_nodelay_task_scope():
    assert task_phases("LexicalNoDelay") == LEXICAL_NO_DELAY_PHASES
    assert task_descriptions("LexicalNoDelay") == ("Decision", "Repeat")
    assert task_features("LexicalNoDelay") == ("phoneme", "articulator", "lexicality")
    assert task_phases("PhonemeSequence") == (
        "Stimulus",
        "Delay",
        "Go",
        "Response",
    )


def test_output_path_uses_ins_pseudo_subject():
    path = BIDSPath(
        root="/tmp/decoding(bipolar)",
        subject="INSl",
        task="LexicalNoDelay",
        processing="Stimulus",
        description="Repeat",
        datatype="lexicality",
        suffix="highgamma",
        extension=".h5",
        check=False,
    )
    assert "sub-INSl" in str(path)
    assert "Insula" not in str(path)
