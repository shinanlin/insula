"""Tests for HGA Explorer Phase 1a export."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from viewer.hga_explorer.export.compute_hga_explorer import (
    DISPLAY_WAVEFORM_RANGE,
    PHASES,
    V1_TASKS,
    build_payload,
    build_traces,
    discover_subjects,
    validate_payload,
    validate_traces,
    write_split_layout,
)


def _sample_row(**overrides):
    base = {
        "time": 0.1,
        "channel": "D0094_LPAS2-3",
        "value": 0.42,
        "mask": True,
        "subject": "D0094",
        "description": "Repeat",
        "task": "PhonemeSequencing",
        "phase": "Delay",
        "modality": "sound",
        "label": "ctx_lh_G_front_sup",
        "roi": "SFG",
        "hemi": "L",
        "x": -5.8,
        "y": 2.1,
        "z": 38.3,
        "x_native": -2.25,
        "y_native": -19.1,
        "z_native": 70.7,
        "x1_native": -0.5,
        "y1_native": -19.2,
        "z1_native": 70.5,
        "x2_native": -4.0,
        "y2_native": -19.0,
        "z2_native": 71.0,
        "x1_template": -3.8,
        "y1_template": 2.0,
        "z1_template": 38.1,
        "x2_template": -7.7,
        "y2_template": 2.2,
        "z2_template": 38.7,
        "contact_1": "LPAS2",
        "contact_2": "LPAS3",
        "contact_1_label": "ctx_lh_G_front_sup",
        "contact_2_label": "Left-Cerebral-White-Matter",
    }
    base.update(overrides)
    return base


def test_discover_subjects_union_across_tasks(tmp_path: Path):
    ps_root = tmp_path / "PhonemeSequencing(bipolar)(hammers)"
    ld_root = tmp_path / "LexicalDelay(bipolar)(hammers)"
    (ps_root / "sub-D0001" / "HGA").mkdir(parents=True)
    (ps_root / "sub-D0002" / "HGA").mkdir(parents=True)
    (ld_root / "sub-D0002" / "HGA").mkdir(parents=True)
    (ld_root / "sub-D0003" / "HGA").mkdir(parents=True)
    (ps_root / "sub-D0001" / "HGA" / "x_time.csv").write_text("time\n", encoding="utf-8")
    (ps_root / "sub-D0002" / "HGA" / "x_time.csv").write_text("time\n", encoding="utf-8")
    (ld_root / "sub-D0002" / "HGA" / "x_time.csv").write_text("time\n", encoding="utf-8")
    (ld_root / "sub-D0003" / "HGA" / "x_time.csv").write_text("time\n", encoding="utf-8")

    subjects = discover_subjects(tmp_path, list(V1_TASKS), reference="bipolar", atlas="hammers")
    assert subjects == ["D0001", "D0002", "D0003"]


def test_build_payload_phase_flags_and_hga_by_task():
    rows = [
        _sample_row(task="PhonemeSequencing", phase="Delay", value=0.4),
        _sample_row(task="PhonemeSequencing", phase="Go", value=0.6),
        _sample_row(task="LexicalDelay", phase="Delay", value=0.5),
        _sample_row(task="LexicalDelay", phase="Delay", value=0.7, description="Decision"),
    ]
    payload = build_payload(pd.DataFrame(rows), tasks=list(V1_TASKS), subjects=["D0094"])
    electrode = payload["electrodes"][0]

    assert electrode["phase_flags"]["delay"] is True
    assert electrode["phase_flags"]["go"] is True
    assert set(electrode["hga_by_task"]) == set(V1_TASKS)
    assert electrode["hga_by_task"]["PhonemeSequencing"] == pytest.approx(0.5)
    assert electrode["hga_by_task"]["LexicalDelay"] == pytest.approx(0.5)
    assert not any(key.startswith("maper_") for key in electrode)


def test_write_split_layout_creates_manifest_and_electrodes(tmp_path: Path):
    rows = [_sample_row()]
    payload = build_payload(pd.DataFrame(rows), tasks=list(V1_TASKS), subjects=["D0094"])
    issues = validate_payload(payload, tasks=list(V1_TASKS))
    assert issues == []

    write_split_layout(payload, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    electrodes_doc = json.loads((tmp_path / "electrodes.json").read_text(encoding="utf-8"))
    trace_doc = json.loads((tmp_path / "traces" / "D0094.json").read_text(encoding="utf-8"))

    assert manifest["layout"] == "split"
    assert manifest["metadata"]["tasks"] == list(V1_TASKS)
    assert manifest["files"]["traces"]["D0094"] == "traces/D0094.json"
    assert len(electrodes_doc["electrodes"]) == 1
    assert set(electrodes_doc["electrodes"][0]["phase_flags"]) == set(PHASES)
    assert trace_doc["subject"] == "D0094"
    assert "D0094|D0094_LPAS2-3" in trace_doc["traces"]


def test_build_traces_clip_and_keying():
    rows = []
    for t in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
        rows.append(_sample_row(time=t, value=float(t), task="PhonemeSequencing", phase="Delay"))
        rows.append(
            _sample_row(
                time=t,
                value=float(t) + 0.1,
                task="LexicalDelay",
                phase="Delay",
                description="Decision",
            )
        )
    df = pd.DataFrame(rows)
    df["phase"] = df["phase"].str.lower()
    df["electrode_id"] = df["subject"].astype(str) + "|" + df["channel"].astype(str)
    traces = build_traces(df, {"D0094|D0094_LPAS2-3"}, list(V1_TASKS), max_trace_points=100)

    ps_trace = traces["D0094|D0094_LPAS2-3"]["PhonemeSequencing"]["delay"]["Repeat"]
    ld_trace = traces["D0094|D0094_LPAS2-3"]["LexicalDelay"]["delay"]["Decision"]
    assert min(ps_trace["time"]) >= DISPLAY_WAVEFORM_RANGE[0]
    assert max(ps_trace["time"]) <= DISPLAY_WAVEFORM_RANGE[1]
    assert -1.0 not in ps_trace["time"]
    assert 1.5 not in ld_trace["time"]
    assert validate_traces(traces, tasks=list(V1_TASKS)) == []


def test_build_roi_mean_sources_from_hga_by_task():
    from viewer.hga_explorer.export.hga_explorer_kde import build_roi_mean_sources

    electrodes = [
        {
            "roi": "SFG",
            "x": 1.0,
            "y": 2.0,
            "z": 3.0,
            "hga_by_task": {"PhonemeSequencing": 0.4, "LexicalDelay": 0.2},
        },
        {
            "roi": "SFG",
            "x": 2.0,
            "y": 3.0,
            "z": 4.0,
            "hga_by_task": {"PhonemeSequencing": 0.6, "LexicalDelay": None},
        },
    ]
    payload = build_roi_mean_sources(electrodes)
    assert len(payload["sources"]) == 1
    assert payload["sources"][0]["roi"] == "SFG"
    assert payload["sources"][0]["n_electrodes"] == 2
    assert payload["sources"][0]["weight"] == pytest.approx(1.0)


def test_build_subject_phase_animation_bundle_keys():
    from viewer.hga_explorer.export.hga_explorer_animation import (
        animation_bundle_keys,
        build_subject_phase_animation_bundle,
    )

    rows = []
    for t in [-0.5, 0.0, 0.5, 1.0]:
        rows.append(_sample_row(time=t, value=float(t), task="PhonemeSequencing", phase="Delay"))
    df = pd.DataFrame(rows)
    df["phase"] = df["phase"].str.lower()
    df["electrode_id"] = df["subject"].astype(str) + "|" + df["channel"].astype(str)
    traces = build_traces(df, {"D0094|D0094_LPAS2-3"}, list(V1_TASKS), max_trace_points=10)
    bundle = build_subject_phase_animation_bundle(
        ["D0094|D0094_LPAS2-3"],
        traces,
        "delay",
        tasks=list(V1_TASKS),
    )
    assert set(bundle["bundles"]) == set(animation_bundle_keys(list(V1_TASKS)))
    default_bundle = bundle["bundles"]["all|Repeat"]
    assert default_bundle["selected_task"] == "all"
    assert default_bundle["times"][0] >= DISPLAY_WAVEFORM_RANGE[0]
    assert default_bundle["times"][-1] <= DISPLAY_WAVEFORM_RANGE[1]
