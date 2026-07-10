"""Tests for insula brain mesh export."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

VIEWER_ROOT = Path(__file__).resolve().parents[1] / "viewer" / "hga_explorer"
EXPORT_DIR = VIEWER_ROOT / "export"
ASSETS_DIR = VIEWER_ROOT / "public" / "assets"
NATIVE_DIR = ASSETS_DIR / "native"
VALIDATION_NATIVE_SUBJECTS = ["D0094", "D0071", "D0084"]

sys.path.insert(0, str(EXPORT_DIR))

from insula_constants import INSULA_PATTERNS, is_insula_label  # noqa: E402


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("ctx_lh_G_insular_short", True),
        ("ctx_rh_S_circular_insula_ant", True),
        ("ctx_lh_G_orbital", False),
        ("", False),
        (None, False),
    ],
)
def test_is_insula_label(label, expected):
    assert is_insula_label(label) is expected


def test_insula_patterns_match_fig2():
    assert "G_insular_short" in INSULA_PATTERNS
    assert "S_circular_insula_sup" in INSULA_PATTERNS
    assert len(INSULA_PATTERNS) == 5


@pytest.mark.skipif(
    not (ASSETS_DIR / "cvs_avg35_pial_insula_mask.json").exists(),
    reason="insula assets not exported",
)
def test_insula_mask_matches_full_brain_vertex_count():
    full_meta = json.loads((ASSETS_DIR / "cvs_avg35_pial.meta.json").read_text(encoding="utf-8"))
    mask_payload = json.loads(
        (ASSETS_DIR / "cvs_avg35_pial_insula_mask.json").read_text(encoding="utf-8")
    )
    assert len(mask_payload["mask"]) == full_meta["n_vertices"]
    assert sum(mask_payload["mask"]) > 0


@pytest.mark.skipif(
    not (ASSETS_DIR / "cvs_avg35_insula.meta.json").exists(),
    reason="insula assets not exported",
)
def test_insula_meta_has_camera_hint():
    meta = json.loads((ASSETS_DIR / "cvs_avg35_insula.meta.json").read_text(encoding="utf-8"))
    assert meta["both_target"] is not None
    assert meta["camera_hint"]["distance"] == 180
    assert (ASSETS_DIR / "cvs_avg35_insula_pial.glb").stat().st_size > 0


@pytest.mark.parametrize("subject", VALIDATION_NATIVE_SUBJECTS)
def test_native_insula_mask_matches_pial_vertex_count(subject):
    pial_meta_path = NATIVE_DIR / f"{subject}_pial.meta.json"
    mask_path = NATIVE_DIR / f"{subject}_pial_insula_mask.json"
    if not pial_meta_path.exists() or not mask_path.exists():
        pytest.skip(f"native insula assets not exported for {subject}")

    pial_meta = json.loads(pial_meta_path.read_text(encoding="utf-8"))
    mask_payload = json.loads(mask_path.read_text(encoding="utf-8"))
    assert len(mask_payload["mask"]) == pial_meta["n_vertices"]
    assert sum(mask_payload["mask"]) > 0


@pytest.mark.parametrize("subject", VALIDATION_NATIVE_SUBJECTS)
def test_native_insula_meta_has_camera_hint(subject):
    meta_path = NATIVE_DIR / f"{subject}_insula.meta.json"
    glb_path = NATIVE_DIR / f"{subject}_insula_pial.glb"
    if not meta_path.exists():
        pytest.skip(f"native insula meta not exported for {subject}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["both_target"] is not None
    assert meta["camera_hint"]["target"] is not None
    assert meta["camera_hint"]["distance"] == 180
    assert glb_path.stat().st_size > 0
