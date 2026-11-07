#!/usr/bin/env python3
"""Validate current formal Figure 4 inference and unchanged publication style."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.decoding.figure4_collect import validate_stage
from src.decoding.figure4_formal import AUDIT, load_plan, verify_protected, write_json


def validate():
    values = validate_stage()
    plan, spec = load_plan()
    previous = json.loads((AUDIT / "before/vizpub/fig4.ipynb").read_text())
    current = json.loads((ROOT / "vizpub/fig4.ipynb").read_text())
    sources = lambda nb: {c["id"]: "".join(c.get("source", [])) for c in nb["cells"]}
    old, new = sources(previous), sources(current)
    for key in ("fig4-style", "6d90cfcc"):
        assert new[key] == old[key], f"Original style/confusion/lexical source changed: {key}"
    # User-approved 2026-10-21 display crop: only the 1-D lexical axes change.
    expected_lexical = old["fa9672cb"].replace(
        '        ax.set_xlim(-0.5, 1.5)',
        '        ax.set_xlim(*LEXICAL_DISPLAY_XLIMS[phase])\n'
        '        ax.set_xticks(LEXICAL_DISPLAY_XTICKS[phase], labels=LEXICAL_DISPLAY_XTICKLABELS[phase])')
    expected_lexical = expected_lexical.replace(
        '_row_h_cm = (2.0 * _scale, 2.0 * _scale, _ax_cm)',
        '_tgm_h_cm = 3 * _ax_cm * max(LEXICAL_DISPLAY_WIDTH_RATIOS) / sum(LEXICAL_DISPLAY_WIDTH_RATIOS)\n'
        '_row_h_cm = (2.0 * _scale, 2.0 * _scale, _tgm_h_cm)').replace(
        'gridspec_kw={"height_ratios": list(_row_h_cm)},',
        'gridspec_kw={"height_ratios": list(_row_h_cm), "width_ratios": LEXICAL_DISPLAY_WIDTH_RATIOS},').replace(
        '    ax.set_xlim(-0.5, 1.5)',
        '    ax.set_xlim(*LEXICAL_DISPLAY_XLIMS[phase])\n'
        '    ax.set_xticks(LEXICAL_DISPLAY_XTICKS[phase], labels=LEXICAL_DISPLAY_XTICKLABELS[phase])').replace(
        '# side length, so those panels stay square; Repeat and Decision rows keep',
        '# phase-dependent widths; TGM aspect retains equal time units. Repeat and Decision rows keep')
    assert new["fa9672cb"] == expected_lexical, "Unapproved lexical source change"
    assert new["fig4-plot"] == old["fig4-plot"].replace('["p_value"]', '["p_value_fdr"]')
    receipt = dict(passed=True, completed_utc=datetime.now(timezone.utc).isoformat(),
        inference=spec["inference"], n_repeats=30, n_folds=5, n_permutations=500,
        correction="fdr_bh", correction_family_size=9, all_fits_converged=True,
        original_bar_geometry_and_confusion_style_unchanged=True,
        lexical_plot_change="User-approved phase x axes and proportional panel widths, 2026-10-21",
        protected_count=verify_protected(),
        specification_sha256=plan["specification_sha256"],
        models=[dict(roi=m["roi"], phase=m["phase"], n_sites=v["n_electrodes"],
            n_subjects=v["n_subjects"], n_trials=v["n_trials"], mean=v["mean"], SE=v["SE"],
            p_raw=v["p_value"], p_fdr=v["p_value_fdr"],
            max_iter=v["max_iter"],
            max_observed_iterations=v["max_observed_iterations"],
            max_null_iterations=v["max_null_iterations"])
            for m, v in zip(plan["models"][:9], values)])
    write_json(AUDIT / "validation.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    return receipt


if __name__ == "__main__":
    validate()
