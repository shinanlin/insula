"""Render supplementary SVG previews and verify the isolated source copy."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import cairosvg
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("destination", type=Path)
args = parser.parse_args()
args.destination.mkdir(parents=True, exist_ok=True)
topic = ROOT / "results/supplementary/context_comparisons"
source = json.loads((ROOT / "vizpub/fig3.ipynb").read_text())
copied = json.loads((ROOT / "vizpub/supplementary/context_comparisons.ipynb").read_text())
provenance = json.loads((topic / "source_provenance.json").read_text())
cells = {c.get("id"): "".join(c["source"]) for c in source["cells"]}
definitions = "".join(next(c for c in copied["cells"] if c.get("id") == "context-original-functions")["source"])
copied_functions = {n.name: ast.get_source_segment(definitions, n) for n in ast.parse(definitions).body if isinstance(n, ast.FunctionDef)}
for proof in provenance["copied_functions"]:
    assert hashlib.sha256(copied_functions[proof["function"]].encode()).hexdigest() == proof["copied_text_sha256"]
    original = cells[proof["source_cell"]]
    node = next(n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef) and n.name == proof["function"])
    assert copied_functions[proof["function"]] == ast.get_source_segment(original, node)
outputs = []
frozen = json.loads((topic / 'history/three_phase_union_2026-10-20/waveform_data_hashes.json').read_text())
for name, identity in frozen.items():
    assert hashlib.sha256((topic / name).read_bytes()).hexdigest() == identity, name
data = pd.read_pickle(topic / 'waveforms.pkl')
map_counts = []
for key in ("sound_image", "decision_repeat", "repeat_passive"):
    points = pd.read_csv(topic / f'{key}_brain_membership.csv')
    phase = 'delay' if key == 'decision_repeat' else 'stimulus'
    assert points.map_phase.eq(phase).all()
    bounds = points[['map_window_start', 'map_window_end']].drop_duplicates()
    assert len(bounds) == 1
    start, end = bounds.iloc[0]
    for level, group in points.groupby('condition'):
        expected = data.loc[(data.contrast == key) & (data.level == level) &
            (data.phase == phase) & data.time.between(start, end) & data['mask'],
            ['subject', 'channel']].drop_duplicates()
        actual = group.loc[group.significant, ['subject', 'channel']]
        assert set(map(tuple, expected.to_numpy())) == set(map(tuple, actual.to_numpy()))
        map_counts.append(dict(contrast=key, condition=level, phase=phase,
            window=[start, end], n_significant=len(actual)))
    for part in ("waveforms_heatmaps", "brain_maps"):
        svg = ROOT / "img/supplementary/context_comparisons" / f"supp_{key}_{part}.svg"
        png = args.destination / f"supp_{key}_{part}.png"
        cairosvg.svg2png(url=str(svg), write_to=str(png), output_width=2200, background_color="white")
        outputs.append(dict(contrast=key, part=part, svg=str(svg), preview=str(png)))
report = dict(original_function_texts_identical=True, copied_functions=len(provenance["copied_functions"]),
    waveform_data_unchanged=True, selected_phase_map_membership_verified=True,
    map_counts=map_counts, outputs=outputs)
(topic / "preview_validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
