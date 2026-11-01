"""Copy the Figure 3 multiphase plotting definitions into an independent SI notebook."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vizpub/fig3.ipynb"
DEST = ROOT / "vizpub/supplementary/context_comparisons.ipynb"
OUT = ROOT / "results/supplementary/context_comparisons"
OUT.mkdir(parents=True, exist_ok=True)
source = json.loads(SOURCE.read_text())
cells = {c.get("id"): "".join(c["source"]) for c in source["cells"]}
names = {"rfig3_xlim_span", "rfig3_time_width_cm", "rfig3_compute_shared_ylim",
         "rfig3_blend_rgb", "rfig3_darken", "rfig3_lighten", "rfig3_condition_color",
         "rfig3_condition_cmap", "rfig3_plot_epoch_raster_traces_stack_phases",
         "rfig3_crop_insula_screenshot", "rfig3_electrode_key"}
definitions, proofs = [], []
for cell_id in ("6f90f72d", "2c822034"):
    text = cells[cell_id]
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            block = ast.get_source_segment(text, node)
            definitions.append(block)
            proofs.append(dict(function=node.name, source_cell=cell_id,
                               copied_text_sha256=hashlib.sha256(block.encode()).hexdigest()))
assert {p["function"] for p in proofs} == names

def markdown(text, identity):
    return dict(cell_type="markdown", id=identity, metadata={}, source=text)

def code(text, identity):
    return dict(cell_type="code", id=identity, metadata={}, execution_count=None, outputs=[], source=text)

setup = """%matplotlib inline
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('svg')
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D
from matplotlib import font_manager
from IPython.display import SVG, display
ROOT = Path('/cwork/ns458/insula-paper')
sys.path.insert(0, str(ROOT))
from src.nmf.waveform_analysis import FUNCTION_COLORS, CLUSTER_LABELS
from src.paths import save_svg
from src.supplementary.context_comparisons import load_context_data, condition_phase_maps, draw_brain_maps, fit_hga_canvas, highlight_mapped_phase, CONTRASTS, PHASES, DISPLAY_WINDOWS
OUT = ROOT / 'results/supplementary/context_comparisons'
IMG = ROOT / 'img/supplementary/context_comparisons'
IMG.mkdir(parents=True, exist_ok=True)
cm = 1 / 2.54
fontsize = 7
for font in Path(sys.prefix, 'fonts').glob('arial*.ttf'):
    font_manager.fontManager.addfont(str(font))
plt.rcParams.update({'svg.fonttype':'none', 'font.family':'sans-serif', 'font.sans-serif':['Arial'], 'font.size':7})
rfig3_clusters = ['sensory', 'sustain', 'motor']
rfig3_cluster_labels = dict(sensory='sensory', sustain='sustained', motor='motor')
RFIG3_STACK_BASE_WIDTH_CM = 2.5
RFIG3_TRACE_HEIGHT_CM = 1.5
RFIG3_CM_PER_RASTER_ROW = 0.02
RFIG3_PHASE_ORDER = ('stimulus', 'delay', 'response')
"""
parameter_cell = cells["7a420cef"]
parameter_node = next(node for node in ast.parse(parameter_cell).body
                      if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'rfig3_epoch_specs' for t in node.targets))
setup += "\n" + ast.get_source_segment(parameter_cell, parameter_node)
nb_cells = [markdown("""# Figure 3 condition comparisons — supplementary

Independent supplementary notebook. The original multiphase plotting
definitions below are copied verbatim from `vizpub/fig3.ipynb` (helpers and
restructured multiphase stack). No main notebook is executed or modified.

Three comparisons: Picture Naming Repeat sound/image; Lexical Delay sound
Decision/Repeat; Picture Naming sound Repeat/Passive. Each comparison is
saved and shown as two SVGs, not embedded into one file: the three-phase
HGA stack, and the brain maps. The HGA stack has three phases and three
component rows, each with electrode mean ± SEM above paired, baseline-masked
heatmaps. The brain maps have one row per condition and left/right insula.
The mapped phase is **Stimulus** for Sound–Image and Repeat–Passive, and
**Delay** for Decision–Repeat. Its HGA column is labeled without a border;
the brain-map header states the exact same displayed window. Colored sites
are responsive to baseline in that window, not between-condition differences.

Frozen current NMF labels and original colors, display limits, summary,
ordering, P95 scales and mask-false blanking are retained. Native restoration
QC remains in force. Passive Response reuses the original stored nominal
alignment, restricted to the current good task electrode set. It has no
actual vocal onset. No new inferential test or time alignment is introduced.
""", "context-title"), code(setup, "context-imports"),
    markdown("## Original Figure 3 plotting definitions\n\nThese function bodies are copied verbatim. Provenance is saved beside the source lists.", "context-code-note"),
    code("\n\n".join(definitions), "context-original-functions"),
    code("data, assignments, counts = load_context_data(OUT)\ndisplay(counts)", "context-data")]
for spec in (("sound_image", "Sound–Image"), ("decision_repeat", "Decision–Repeat"), ("repeat_passive", "Repeat–Passive")):
    key, title = spec
    mapped_phase = 'Delay' if key == 'decision_repeat' else 'Stimulus'
    nb_cells.append(markdown(f"## {title}\n\nShown separately: the three-phase HGA stack, then the **{mapped_phase}** brain maps. Colored sites are baseline-responsive in the labeled window.", f"context-{key}-note"))
    block = f"""spec = next(s for s in CONTRASTS if s['key'] == {key!r})
frame = data.loc[data.contrast.eq(spec['key'])].copy()
xlims = dict(DISPLAY_WINDOWS)
original_spec = rfig3_epoch_specs[spec['source_spec']]
xlims[original_spec['phase']] = tuple(original_spec['xlim'])
floor = max(.4, float(original_spec.get('ylim', (0, .4))[1]))
ylim = rfig3_compute_shared_ylim(frame, phases=PHASES, compare_field=spec['field'], levels=spec['levels'], xlim_by_phase=xlims, ymax_floor=floor)
wave_path = IMG / ('supp_' + spec['key'] + '_waveforms_heatmaps.svg')
fig, axes = rfig3_plot_epoch_raster_traces_stack_phases(frame, phases=PHASES, compare_field=spec['field'], levels=spec['levels'], xlim_by_phase=xlims, ylim=ylim, colors=FUNCTION_COLORS, cm_per_raster_row=0.032)
fig.axes[0].get_subplotspec().get_gridspec().update(wspace=0.26)
for ax in fig.axes:
    ticklabels = ax.get_xticklabels()
    if not ticklabels or not any(label.get_visible() for label in ticklabels):
        continue
    ticks = ax.get_xticks()
    labels = [label.get_text() for label in ticklabels]
    if len(labels) != len(ticks):
        continue
    for i, tick in enumerate(ticks):
        if 1e-8 < abs(tick) < 0.08:
            labels[i] = ''
    ax.set_xticklabels(labels)
highlight_mapped_phase(fig, axes, spec)
fit_hga_canvas(fig, width_cm=13.0, max_height_cm=None)
save_svg(fig, wave_path, close=False)
plt.close(fig)
points = condition_phase_maps(frame, assignments, spec, OUT, xlims)
map_path = IMG / ('supp_' + spec['key'] + '_brain_maps.svg')
fig_maps = draw_brain_maps(points, spec, map_path)
plt.close(fig_maps)
display(SVG(filename=str(wave_path)))
display(SVG(filename=str(map_path)))
"""
    nb_cells.append(code(block, f"context-{key}-plot"))
nb_cells.append(code("""from src.figure3_context_draft import write_json, digest
write_json(OUT / 'completion.json', dict(contrasts=[s['key'] for s in CONTRASTS], three_phases=list(PHASES), layout='separate SVGs; HGA stack and brain maps are not assembled', brain_map_scope='condition-specific baseline masks in the labeled displayed phase window', mapped_phases={s['key']:s['map_phase'] for s in CONTRASTS}, new_condition_tests=False, assignments_sha256=digest(ROOT/'results/nmf/channel_assignments.csv')))
""", "context-completion"))
nb = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(display_name="Insula supplementary", language="python", name="insula-supplementary")), cells=nb_cells)
DEST.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
(OUT / "source_provenance.json").write_text(json.dumps(dict(source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), copied_functions=proofs), indent=2) + "\n")
print(DEST)
