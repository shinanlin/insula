"""Execute the grant notebook's temporal cells from frozen results on Slurm."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import os
import xml.etree.ElementTree as ET
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/grant_nmf_30subjects'
FIG = ROOT / 'img/grant_nmf_30subjects'
SOURCE = ROOT / 'notebooks/grant_nmf_30subjects.ipynb'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    notebook = nbformat.read(SOURCE, as_version=4)
    ids = ('grant30-temporal', 'grant30-wholebrain-temporal')
    originals = {c.id: c.source for c in notebook.cells if c.id in ids}
    protected = list(OUT.glob('*.csv')) + list(OUT.glob('*.npz'))
    protected += [OUT/'cohort.json', OUT/'nmf_manifest.json', ROOT/'results/nmf/H_by_phase.csv']
    protected += list((OUT/'whole_brain_projection').glob('*.csv'))
    protected += list((OUT/'whole_brain_projection').glob('*.npz'))
    protected += [OUT/'whole_brain_projection/manifest.json']
    protected += [p for p in FIG.glob('*.svg') if 'temporal' not in p.name]
    protected += [ROOT/'vizpub/fig2.ipynb', ROOT/'img/fig2/fig2_nmf_traces.svg',
                  ROOT/'img/fig2/fig2_hga_traces.svg']
    before = {str(p): digest(p) for p in protected}
    bootstrap = '''from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from IPython.display import SVG, display
ROOT=Path.cwd()
sys.path.insert(0,str(ROOT))
from src.nmf.waveform_analysis import CLUSTER_ORDER, CLUSTER_LABELS, FUNCTION_COLORS
OUT=ROOT/'results/grant_nmf_30subjects'
FIG=ROOT/'img/grant_nmf_30subjects'
PROJ=OUT/'whole_brain_projection'
CANONICAL=ROOT/'results/nmf'
COMPONENTS=tuple(CLUSTER_ORDER)
cm=1/2.54
for p in (Path(sys.prefix)/'fonts').glob('arial*.ttf'):font_manager.fontManager.addfont(str(p))
plt.rcParams.update({'svg.fonttype':'none','font.family':'sans-serif','font.sans-serif':['Arial'],'font.size':7})
TEMPORAL_REDRAW_ONLY=True
'''
    collect = "(OUT/'temporal_layouts.json').write_text(json.dumps(temporal_layouts,indent=2))"
    runbook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(bootstrap)] +
        [c for c in notebook.cells if c.id in ids] + [nbformat.v4.new_code_cell(collect)],
        metadata=notebook.metadata)
    executed = OUT/'grant_nmf_temporal_redraw.executed.ipynb'
    client = NotebookClient(runbook, timeout=300, kernel_name='python3',
        resources={'metadata': {'path': str(ROOT)}},
        on_cell_start=lambda cell, cell_index, **kw: print(f'Cell {cell_index}: {cell.id}', flush=True))
    try:
        client.execute()
    finally:
        nbformat.write(runbook, executed)
    assert all(digest(Path(p)) == value for p, value in before.items()), 'Protected input changed'
    # Independently compare the new plot metadata against the requested notebook.
    reference = nbformat.read(ROOT/'vizpub/fig2.ipynb', as_version=4)
    constants = {}
    wanted = {'FIG2_TRACE_XLIMS', 'FIG2_TRACE_XTICKS', 'FIG2_PHASE_XLIMS', 'FIG2_PHASE_XTICKS'}
    for cell in reference.cells:
        if cell.cell_type != 'code': continue
        for node in ast.parse(cell.source).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                key = node.targets[0].id
                if key in wanted: constants[key] = ast.literal_eval(node.value)
    layouts = json.loads((OUT/'temporal_layouts.json').read_text())
    assert len(layouts) == 4
    figures = {}
    for layout in layouts:
        path = FIG/layout['filename']
        root = ET.parse(path).getroot()
        size = [float(root.attrib[k].removesuffix('pt'))*2.54/72 for k in ('width','height')]
        assert all(abs(a-b)<1e-6 for a,b in zip(size,[8.,3.])), size
        hga = path.name == 'insula_hga_temporal.svg'
        windows = constants['FIG2_TRACE_XLIMS' if hga else 'FIG2_PHASE_XLIMS']
        ticks = constants['FIG2_TRACE_XTICKS' if hga else 'FIG2_PHASE_XTICKS']
        # User requested the Figure 2 display with the Go column omitted.
        windows = {k:v for k,v in windows.items() if k != 'go'}
        ticks = {k:v for k,v in ticks.items() if k != 'go'}
        assert list(layout['windows']) == ['stimulus','delay','response']
        assert len(layout['axes_bounds']) == 3
        assert 'Go' not in [t.text for t in root.iter('{http://www.w3.org/2000/svg}text')]
        assert layout['xlims'] == [list(v) for v in windows.values()]
        assert layout['xticks'] == [list(v) for v in ticks.values()]
        if hga: assert layout['ylims'] == [[-.1,.8]]*3
        figures[path.name] = digest(path)
    # Preserve concurrently updated unrelated notebook cells and metadata.
    current = nbformat.read(SOURCE, as_version=4)
    updates = {c.id: c for c in runbook.cells if c.id in ids}
    for i,cell in enumerate(current.cells):
        if cell.id in ids:
            assert cell.source == originals[cell.id], 'Temporal source changed; executed copy retained'
            current.cells[i] = updates[cell.id]
    nbformat.write(current, SOURCE)
    nbformat.write(current, OUT/'grant_nmf_30subjects.executed.ipynb')
    validation = json.loads((OUT/'validation.json').read_text())
    validation['figures'].update(figures)
    validation['temporal_redraw'] = dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        slurm_job=os.environ.get('SLURM_JOB_ID'),reference='vizpub/fig2.ipynb',
        display_phases=['stimulus','delay','response'],go_display_omitted=True,
        data_and_spatial_maps_unchanged=True,subject_sem_preserved=True,
        figures=figures,layouts=layouts,protected_sha256=before)
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    (OUT/'temporal_redraw_validation.json').write_text(json.dumps(validation['temporal_redraw'],indent=2)+'\n')
    print('Validated four 8 x 3 cm SVGs, three phases each, no Go; retained Figure 2 windows/ticks matched; inputs and spatial maps unchanged.',flush=True)


if __name__ == '__main__':
    main()
