#!/usr/bin/env python3
"""Execute an approved publication notebook and retain its SVG-only outputs."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('figure', type=int, choices=(3, 4, 5, 6))
    parser.add_argument('--available-only', action='store_true', help='Figure 3 cached-statistics/spatial panels; no waveform substitution')
    args = parser.parse_args()
    number = args.figure
    out = ROOT / f'results/fig{number}'
    out.mkdir(parents=True, exist_ok=True)
    path = ROOT / f'vizpub/fig{number}.ipynb'
    nb = nbformat.read(path, as_version=4)
    if args.available_only:
        assert number == 3
        original = nb.cells
        selected = [original[i] for i in (1, 2, 4)]
        selected.append(nbformat.v4.new_code_cell('assignments = load_assignments()\nprint("Frozen assignments", len(assignments))', id='available-assignment'))
        selected.extend(original[i] for i in (14, 15, 17))
        setup = original[19].source.split('pn_repeat = load_picture_naming(assignments)')[0]
        selected.append(nbformat.v4.new_code_cell(setup, id='available-pn-spatial-setup'))
        selected.extend(original[i] for i in (22, 23, 25, 28))
        for index in (34, 36):
            selected.append(nbformat.v4.new_code_cell(original[index].source.split('# Phases actually present')[0], id=f'available-spatial-{index}'))
        selected.append(nbformat.v4.new_code_cell('''from pathlib import Path
_out = ROOT / "results/fig3"
for _name in ("table", "rvp", "dvr", "members", "spatial_counts"):
    if _name in globals() and isinstance(globals()[_name], pd.DataFrame):
        globals()[_name].to_csv(_out / f"available_{_name}.csv", index=False)
pd.DataFrame(cluster_stats, columns=["contrast_row", "functional_cluster", "n_sites", "n_highlighted", "wilcoxon_p"]).to_csv(_out / "available_paired_statistics.csv", index=False)
''', id='available-audit'))
        nb.cells = selected
    protected = json.loads((ROOT / 'docs/evidence/figures3_6_refresh_2026-10-17/baseline.json').read_text())['protected']
    assert all(sha(p) == digest for p, digest in protected.items()), 'Frozen Figure 1/2 or NMF changed'
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    jupyter = out / 'jupyter'
    spec = jupyter / f'kernels/insula-paper-fig{number}'
    spec.mkdir(parents=True, exist_ok=True)
    (spec / 'kernel.json').write_text(json.dumps(dict(
        argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        display_name=f'Insula Figure {number}', language='python')))
    os.environ['JUPYTER_PATH'] = str(jupyter) + os.pathsep + os.environ.get('JUPYTER_PATH', '')
    def started(cell, cell_index, **kwargs):
        print(datetime.now(timezone.utc).isoformat(), 'cell', cell_index,
              cell.source.splitlines()[0][:100] if cell.source else '', flush=True)
    client = NotebookClient(nb, kernel_name=f'insula-paper-fig{number}', timeout=3600,
        iopub_timeout=180, raise_on_iopub_timeout=True, record_timing=True,
        resources={'metadata': {'path': str(ROOT / 'vizpub')}}, on_cell_start=started)
    try:
        client.execute()
    except Exception:
        nbformat.write(nb, out / f'fig{number}_failed.ipynb')
        raise
    assert all(sha(p) == digest for p, digest in protected.items()), 'Frozen inputs changed'
    errors = [o for c in nb.cells for o in c.get('outputs', []) if o.get('output_type') == 'error']
    assert not errors
    assert not [o for c in nb.cells for o in c.get('outputs', []) if 'image/png' in o.get('data', {})]
    assert all(c.execution_count is not None for c in nb.cells if c.cell_type == 'code')
    nb.metadata['insula_refresh'] = dict(executed_utc=datetime.now(timezone.utc).isoformat(),
        cohort='sound_only_2026-10-17', assignments_sha256=sha(ROOT / 'results/nmf/channel_assignments.csv'),
        approved_plan='figures3_6_refresh_2026-10-17', svg_inline_only=True)
    destination = out / 'available_spatial_statistics.ipynb' if args.available_only else path
    temporary = destination.with_suffix('.executed.tmp.ipynb')
    nbformat.write(nb, temporary)
    temporary.replace(destination)
    if args.available_only:
        # Put genuine executed outputs back into the publication notebook;
        # every other cell remains explicitly unexecuted while epochs lack.
        main_nb = nbformat.read(path, as_version=4)
        by_id = {c.id: c for c in nb.cells}
        completed_ids = []
        for cell in main_nb.cells:
            executed = by_id.get(cell.id)
            if executed is not None and cell.cell_type == 'code' and cell.source == executed.source:
                cell.outputs = executed.outputs
                cell.execution_count = executed.execution_count
                cell.metadata = executed.metadata
                completed_ids.append(cell.id)
        main_nb.metadata['insula_refresh'] = dict(scope='partial: original spatial/statistical cells only',
            completed_cell_ids=completed_ids, full_waveforms_pending_original_epochs=True,
            assignments_sha256=sha(ROOT / 'results/nmf/channel_assignments.csv'))
        nbformat.write(main_nb, path)
    images = sorted((ROOT / f'img/fig{number}').glob('*.svg'))
    result = dict(passed=True, figure=number, completed_utc=datetime.now(timezone.utc).isoformat(),
        code_cells=sum(c.cell_type == 'code' for c in nb.cells), all_code_cells_executed=True,
        svg_inline_only=True, notebook_sha256=sha(destination), frozen_figure1_2_and_nmf_unchanged=True,
        scope='cached statistical/spatial panels only; full waveform panels pending original epochs' if args.available_only else 'full notebook',
        figures=[dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size) for p in images])
    assert images
    receipt = 'available_execution_validation.json' if args.available_only else 'execution_validation.json'
    (out / receipt).write_text(json.dumps(result, indent=2) + '\n')
    (out / f'fig{number}_failed.ipynb').unlink(missing_ok=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
