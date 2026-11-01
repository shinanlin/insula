"""Append only three author-approved largest-weight Gaussian map subpanels."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm'
    import nbformat
    from nbclient import NotebookClient
    from scripts.refresh_lexical_condition_sets import ROOT, protected, sha, write
    from scripts.refresh_picture_condition_sets import NB, OUT, IMG, protected as picture_protected
    from src.supplementary.picture_top_weight_gaussian import AUDIT, OUTPUT, EXPORTS, notebook_cells
    AUDIT.mkdir(parents=True, exist_ok=True)
    protected(); picture_protected()
    current = json.loads((OUT / 'current_display.json').read_text())
    prior = json.loads(Path(current['completion']).read_text())
    assert prior['passed'] and sha(NB) == prior.get('notebook_after_sha256', prior.get('notebook_sha256'))
    original = nbformat.read(NB, as_version=4)
    original_sha = sha(NB)
    cells = notebook_cells()
    ids = {c.id for c in cells}
    owned = [c for c in original.cells if c.id in ids]
    assert not owned or {c.id for c in owned} == ids
    archive = AUDIT / 'notebook_before.ipynb'
    if not archive.exists():
        shutil.copy2(NB, archive)
    if sha(archive) != original_sha:
        assert len(owned) == len(cells), 'Unrelated notebook change since snapshot'
        rerun = AUDIT / 'reruns' / original_sha
        rerun.mkdir(parents=True, exist_ok=True)
        shutil.copy2(NB, rerun / 'notebook_before.ipynb')
        for name in ['completion.json', 'current_display_before.json']:
            if (AUDIT / name).exists():
                shutil.copy2(AUDIT / name, rerun / name)
        if OUTPUT.exists():
            shutil.copytree(OUTPUT, rerun / 'display_data_before', dirs_exist_ok=True)
        for name in EXPORTS.values():
            if (IMG / name).exists():
                shutil.copy2(IMG / name, rerun / name)
    shutil.copy2(OUT / 'current_display.json', AUDIT / 'current_display_before.json')
    additions = {IMG / name for name in EXPORTS.values()}
    files = [p for p in IMG.rglob('*') if p.is_file() and p not in additions]
    files += [p for p in OUT.rglob('*') if p.is_file() and OUTPUT not in p.parents and p.name != 'current_display.json']
    files.append(ROOT / 'vizpub/supplementary/fig3_lexical_delay_condition_sets.ipynb')
    for folder in ['img/supplementary/fig3_lexical_delay_condition_sets', 'results/supplementary/fig3_lexical_delay_condition_sets']:
        files += [p for p in (ROOT / folder).rglob('*') if p.is_file()]
    frozen = {str(p): sha(p) for p in sorted(set(files))}
    write(AUDIT / 'protected_existing_outputs.json', frozen)
    kernel = AUDIT / 'jupyter/kernels/picture-top-weight-gaussian'
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / 'kernel.json').write_text(json.dumps(dict(argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        display_name='PN top original NMF loading Gaussian maps', language='python')))
    os.environ['JUPYTER_PATH'] = str(AUDIT / 'jupyter') + os.pathsep + os.environ.get('JUPYTER_PATH', '')
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    new = nbformat.v4.new_notebook(cells=cells)
    try:
        NotebookClient(new, kernel_name='picture-top-weight-gaussian', timeout=1200,
            resources={'metadata': {'path': str(NB.parent)}}).execute()
    except Exception:
        nbformat.write(new, AUDIT / 'failed.ipynb')
        raise
    assert all(c.execution_count is not None for c in new.cells if c.cell_type == 'code')
    assert not [o for c in new.cells for o in c.get('outputs', []) if o.get('output_type') == 'error' or 'image/png' in o.get('data', {})]
    assert sha(NB) == original_sha, 'Notebook changed during execution; preserve author edits'
    assert all(sha(p) == h for p, h in frozen.items()), 'An earlier figure, source or result changed'
    if owned:
        untouched = [c for c in original.cells if c.id not in ids]
        by_id = {c.id: c for c in new.cells}
        original.cells = [by_id.get(c.id, c) for c in original.cells]
        assert [c for c in original.cells if c.id not in ids] == untouched
    else:
        old = list(original.cells)
        original.cells.extend(new.cells)
        assert original.cells[:-len(new.cells)] == old
    exports = {}
    for path in additions:
        tree = ET.fromstring(path.read_text())
        identifiers = [e.attrib['id'] for e in tree.iter() if 'id' in e.attrib]
        assert len(identifiers) == len(set(identifiers)), path
        exports[str(path)] = sha(path)
    main_count = protected()
    picture_protected()
    nbformat.write(original, NB)
    write(AUDIT / 'completion.json', dict(passed=True, job_id=os.environ['SLURM_JOB_ID'], notebook=str(NB),
        notebook_before_sha256=original_sha, notebook_after_sha256=sha(NB),
        original_notebook_cells_and_outputs_preserved=not bool(owned),
        earlier_notebook_sections_and_outputs_preserved=True, replaced_owned_cell_ids=sorted(ids) if owned else [],
        existing_sources_and_outputs_preserved=len(frozen),
        main_protected_files_unchanged=main_count, exports=exports, independent_subpanels_only=True,
        no_composite_created=True, provenance=json.loads((OUTPUT / 'provenance.json').read_text()),
        completed_utc=datetime.now(timezone.utc).isoformat()))
    write(OUT / 'current_display.json', dict(completion=str(AUDIT / 'completion.json'), notebook=str(NB), steps=8,
        scope='Three independent bilateral Gaussian maps using only the largest original loading per electrode'))
    print('COMPLETE: three independent top-weight Gaussian maps', json.dumps(exports, indent=2), flush=True)


if __name__ == '__main__':
    main()
