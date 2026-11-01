"""Change only three top-weight backgrounds, retaining all saved numeric data."""
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
    from src.supplementary.picture_top_weight_gaussian import (
        DISPLAY_AUDIT as AUDIT, DISPLAY_OUTPUT, EXPORTS, notebook_cells,
    )
    protected(); picture_protected()
    AUDIT.mkdir(parents=True, exist_ok=True)
    current = json.loads((OUT / 'current_display.json').read_text())
    prior = json.loads(Path(current['completion']).read_text())
    original_sha = sha(NB)
    assert prior['passed'] and original_sha == prior.get('notebook_after_sha256', prior.get('notebook_sha256'))
    archive = AUDIT / 'before'
    archive.mkdir(exist_ok=True)
    if not (archive / 'notebook.ipynb').exists():
        shutil.copy2(NB, archive / 'notebook.ipynb')
        shutil.copy2(OUT / 'current_display.json', archive / 'current_display.json')
    assert sha(archive / 'notebook.ipynb') == original_sha
    additions = {IMG / name for name in EXPORTS.values()}
    for path in additions:
        destination = archive / path.name
        if not destination.exists():
            shutil.copy2(path, destination)
        assert sha(path) == sha(destination)
    files = [p for p in IMG.rglob('*') if p.is_file() and p not in additions]
    files += [p for p in OUT.rglob('*') if p.is_file() and DISPLAY_OUTPUT not in p.parents and p.name != 'current_display.json']
    files.append(ROOT / 'vizpub/supplementary/fig3_lexical_delay_condition_sets.ipynb')
    for folder in ['img/supplementary/fig3_lexical_delay_condition_sets', 'results/supplementary/fig3_lexical_delay_condition_sets']:
        files += [p for p in (ROOT / folder).rglob('*') if p.is_file()]
    frozen = {str(p): sha(p) for p in sorted(set(files))}
    write(AUDIT / 'protected_existing_outputs.json', frozen)
    original = nbformat.read(NB, as_version=4)
    new = nbformat.v4.new_notebook(cells=notebook_cells())
    ids = {c.id for c in new.cells}
    assert ids <= {c.id for c in original.cells}
    untouched = [c for c in original.cells if c.id not in ids]
    kernel = AUDIT / 'jupyter/kernels/picture-top-weight-local'
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / 'kernel.json').write_text(json.dumps(dict(argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        display_name='PN unchanged top-weight fields, Figure 1 local insula', language='python')))
    os.environ['JUPYTER_PATH'] = str(AUDIT / 'jupyter') + os.pathsep + os.environ.get('JUPYTER_PATH', '')
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    try:
        NotebookClient(new, kernel_name='picture-top-weight-local', timeout=1200,
            resources={'metadata': {'path': str(NB.parent)}}).execute()
    except Exception:
        nbformat.write(new, AUDIT / 'failed.ipynb')
        raise
    assert all(c.execution_count is not None for c in new.cells if c.cell_type == 'code')
    assert not [o for c in new.cells for o in c.get('outputs', []) if o.get('output_type') == 'error' or 'image/png' in o.get('data', {})]
    assert sha(NB) == original_sha, 'Notebook changed during display refresh'
    assert all(sha(p) == h for p, h in frozen.items()), 'A saved data file or unrelated figure changed'
    replacements = {c.id: c for c in new.cells}
    original.cells = [replacements.get(c.id, c) for c in original.cells]
    assert [c for c in original.cells if c.id not in ids] == untouched
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
        earlier_notebook_sections_and_outputs_preserved=True, replaced_owned_cell_ids=sorted(ids),
        existing_sources_and_outputs_preserved=len(frozen), main_protected_files_unchanged=main_count,
        exports=exports, original_images_archived=str(archive), no_composite_created=True,
        provenance=json.loads((DISPLAY_OUTPUT / 'provenance.json').read_text()),
        completed_utc=datetime.now(timezone.utc).isoformat()))
    write(OUT / 'current_display.json', dict(completion=str(AUDIT / 'completion.json'), notebook=str(NB), steps=8,
        scope='Three independent top-weight Gaussian maps on clean Figure 1 local-insula backgrounds; numeric fields unchanged'))
    print('COMPLETE: clean Figure 1 local insula, saved Gaussian data unchanged', json.dumps(exports, indent=2), flush=True)


if __name__ == '__main__':
    main()
