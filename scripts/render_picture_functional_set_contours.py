"""Append only the authorized functional-set maps to the PN notebook."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET


def sync_reviewed_lexical_panel(root,audit,sha,write):
    # The preceding author-approved addition changed only the Lexical notebook.
    registry=root/'results/audit/picture_condition_sets_full_2026-10-27/protected.json'
    hashes=json.loads(registry.read_text())
    lexical=str(root/'vizpub/supplementary/fig3_lexical_delay_condition_sets.ipynb')
    approved=json.loads((root/'results/audit/lexical_sampling_response_2026-10-27/completion.json').read_text())
    assert approved['passed'] and approved['original_notebook_cells_and_outputs_preserved']
    assert sha(lexical)==approved['notebook_after_sha256']
    others=[p for p,h in hashes.items() if p!=lexical and sha(p)!=h]
    assert not others,others
    old=hashes[lexical]
    if old!=approved['notebook_after_sha256']:
        assert old==approved['notebook_before_sha256']
        shutil.copy2(registry,audit/'picture_protected_before_approved_lexical_addition.json')
        hashes[lexical]=approved['notebook_after_sha256']
        write(registry,hashes)
    write(audit/'approved_lexical_notebook_registry_sync.json',dict(
        author_request='Add Lexical sampling/response panel on 2026-10-27',
        verified_completion=str(root/'results/audit/lexical_sampling_response_2026-10-27/completion.json'),
        old_hash=old,new_hash=hashes[lexical],other_registry_entries_unchanged=True))


def main():
    assert os.environ.get('SLURM_JOB_ID'),'Use Slurm'
    import nbformat
    from nbclient import NotebookClient
    from scripts.refresh_lexical_condition_sets import protected,sha,ROOT,write
    from scripts.refresh_picture_condition_sets import NB,IMG,OUT,protected as picture_protected
    from src.supplementary.picture_functional_set_contours import AUDIT,DATA,EXPORT_NAMES,SHEET,notebook_cells
    AUDIT.mkdir(parents=True,exist_ok=True)
    protected()
    sync_reviewed_lexical_panel(ROOT,AUDIT,sha,write)
    picture_protected()
    original=nbformat.read(NB,as_version=4);original_sha=sha(NB)
    archive=AUDIT/'notebook_before.ipynb'
    if not archive.exists():shutil.copy2(NB,archive)
    assert sha(archive)==original_sha,'Notebook changed after its pre-edit archive'
    additions={IMG/name for name in list(EXPORT_NAMES.values())+[SHEET]}
    files=[p for p in IMG.glob('*.svg') if p not in additions]
    files += [p for p in OUT.rglob('*') if p.is_file() and DATA not in p.parents and p.name!='current_display.json']
    files += [ROOT/'vizpub/supplementary/fig3_lexical_delay_condition_sets.ipynb']
    for folder in ['img/supplementary/fig3_lexical_delay_condition_sets','results/supplementary/fig3_lexical_delay_condition_sets']:
        files += [p for p in (ROOT/folder).rglob('*') if p.is_file()]
    files.append(ROOT/'results/audit/picture_condition_sets_full_2026-10-27/protected.json')
    frozen={str(p):sha(p) for p in sorted(set(files))}
    write(AUDIT/'protected_existing_outputs.json',frozen)
    shutil.copy2(OUT/'current_display.json',AUDIT/'current_display_before.json')
    kernel=AUDIT/'jupyter/kernels/picture-functional-sets'
    kernel.mkdir(parents=True,exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],display_name='Picture response sets by component',language='python')))
    os.environ['JUPYTER_PATH']=str(AUDIT/'jupyter')+os.pathsep+os.environ.get('JUPYTER_PATH','')
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    new=nbformat.v4.new_notebook(cells=notebook_cells())
    try:
        NotebookClient(new,kernel_name='picture-functional-sets',timeout=1200,
            resources={'metadata':{'path':str(NB.parent)}}).execute()
    except Exception:
        nbformat.write(new,AUDIT/'failed.ipynb');raise
    assert all(c.execution_count is not None for c in new.cells if c.cell_type=='code')
    assert not [o for c in new.cells for o in c.get('outputs',[]) if o.get('output_type')=='error' or 'image/png' in o.get('data',{})]
    assert sha(NB)==original_sha,'Notebook changed during rendering; retain author edits'
    assert all(sha(p)==h for p,h in frozen.items()),'An existing figure or scientific result changed'
    ids={c.id for c in new.cells}
    retained=[c for c in original.cells if c.id not in ids]
    original.cells=retained+new.cells
    assert [c for c in original.cells if c.id not in ids]==retained
    exports={}
    for p in sorted(additions):
        tree=ET.fromstring(p.read_text())
        identifiers=[e.attrib['id'] for e in tree.iter() if 'id' in e.attrib]
        assert len(identifiers)==len(set(identifiers)),p
        exports[str(p)]=sha(p)
    protected_count=protected();picture_protected()
    nbformat.write(original,NB)
    write(AUDIT/'completion.json',dict(passed=True,job_id=os.environ['SLURM_JOB_ID'],
        notebook=str(NB),notebook_before_sha256=original_sha,notebook_after_sha256=sha(NB),
        original_notebook_cells_and_outputs_preserved=True,original_output_files_preserved=len(frozen),
        main_protected_files_unchanged=protected_count,exports=exports,
        panel_provenance=json.loads((DATA/'provenance.json').read_text()),
        completed_utc=datetime.now(timezone.utc).isoformat()))
    write(OUT/'current_display.json',dict(completion=str(AUDIT/'completion.json'),notebook=str(NB),steps=3,
        scope='Baseline response sets and frozen functional-color spatial-density contours'))
    print('COMPLETE',IMG/SHEET,flush=True)


if __name__=='__main__':main()
