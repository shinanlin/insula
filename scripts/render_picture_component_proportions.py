"""Append approved Venn-region composition and continuous Gaussian maps."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET


def main():
    assert os.environ.get('SLURM_JOB_ID'),'Use Slurm'
    import nbformat
    from nbclient import NotebookClient
    from scripts.refresh_lexical_condition_sets import ROOT,protected,sha,write
    from scripts.refresh_picture_condition_sets import NB,OUT,IMG,protected as picture_protected
    from src.supplementary.picture_component_proportions import AUDIT,OUTPUT,EXPORTS,notebook_cells
    AUDIT.mkdir(parents=True,exist_ok=True)
    protected();picture_protected()
    current=json.loads((OUT/'current_display.json').read_text())
    previous_report=json.loads(Path(current['completion']).read_text())
    assert previous_report['passed'] and sha(NB)==previous_report.get('notebook_after_sha256',previous_report.get('notebook_sha256'))
    original=nbformat.read(NB,as_version=4);original_sha=sha(NB)
    archive=AUDIT/'notebook_before.ipynb'
    if not archive.exists():shutil.copy2(NB,archive)
    assert sha(archive)==original_sha,'Notebook changed after task snapshot'
    shutil.copy2(OUT/'current_display.json',AUDIT/'current_display_before.json')
    additions={IMG/name for name in EXPORTS.values()}
    files=[p for p in IMG.rglob('*') if p.is_file() and p not in additions]
    files += [p for p in OUT.rglob('*') if p.is_file() and OUTPUT not in p.parents and p.name!='current_display.json']
    files += [ROOT/'vizpub/supplementary/fig3_lexical_delay_condition_sets.ipynb']
    for folder in ['img/supplementary/fig3_lexical_delay_condition_sets','results/supplementary/fig3_lexical_delay_condition_sets']:
        files += [p for p in (ROOT/folder).rglob('*') if p.is_file()]
    frozen={str(p):sha(p) for p in sorted(set(files))}
    write(AUDIT/'protected_existing_outputs.json',frozen)
    kernel=AUDIT/'jupyter/kernels/picture-component-proportions'
    kernel.mkdir(parents=True,exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        display_name='PN regional component composition',language='python')))
    os.environ['JUPYTER_PATH']=str(AUDIT/'jupyter')+os.pathsep+os.environ.get('JUPYTER_PATH','')
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    new=nbformat.v4.new_notebook(cells=notebook_cells())
    try:
        NotebookClient(new,kernel_name='picture-component-proportions',timeout=1800,
            resources={'metadata':{'path':str(NB.parent)}}).execute()
    except Exception:
        nbformat.write(new,AUDIT/'failed.ipynb');raise
    assert all(c.execution_count is not None for c in new.cells if c.cell_type=='code')
    assert not [o for c in new.cells for o in c.get('outputs',[]) if o.get('output_type')=='error' or 'image/png' in o.get('data',{})]
    assert sha(NB)==original_sha,'Notebook changed during rendering; preserve author edits'
    assert all(sha(p)==h for p,h in frozen.items()),'An existing source, statistic or figure changed'
    ids={c.id for c in new.cells}
    assert not ids & {c.id for c in original.cells},'New composition cells already present'
    old_cells=list(original.cells);original.cells.extend(new.cells)
    assert original.cells[:-len(new.cells)]==old_cells
    exports={}
    for path in additions:
        tree=ET.fromstring(path.read_text());identifiers=[e.attrib['id'] for e in tree.iter() if 'id' in e.attrib]
        assert len(identifiers)==len(set(identifiers)),path
        exports[str(path)]=sha(path)
    main_count=protected();picture_protected()
    nbformat.write(original,NB)
    write(AUDIT/'completion.json',dict(passed=True,job_id=os.environ['SLURM_JOB_ID'],notebook=str(NB),
        notebook_before_sha256=original_sha,notebook_after_sha256=sha(NB),
        original_notebook_cells_and_outputs_preserved=True,existing_sources_and_outputs_preserved=len(frozen),
        main_protected_files_unchanged=main_count,
        no_refit_or_new_statistical_test=True,exports=exports,
        provenance=json.loads((OUTPUT/'provenance.json').read_text()),
        completed_utc=datetime.now(timezone.utc).isoformat()))
    write(OUT/'current_display.json',dict(completion=str(AUDIT/'completion.json'),notebook=str(NB),steps=5,
        scope='Response sets, paired expression, relative component compositions and Gaussian spatial maps'))
    print('COMPLETE',json.dumps(exports,indent=2),flush=True)


if __name__=='__main__':main()
