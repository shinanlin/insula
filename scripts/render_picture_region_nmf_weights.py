"""Replace only the superseded six-map section with three original-W maps."""
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
    from src.supplementary.picture_region_nmf_weights import AUDIT,OUTPUT,EXPORTS,notebook_cells
    AUDIT.mkdir(parents=True,exist_ok=True)
    protected();picture_protected()
    current=json.loads((OUT/'current_display.json').read_text())
    prior=json.loads(Path(current['completion']).read_text())
    assert prior['passed'] and sha(NB)==prior.get('notebook_after_sha256',prior.get('notebook_sha256'))
    original=nbformat.read(NB,as_version=4);original_sha=sha(NB)
    archive=AUDIT/'notebook_before.ipynb'
    if not archive.exists():shutil.copy2(NB,archive)
    assert sha(archive)==original_sha,'Notebook changed since archive'
    shutil.copy2(OUT/'current_display.json',AUDIT/'current_display_before.json')
    replacements={IMG/name for name in EXPORTS.values()}
    before_exports={}
    for path in replacements:
        destination=AUDIT/'before'/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():shutil.copy2(path,destination)
        assert sha(destination)==sha(path)
        before_exports[str(path)]=dict(sha256=sha(path),archive=str(destination))
    write(AUDIT/'replaced_exports_before.json',before_exports)
    files=[p for p in IMG.rglob('*') if p.is_file() and p not in replacements]
    files += [p for p in OUT.rglob('*') if p.is_file() and OUTPUT not in p.parents and p.name!='current_display.json']
    files.append(ROOT/'vizpub/supplementary/fig3_lexical_delay_condition_sets.ipynb')
    for folder in ['img/supplementary/fig3_lexical_delay_condition_sets','results/supplementary/fig3_lexical_delay_condition_sets']:
        files += [p for p in (ROOT/folder).rglob('*') if p.is_file()]
    frozen={str(p):sha(p) for p in sorted(set(files))}
    write(AUDIT/'protected_existing_outputs.json',frozen)
    new=nbformat.v4.new_notebook(cells=notebook_cells())
    ids={c.id for c in new.cells}
    old_owned=[c for c in original.cells if c.id in ids]
    assert len(old_owned)==2 and {c.id for c in old_owned}==ids
    assert any('src.supplementary.picture_component_proportions' in c.source for c in old_owned)
    kernel=AUDIT/'jupyter/kernels/picture-original-nmf-regions'
    kernel.mkdir(parents=True,exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        display_name='Three regions, original frozen NMF W',language='python')))
    os.environ['JUPYTER_PATH']=str(AUDIT/'jupyter')+os.pathsep+os.environ.get('JUPYTER_PATH','')
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    try:
        NotebookClient(new,kernel_name='picture-original-nmf-regions',timeout=1200,
            resources={'metadata':{'path':str(NB.parent)}}).execute()
    except Exception:
        nbformat.write(new,AUDIT/'failed.ipynb');raise
    assert all(c.execution_count is not None for c in new.cells if c.cell_type=='code')
    assert not [o for c in new.cells for o in c.get('outputs',[]) if o.get('output_type')=='error' or 'image/png' in o.get('data',{})]
    assert sha(NB)==original_sha,'Notebook changed during rendering; retain author edits'
    assert all(sha(p)==h for p,h in frozen.items()),'An unrelated analysis, figure or input changed'
    untouched=[c for c in original.cells if c.id not in ids]
    by_id={c.id:c for c in new.cells}
    original.cells=[by_id.get(c.id,c) for c in original.cells]
    assert [c for c in original.cells if c.id not in ids]==untouched
    exports={}
    for path in replacements:
        tree=ET.fromstring(path.read_text());identifiers=[e.attrib['id'] for e in tree.iter() if 'id' in e.attrib]
        assert len(identifiers)==len(set(identifiers)),path
        exports[str(path)]=sha(path)
    assert all(sha(r['archive'])==r['sha256'] for r in before_exports.values())
    main_count=protected();picture_protected()
    nbformat.write(original,NB)
    write(AUDIT/'completion.json',dict(passed=True,job_id=os.environ['SLURM_JOB_ID'],notebook=str(NB),
        notebook_before_sha256=original_sha,notebook_after_sha256=sha(NB),replaced_cell_ids=sorted(ids),
        other_notebook_cells_and_outputs_preserved=True,prior_six_map_exports_archived=True,
        existing_scientific_sources_and_outputs_preserved=len(frozen),main_protected_files_unchanged=main_count,
        n_bilateral_maps=3,n_original_NMF_weights=44,n_unassigned=5,
        exports=exports,provenance=json.loads((OUTPUT/'provenance.json').read_text()),
        completed_utc=datetime.now(timezone.utc).isoformat()))
    write(OUT/'current_display.json',dict(completion=str(AUDIT/'completion.json'),notebook=str(NB),steps=5,
        scope='Three response regions characterized by original frozen NMF weights; five Unassigned sites explicit'))
    print('COMPLETE: THREE bilateral maps, original NMF W',json.dumps(exports,indent=2),flush=True)


if __name__=='__main__':main()
