"""Update just the two area-proportional diagrams and their notebook outputs."""
from pathlib import Path
import json
import os
import shutil
import nbformat
import numpy as np
import pandas as pd

ROOT=Path('/cwork/ns458/insula-paper')
AUDIT=ROOT/'results/audit/fig3_condition_difference_areas_2026-10-28'


def main():
    assert os.environ.get('SLURM_JOB_ID')
    from scripts.refresh_lexical_condition_sets import protected,sha,write
    from scripts.refresh_picture_condition_sets import protected as pn_protected
    from src.supplementary.condition_difference_group_panels import OUT,IMG,NOTEBOOK,SPECS,diagram
    protected();pn_protected()
    AUDIT.mkdir(parents=True,exist_ok=True)
    before=AUDIT/'before';before.mkdir(exist_ok=True)
    changed={IMG/f'fig3_{key}_condition_difference_schematic.svg' for key in SPECS}
    for path in [NOTEBOOK,*sorted(changed)]:
        target=before/path.name
        if not target.exists():shutil.copy2(path,target)
    original_nb_hash=sha(NOTEBOOK)
    frozen=json.loads((OUT/'protected_existing_files.json').read_text())
    frozen.update({str(p):sha(p) for p in IMG.glob('*_functional_brain.svg')})
    frozen.update({str(p):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='diagram_area_provenance.json'})
    assert all(sha(p)==h for p,h in frozen.items())
    write(AUDIT/'protected_files.json',frozen)
    nb=nbformat.read(NOTEBOOK,as_version=4)
    originals=[json.dumps(c,sort_keys=True) for c in nb.cells]
    patched={0};areas={}
    for key in SPECS:
        pool=pd.read_csv(OUT/key/'responsive_pool_electrodes.csv')
        groups={name:pd.read_csv(OUT/key/f'{name}_electrodes.csv') for name in
            ('first_only','no_detected_difference','second_only','bidirectional')}
        assert sum(len(x) for x in groups.values())==len(pool)
        path=diagram(key,pool,groups)
        cell_index=next(i for i,c in enumerate(nb.cells) if
            f"display(SVG(filename=results['{key}']['diagram']))" in c.source)
        outputs=[o for o in nb.cells[cell_index].outputs if 'image/svg+xml' in o.get('data',{})]
        assert len(outputs)==1
        outputs[0].data['image/svg+xml']=path.read_text()
        patched.add(cell_index)
        areas[key]=json.loads((OUT/key/'diagram_area_provenance.json').read_text())
    np.testing.assert_allclose(areas['picture']['pixels_per_data_unit'],
        areas['lexical']['pixels_per_data_unit'],atol=1e-8,rtol=0)
    old='These are schematic three-category classification diagrams, not literal set-intersection Venn diagrams; region areas are not proportional to counts.'
    new='These are three-category classification diagrams, not literal response-set intersections. Region areas are proportional to electrode counts, with the same area per electrode in both panels. Each full circle includes the center: PN circle totals are 44 and 29; Lexical totals are 77 and 50.'
    assert old in nb.cells[0].source or new in nb.cells[0].source
    nb.cells[0].source=nb.cells[0].source.replace(old,new)
    assert all(json.dumps(c,sort_keys=True)==originals[i] for i,c in enumerate(nb.cells) if i not in patched)
    assert sha(NOTEBOOK)==original_nb_hash and all(sha(p)==h for p,h in frozen.items())
    nbformat.write(nb,NOTEBOOK)
    write(AUDIT/'completion.json',dict(passed=True,job_id=os.environ['SLURM_JOB_ID'],
        changed_diagrams=[str(p) for p in sorted(changed)],notebook=str(NOTEBOOK),
        notebook_cells_changed=sorted(patched),all_other_notebook_cells_outputs_preserved=True,
        all_seven_brain_maps_byte_identical=True,existing_files_unchanged=len(frozen),
        main_guard=protected(),picture_guard=pn_protected(),areas=areas,
        source_statistics_and_NMF_unchanged=True))
    print('COMPLETE: two area-proportional diagrams; seven brain maps unchanged',flush=True)


if __name__=='__main__':main()
