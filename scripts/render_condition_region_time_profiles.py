"""Insert only the two new temporal portraits beside their spatial regions."""
from pathlib import Path
from datetime import datetime,timezone
import json
import os
import shutil
import sys
import nbformat
from nbclient import NotebookClient

ROOT=Path('/cwork/ns458/insula-paper')
AUDIT=ROOT/'results/audit/fig3_condition_time_profiles_2026-11-01'


def main():
    assert os.environ.get('SLURM_JOB_ID')
    from scripts.refresh_lexical_condition_sets import sha,write
    from src.supplementary.condition_region_time_profiles import OUT,NOTEBOOK,notebook_cells
    AUDIT.mkdir(parents=True,exist_ok=True)
    before=AUDIT/'before';before.mkdir(exist_ok=True)
    if not (before/NOTEBOOK.name).exists():shutil.copy2(NOTEBOOK,before/NOTEBOOK.name)
    old_sha=sha(NOTEBOOK)
    nb=nbformat.read(NOTEBOOK,as_version=4)
    new_ids={f'condition-time-{key}-{suffix}' for key in ('picture','lexical') for suffix in ('note','plot')}
    original_cells=[json.dumps(c,sort_keys=True) for c in nb.cells if c.id not in new_ids]
    paths=[]
    for folder in ('results/nmf','img/fig1','img/fig2',
        'results/supplementary/decision_repeat_test','results/supplementary/sound_image_test',
        'results/supplementary/fig3_lexical_delay_condition_sets',
        'results/supplementary/fig3_picture_stimulus_condition_sets',
        'results/supplementary/fig3_condition_difference_groups_2026-10-28',
        'img/supplementary/fig3_condition_difference_groups',
        'img/supplementary/decision_repeat_test','img/supplementary/sound_image_test'):
        paths += [p for p in (ROOT/folder).rglob('*') if p.is_file()]
    paths += list((ROOT/'vizpub').glob('fig[1-6].ipynb'))
    paths += [p for p in (ROOT/'vizpub/supplementary').glob('*.ipynb') if p!=NOTEBOOK]
    frozen={str(p):sha(p) for p in sorted(set(paths))};write(AUDIT/'protected_existing_files.json',frozen)
    kernel=AUDIT/'jupyter/kernels/condition-region-time'
    kernel.mkdir(parents=True,exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        display_name='Audited condition-region temporal profiles',language='python')))
    os.environ['JUPYTER_PATH']=str(AUDIT/'jupyter')+os.pathsep+os.environ.get('JUPYTER_PATH','')
    context=nbformat.v4.new_code_cell("import sys\nsys.path.insert(0,'/cwork/ns458/insula-paper')\nfrom IPython.display import SVG,display\n")
    addition=nbformat.v4.new_notebook(cells=[context]+notebook_cells('picture')+notebook_cells('lexical'))
    try:NotebookClient(addition,kernel_name='condition-region-time',timeout=1200,
        resources={'metadata':{'path':str(ROOT)}}).execute()
    except Exception:nbformat.write(addition,AUDIT/'failed.ipynb');raise
    assert not [o for c in addition.cells for o in c.get('outputs',[]) if
        o.get('output_type')=='error' or 'image/png' in o.get('data',{})]
    assert sha(NOTEBOOK)==old_sha and all(sha(p)==h for p,h in frozen.items())
    nb.cells=[c for c in nb.cells if c.id not in new_ids]
    for key,marker in [('picture',"display(SVG(filename=results['picture']['brains']['no_detected_difference']))"),
        ('lexical',"display(SVG(filename=results['lexical']['brains']['first_only']))")]:
        index=next(i for i,c in enumerate(nb.cells) if marker in c.source)
        owned=[c for c in addition.cells[1:] if c.id in {f'condition-time-{key}-note',f'condition-time-{key}-plot'}]
        assert len(owned)==2
        nb.cells=nb.cells[:index+1]+owned+nb.cells[index+1:]
    assert [json.dumps(c,sort_keys=True) for c in nb.cells if c.id not in new_ids]==original_cells
    nbformat.write(nb,NOTEBOOK)
    write(AUDIT/'completion.json',dict(passed=True,job_id=os.environ['SLURM_JOB_ID'],
        completed_utc=datetime.now(timezone.utc).isoformat(),notebook=str(NOTEBOOK),
        notebook_before_sha256=old_sha,notebook_after_sha256=sha(NOTEBOOK),
        prior_cells_and_outputs_preserved=len(original_cells),new_cells=sorted(new_ids),
        existing_files_unchanged=len(frozen),original_nine_spatial_exports_unchanged=True,
        no_new_statistics=True,no_NMF_changes=True,
        profiles={key:json.loads((OUT/key/'plot_validation.json').read_text()) for key in ('picture','lexical')}))
    print('COMPLETE: PN25 average and LD28 difference time profiles added to the same notebook',flush=True)


if __name__=='__main__':main()
