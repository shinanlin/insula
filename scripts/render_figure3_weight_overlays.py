"""Redraw the two overlaid density triangles and refresh that notebook cell."""
from pathlib import Path
import json, os, sys
import nbformat as nbf
from nbclient import NotebookClient

ROOT=Path('/cwork/ns458/insula-paper')
sys.path.insert(0,str(ROOT))
NOTEBOOK=ROOT/'vizpub/fig3.ipynb'
AUDIT=ROOT/'results/audit/figure3_weight_pairs_2026-11-07'


def main():
    assert os.environ.get('SLURM_JOB_ID'),'Use Slurm'
    kernel=AUDIT/'jupyter/kernels/fig3-weight-pairs'
    kernel.mkdir(parents=True,exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        display_name='Figure 3 weight pairs',language='python')))
    os.environ['JUPYTER_PATH']=str(AUDIT/'jupyter')+os.pathsep+os.environ.get('JUPYTER_PATH','')
    nb=nbf.read(NOTEBOOK,as_version=4)
    source=next(cell.source for cell in nb.cells if cell.id=='fig3-weight-overlays')
    small=nbf.v4.new_notebook(cells=[nbf.v4.new_code_cell(
        "import sys\nsys.path.insert(0,'/cwork/ns458/insula-paper')\nfrom IPython.display import SVG,display\n",id='imports'),
        nbf.v4.new_code_cell(source,id='fig3-weight-overlays')])
    try:NotebookClient(small,kernel_name='fig3-weight-pairs',timeout=600,
        resources={'metadata':{'path':str(ROOT)}}).execute()
    except Exception:
        nbf.write(small,AUDIT/'overlay_failed.ipynb');raise
    assert not [o for c in small.cells for o in c.get('outputs',[]) if o.get('output_type')=='error' or 'image/png' in o.get('data',{})]
    produced=next(cell for cell in small.cells if cell.id=='fig3-weight-overlays')
    for cell in nb.cells:
        if cell.id=='fig3-weight-overlays':
            cell.outputs=produced.outputs
            cell.execution_count=produced.execution_count
    nbf.write(nb,NOTEBOOK)
    print('COMPLETE',NOTEBOOK,flush=True)


if __name__=='__main__':main()
