"""Draw the two missing Figure 3 task-matched weight pairs and store their notebook outputs."""
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
    wanted=('fig3-weight-picture-pair','fig3-weight-lexical-pair','fig3-weight-overlays')
    sources={cell.id:cell.source for cell in nb.cells if cell.id in wanted}
    assert set(sources)==set(wanted)
    small=nbf.v4.new_notebook(cells=[nbf.v4.new_code_cell(
        "import sys\nsys.path.insert(0,'/cwork/ns458/insula-paper')\nfrom IPython.display import SVG,display\n",id='imports'),
        nbf.v4.new_code_cell(sources['fig3-weight-picture-pair'],id='fig3-weight-picture-pair'),
        nbf.v4.new_code_cell(sources['fig3-weight-lexical-pair'],id='fig3-weight-lexical-pair'),
        nbf.v4.new_code_cell(sources['fig3-weight-overlays'],id='fig3-weight-overlays')])
    try:NotebookClient(small,kernel_name='fig3-weight-pairs',timeout=600,
        resources={'metadata':{'path':str(ROOT)}}).execute()
    except Exception:
        nbf.write(small,AUDIT/'failed.ipynb');raise
    assert not [o for c in small.cells for o in c.get('outputs',[]) if o.get('output_type')=='error' or 'image/png' in o.get('data',{})]
    produced={cell.id:cell for cell in small.cells}
    for cell in nb.cells:
        if cell.id in wanted:
            cell.outputs=produced[cell.id].outputs
            cell.execution_count=produced[cell.id].execution_count
    nbf.write(nb,NOTEBOOK)
    print('COMPLETE',NOTEBOOK,flush=True)


if __name__=='__main__':main()
