"""Execute, save, and validate the authorized supplementary notebook only."""
from pathlib import Path
import json
import os
import sys
from datetime import datetime,timezone


def main():
    import nbformat
    from nbclient import NotebookClient
    from scripts.refresh_lexical_condition_sets import ROOT,AUDIT,OUT,NB,IMG,protected,sha,write
    from scripts.build_lexical_condition_notebook import build
    assert os.environ.get('SLURM_JOB_ID')
    protected()
    build()
    kernels=AUDIT/'jupyter/kernels/lexical-full'
    kernels.mkdir(parents=True,exist_ok=True)
    (kernels/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],display_name='Lexical full cohort',language='python')))
    os.environ['JUPYTER_PATH']=str(AUDIT/'jupyter')+os.pathsep+os.environ.get('JUPYTER_PATH','')
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    nb=nbformat.read(NB,as_version=4)
    def started(cell,cell_index,**kw):print(datetime.now(timezone.utc).isoformat(),'cell',cell_index,flush=True)
    try:
        NotebookClient(nb,kernel_name='lexical-full',timeout=1800,iopub_timeout=180,
            resources={'metadata':{'path':str(NB.parent)}},on_cell_start=started).execute()
    except Exception:
        nbformat.write(nb,AUDIT/'failed.ipynb')
        raise
    errors=[o for c in nb.cells for o in c.get('outputs',[]) if o.get('output_type')=='error' or 'image/png' in o.get('data',{})]
    assert not errors
    assert all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')
    protected()
    nbformat.write(nb,NB)
    exports={str(p):sha(p) for p in IMG.glob('*.svg')}
    write(AUDIT/'completion.json',dict(passed=True,notebook=str(NB),notebook_sha256=sha(NB),
        protected_files_unchanged=protected(),exports=exports,
        final_summary=json.loads((OUT/'summary.json').read_text()),
        job_id=os.environ['SLURM_JOB_ID'],completed_utc=datetime.now(timezone.utc).isoformat()))
    print('COMPLETE',NB,flush=True)


if __name__=='__main__':main()
