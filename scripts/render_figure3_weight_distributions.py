"""Execute the nonspatial Figure 3 component-weight distribution draft."""
from pathlib import Path
import json,os,sys
import nbformat as nbf
from nbclient import NotebookClient

ROOT=Path('/cwork/ns458/insula-paper')
AUDIT=ROOT/'results/audit/figure3_weight_distribution_2026-11-02'


def main():
    assert os.environ.get('SLURM_JOB_ID'),'Use Slurm'
    kernel=AUDIT/'jupyter/kernels/fig3-weight-distributions';kernel.mkdir(parents=True,exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        display_name='Figure 3 weight distributions',language='python')))
    os.environ['JUPYTER_PATH']=str(AUDIT/'jupyter')+os.pathsep+os.environ.get('JUPYTER_PATH','')
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('''# Figure 3 · Component-weight distributions · review draft

Nonspatial distributions of the frozen continuous NMF weights. Retain the same 21 Picture Naming no-detected-Sound–Image-difference electrodes / 14 participants and 27 Lexical Delay Decision > Repeat electrodes / 16 participants. The first set is not a literal baseline-response intersection. The rejected brain-map draft and its dedicated code/caches have been removed.

Each electrode contributes all three `loading_c / sum(loading_sensory, loading_sustain, loading_motor)` proportions. Each component is displayed as a smooth **Gaussian-kernel density** curve, using Scott's bandwidth rule, reflecting at 0 and 1 and normalizing to unit area on [0,1]. Curves describe the observed weights; normality of the electrode weights is not assumed. Both groups share the same x/y ranges. Group summaries are descriptive and electrode-equal; no new test, normalization of HGA, condition-specific fit, NMF refit or change to main/SI panels. The previous box/scatter/paired-line display has been replaced.
'''),nbf.v4.new_code_cell("import sys\nsys.path.insert(0,'/cwork/ns458/insula-paper')\nfrom IPython.display import SVG,display\nimport pandas as pd\nfrom src.diagnostics.figure3_weight_distributions import render,RESULTS\nsvg=render()\ndisplay(SVG(filename=str(svg)))\ndisplay(pd.read_csv(RESULTS/'descriptive_summary.csv'))\n")])
    try:NotebookClient(nb,kernel_name='fig3-weight-distributions',timeout=600,resources={'metadata':{'path':str(ROOT)}}).execute()
    except Exception:nbf.write(nb,AUDIT/'failed.ipynb');raise
    assert not [o for c in nb.cells for o in c.get('outputs',[]) if o.get('output_type')=='error' or 'image/png' in o.get('data',{})]
    destination=ROOT/'vizpub/diagnostics/fig3_weight_distributions.ipynb'
    nbf.write(nb,destination)
    print('COMPLETE',destination,flush=True)


if __name__=='__main__':main()
