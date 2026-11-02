"""Execute the independent Figure 3 continuous-loading draft notebook on Slurm."""
from pathlib import Path
import json
import os
import sys
import nbformat as nbf
from nbclient import NotebookClient

ROOT=Path('/cwork/ns458/insula-paper')
AUDIT=ROOT/'results/audit/fig3_continuous_weights_2026-11-02'


def main():
    assert os.environ.get('SLURM_JOB_ID'),'Use Slurm'
    AUDIT.mkdir(parents=True,exist_ok=True)
    kernel=AUDIT/'jupyter/kernels/fig3-continuous-weights';kernel.mkdir(parents=True,exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        display_name='Figure 3 continuous weights',language='python')))
    os.environ['JUPYTER_PATH']=str(AUDIT/'jupyter')+os.pathsep+os.environ.get('JUPYTER_PATH','')
    cells=[nbf.v4.new_markdown_cell('''# Figure 3 · Continuous NMF weights · review draft

Two rows: the **21 Sound–Image no-detected-difference electrodes / 14 participants** in the current main figure, and the **27 Delay Decision > Repeat electrodes / 16 participants**. The first row is not the baseline-response intersection.

Three columns show each original component continuously: Sensory, Sustained and Motor. At each electrode, color encodes `loading_component / (loading_sensory + loading_sustain + loading_motor)` with identical 0–1 limits. The original H rows have unit L2 norm. We reuse the frozen sound-only model and do not fit condition-specific factors. Group summaries are electrode-equal descriptive means.

Use the same clean local-insula surface, coordinates, camera-facing projection, 4-mm lift, 24-pixel points and hemisphere views as the accepted main maps. All selected sites appear in all three columns; their argmax labels do not determine display colors. Each component uses its own white-to-component-color scale. No RGB mixing, interpolation, spatial smoothing, new inference or changes to main figures/NMF. The cropping is fixed using the original maps, so low loading values cannot change the framing.
'''),nbf.v4.new_code_cell("import sys\nsys.path.insert(0,'/cwork/ns458/insula-paper')\nfrom IPython.display import SVG,display\nimport pandas as pd\nfrom src.diagnostics.figure3_continuous_weights import render,RESULTS\nsvg=render()\ndisplay(SVG(filename=str(svg)))\n"),
        nbf.v4.new_code_cell("display(pd.read_csv(RESULTS/'group_mean_proportions.csv'))\n")]
    notebook=nbf.v4.new_notebook(cells=cells)
    try:
        NotebookClient(notebook,kernel_name='fig3-continuous-weights',timeout=1800,
            resources={'metadata':{'path':str(ROOT)}}).execute()
    except Exception:
        nbf.write(notebook,AUDIT/'failed.ipynb');raise
    assert not [output for cell in notebook.cells for output in cell.get('outputs',[])
        if output.get('output_type')=='error' or 'image/png' in output.get('data',{})]
    destination=ROOT/'vizpub/diagnostics/fig3_continuous_weights.ipynb'
    destination.parent.mkdir(parents=True,exist_ok=True);nbf.write(notebook,destination)
    print('COMPLETE',destination,flush=True)


if __name__=='__main__':main()
