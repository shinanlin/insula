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

Each electrode contributes all three `loading_c / sum(loading_sensory, loading_sustain, loading_motor)` proportions. Two **ternary composition landscapes** retain the original joint coordinates. Every electrode's point color is the continuous weighted sRGB blend of Sensory blue `#2369BD`, Sustained gold `#C4A35A` and Motor red `#A9373B`, with the same proportion vector as its position. No argmax/hard functional label is used to choose its color. Corner labels and the three legend swatches show the pure-component color anchors; intermediate colors represent mixed weights. The dashed line means Sustained = Motor, and the open diamond is the electrode-equal mean composition. Construction guides are limited to50%.

The manifold-like appearance is a **descriptive density overlay**, without learning a new embedding. Isotropic Gaussian display kernels have the same sigma0.09 (triangle side length=1) in both groups. Each group's kernel sum is divided by its electrode count, and both groups share the same peak-based gray density scale. Clouds/contours are clipped to valid weight compositions; contour levels are20%,40%,60%,80% of that shared density peak. Gray cloud intensity describes local concentration, while point hue describes functional composition. Neither indicates HGA amplitude, confidence interval or significance. Original point coordinates have no jitter/smoothing and are inverted back to verify the three proportions. No new inference, NMF refit or changes to main/SI panels.
'''),nbf.v4.new_code_cell("import sys\nsys.path.insert(0,'/cwork/ns458/insula-paper')\nfrom IPython.display import SVG,display\nimport pandas as pd\nfrom src.diagnostics.figure3_weight_distributions import render,RESULTS\nsvg=render()\ndisplay(SVG(filename=str(svg)))\ndisplay(pd.read_csv(RESULTS/'descriptive_summary.csv'))\n")])
    try:NotebookClient(nb,kernel_name='fig3-weight-distributions',timeout=600,resources={'metadata':{'path':str(ROOT)}}).execute()
    except Exception:nbf.write(nb,AUDIT/'failed.ipynb');raise
    assert not [o for c in nb.cells for o in c.get('outputs',[]) if o.get('output_type')=='error' or 'image/png' in o.get('data',{})]
    destination=ROOT/'vizpub/diagnostics/fig3_weight_distributions.ipynb'
    nbf.write(nb,destination)
    print('COMPLETE',destination,flush=True)


if __name__=='__main__':main()
