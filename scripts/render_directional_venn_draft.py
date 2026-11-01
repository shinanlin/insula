"""Execute a separate review notebook, preserving all current figures."""
from pathlib import Path
import json
import os
import sys
import nbformat
from nbclient import NotebookClient

ROOT = Path('/cwork/ns458/insula-paper')
AUDIT = ROOT/'results/supplementary/fig3_directional_venn_draft_2026-10-28'


def main():
    assert os.environ.get('SLURM_JOB_ID')
    kernel = AUDIT/'jupyter/kernels/directional-venn-draft'
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,
        '-m','ipykernel_launcher','-f','{connection_file}'],
        display_name='Direct condition-difference Venn draft',language='python')))
    os.environ['JUPYTER_PATH'] = str(AUDIT/'jupyter')+os.pathsep+os.environ.get('JUPYTER_PATH','')
    cells = [nbformat.v4.new_markdown_cell(
        '# Figure 3: direct condition-difference Venn drafts\n\n'
        'Independent review drafts; existing figures are unchanged. Each parent pool contains '
        'anatomically eligible sites with either-condition native baseline responsiveness in '
        'the target phase (0–1 s). No NMF membership gate.\n\n'
        'Circles represent the two significant directions from the existing per-electrode '
        'two-sided time_perm_cluster (2,000 permutations, p_thresh=.05, p_cluster=.05). '
        'Their intersection means opposite effects at different times; the outside region '
        'means no detected difference, not equivalence. Participant counts overlap across groups. '
        'Venn colors encode difference directions; existing brain maps keep the functional '
        'component colors. Counts and labels are reused, with no new tests or NMF fitting.'),
        nbformat.v4.new_code_cell("import sys\nsys.path.insert(0,'/cwork/ns458/insula-paper')\n"
            "from IPython.display import SVG, display\nimport pandas as pd\n"
            "from src.supplementary.directional_venn_draft import run, OUT\nresults = run()")]
    for key,title in [('picture','Picture Naming · Stimulus 0–1 s'),('lexical','Lexical Delay · Delay 0–1 s')]:
        cells += [nbformat.v4.new_markdown_cell('## '+title),
            nbformat.v4.new_code_cell(f"display(SVG(filename=results['{key}']['image']))\n"
                f"display(pd.read_csv(OUT/'{key}'/'group_counts.csv'))")]
    nb = nbformat.v4.new_notebook(cells=cells)
    NotebookClient(nb,kernel_name='directional-venn-draft',timeout=600,
        resources={'metadata':{'path':str(ROOT)}}).execute()
    assert not [o for c in nb.cells for o in c.get('outputs',[]) if
        o.get('output_type')=='error' or 'image/png' in o.get('data',{})]
    path = ROOT/'vizpub/supplementary/fig3_directional_difference_venn_draft.ipynb'
    nbformat.write(nb,path)
    print('COMPLETE '+str(path),flush=True)


if __name__=='__main__': main()
