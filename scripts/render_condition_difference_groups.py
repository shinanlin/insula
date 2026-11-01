"""Replace the earlier Venn interpretation with the author-approved display."""
from pathlib import Path
import json
import os
import shutil
import sys
import nbformat
from nbclient import NotebookClient

ROOT=Path('/cwork/ns458/insula-paper')
AUDIT=ROOT/'results/audit/exclude_unassigned_2026-11-01'
NB=ROOT/'vizpub/supplementary/fig3_directional_difference_venn_draft.ipynb'


def main():
    assert os.environ.get('SLURM_JOB_ID')
    from scripts.refresh_lexical_condition_sets import sha,write
    AUDIT.mkdir(parents=True,exist_ok=True)
    before=AUDIT/'before';before.mkdir(exist_ok=True)
    archived=before/NB.name
    if not archived.exists():shutil.copy2(NB,archived)
    original_sha=sha(NB)
    kernel=AUDIT/'jupyter/kernels/condition-difference-groups'
    kernel.mkdir(parents=True,exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,
        '-m','ipykernel_launcher','-f','{connection_file}'],
        display_name='Condition-difference groups and functional maps',language='python')))
    os.environ['JUPYTER_PATH']=str(AUDIT/'jupyter')+os.pathsep+os.environ.get('JUPYTER_PATH','')
    cells=[nbformat.v4.new_markdown_cell('''# Figure 3 · Condition-difference categories and functional anatomy

**Current statistical policy, author approved 2026-11-01: exclude every functionally Unassigned electrode.** Use only the frozen Sensory/Sustained/Motor assignments for all denominators, category counts, participants, curves, SEM, heatmap rows and maps below. This supersedes earlier notes retaining gray Unassigned rows. Anatomical sampling inventories remain available separately; anatomical Unknown endpoints are not changed.

**The central overlapping region means no detected significant difference.** These are three-category classification diagrams, not literal response-set intersections. Region areas are proportional to electrode counts, with the same area per electrode in both panels. Each full circle includes the center: PN circle totals are 39 and 25; Lexical totals are 70 and 44. No detected difference is not statistical equivalence, nor does it mean both conditions are baseline-responsive.

Pools contain anatomically eligible, native-good, paired-statistics-complete sites with frozen functional assignments AND either-condition native baseline mask + pointwise p<.05 in the target phase. Picture Naming: anatomical inventory128/27, assigned statistical cohort81/24, Stimulus **0–1 s** responsive pool **44 sites / 21 participants**. Lexical Delay: anatomical inventory187/35, assigned statistical cohort135/35, Delay **0–1 s** responsive pool **71 / 27**. Anatomical totals are coverage information, not the functional statistical denominators.

Use the completed per-electrode two-sided `time_perm_cluster`: 2,000 permutations, p_thresh=.05, p_cluster=.05, original trial rules/seed/correction families and saved directional masks. **No new tests, selection-window changes, anatomy changes, NMF fits or functional reassignment.**

PN has four mutually exclusive groups: Sound > Image only **18**, no detected difference **21**, Image > Sound only **4**, and bidirectional **1** (D0080 LPI2–3, opposite effects at different times). The fourth group is shown separately and is not counted in either one-direction map. Lexical has **27 / 43 / 1**, with no bidirectional site.

The native-file audits verified the parent response-union memberships and difference categories. Excluded Unassigned IDs, source hashes and count flows are saved under `results/supplementary/fig3_assigned_only_2026-11-01/`. D0105 LTPS1–2 is now among the excluded Unassigned sites. The no-difference center is not a baseline-response intersection.

All independent brain SVGs use the same accepted Figure 1 local-insula surfaces/cameras/surface display projection/4-mm lift/24-pixel points and original Figure 2 density contours (minimum 3 sites per component per hemisphere). Functional colors: Sensory blue, Sustained gold, Motor red. Sparse groups retain points only. Contours show descriptive spatial density, not significance boundaries. Frozen analytical coordinates and functional labels are unchanged.
'''),nbformat.v4.new_code_cell("import sys\nsys.path.insert(0,'/cwork/ns458/insula-paper')\n"
        "from IPython.display import SVG,display\nimport pandas as pd\n"
        "from src.supplementary.condition_difference_group_panels import run,OUT\nresults=run()")]
    names={
        'picture':dict(first_only='Sound > Image only · n=18',no_detected_difference='No detected Sound–Image difference · n=21',
            second_only='Image > Sound only · n=4',bidirectional='Bidirectional: D0080 LPI2–3 · n=1'),
        'lexical':dict(first_only='Decision > Repeat · n=27',no_detected_difference='No detected Decision–Repeat difference · n=43',
            second_only='Repeat > Decision · n=1'),
    }
    for key,title,groups in [
        ('picture','Picture Naming · Stimulus 0–1 s',list(names['picture'])),
        ('lexical','Lexical Delay · Delay 0–1 s',list(names['lexical']))]:
        cells += [nbformat.v4.new_markdown_cell('## '+title),
            nbformat.v4.new_code_cell(f"display(SVG(filename=results['{key}']['diagram']))\n"
                f"display(pd.read_csv(OUT/'{key}'/'group_counts.csv'))")]
        for group in groups:
            cells += [nbformat.v4.new_markdown_cell('### '+names[key][group]),
                nbformat.v4.new_code_cell(f"display(SVG(filename=results['{key}']['brains']['{group}']))")]
            if (key,group) in [('picture','no_detected_difference'),('lexical','first_only')]:
                from src.supplementary.condition_region_time_profiles import notebook_cells
                cells += notebook_cells(key)
    nb=nbformat.v4.new_notebook(cells=cells)
    try:
        NotebookClient(nb,kernel_name='condition-difference-groups',timeout=1800,
            resources={'metadata':{'path':str(ROOT)}}).execute()
    except Exception:
        nbformat.write(nb,AUDIT/'failed.ipynb');raise
    assert not [o for c in nb.cells for o in c.get('outputs',[]) if
        o.get('output_type')=='error' or 'image/png' in o.get('data',{})]
    assert sha(NB)==original_sha,'Notebook was edited during execution; do not overwrite'
    nbformat.write(nb,NB)
    write(AUDIT/'notebook_update.json',dict(passed=True,notebook=str(NB),
        archived=str(archived),before_sha256=original_sha,after_sha256=sha(NB),
        main_figures_and_original_statistical_results_unchanged=True))
    print('COMPLETE '+str(NB),flush=True)


if __name__=='__main__':main()
