"""Execute the approved main Figure 3 and its combined supplementary notebook."""
from pathlib import Path
from datetime import datetime,timezone
import json
import os
import shutil
import subprocess
import sys
import nbformat as nbf
from nbclient import NotebookClient

ROOT=Path('/cwork/ns458/insula-paper')
sys.path.insert(0,str(ROOT))
AUDIT=ROOT/'results/audit/figure3_consolidation_2026-11-01'


def setup():
    return nbf.v4.new_code_cell("import sys\nsys.path.insert(0,'/cwork/ns458/insula-paper')\nfrom IPython.display import SVG,display\nimport pandas as pd\nfrom src.figure3 import ROOT,MAIN_RESULTS,SAMPLING_RESULTS,HGA_RESULTS\n",id='fig3-imports')


def component_weight_cells():
    return [nbf.v4.new_markdown_cell('''## Continuous functional-component weights · PN and Lexical Delay

Generate the reviewed two-panel ternary plot from the frozen NMF: Picture Naming no-detected-Sound–Image-difference21 sites/14 participants and Lexical Delay Decision > Repeat27 sites/16 participants. Each site's three loadings are divided by their sum. Points retain the continuous Sensory-blue/Sustained-gold/Motor-red blend. Gray density clouds use the approved shared sigma0.09 and reference peak. Three solid guides connect the triangle centroid to its vertices; mean diamonds, the old dashed/50% guide lines and vertex100% text remain omitted. These are geometric sections, not maximum-weight functional classes. No refitting, new testing or selection.

The following cells execute the plotting code and export SVGs to `img/fig3/`, with numerical/provenance outputs under `results/fig3/component_weights/`. Both panels retain their reviewed layout and scales.
''',id='fig3-weight-title'),nbf.v4.new_code_cell("import sys\nsys.path.insert(0,'/cwork/ns458/insula-paper')\nfrom IPython.display import SVG,display\nfrom src.figure3 import render_component_weights\nweight_svg=render_component_weights()\ndisplay(SVG(filename=str(weight_svg)))\n",id='fig3-weight-picture-lexical'),
        nbf.v4.new_markdown_cell('''### Sound > Image only · Stimulus0–1 s

The same18 exclusively Sound > Image sites/12 participants use the frozen continuous weights, the same reference density peak and the same approved drawing. The one bidirectionally significant site remains separate. This is the original standalone component-colored plot; the later gray/blue group overlay remains in the diagnostic notebook.
''',id='fig3-weight-sound-title'),nbf.v4.new_code_cell("sound_weight_svg=render_component_weights(sound_gt_image=True)\ndisplay(SVG(filename=str(sound_weight_svg)))\n",id='fig3-weight-sound-image')]


def main_notebook():
    cells=[nbf.v4.new_markdown_cell('''# Figure 3 · Input modality and task demands

**Current main-figure entry.** All functional statistics use frozen Sensory/Sustained/Motor labels; Unassigned is excluded. Anatomical Unknown endpoints keep the approved anatomical rule. NMF, native trial/QC rules, individual tests and drawing conventions remain unchanged.

**Picture Naming, Stimulus 0–1 s:** response pool44/21 participants. Mutually exclusive Sound>Image18, no detected difference21, Image>Sound4, bidirectional1 (D0080 LPI2–3) separate. **Lexical Delay, Delay 0–1 s:** response pool71/27 participants; Decision>Repeat27, no detected difference43, Repeat>Decision1.

The central diagram region means **no detected significant condition difference**, not equivalence or a literal baseline-response intersection. Region areas are proportional to counts with the same per-electrode area across panels. Functional-color maps use the reviewed local-insula surface, surface projection, point sizes and original component-density contours.

The PN temporal portrait uses the21 no-difference sites: both condition electrode means±SEM above; (Sound+Image)/2 below without difference blanking. LD uses the27 active-pool Decision>Repeat sites: both condition means±SEM above; Decision−Repeat below with the original Decision-baseline display mask. Colors in these portraits do not imply a new per-time condition-difference test. No significance bars from different cohorts are reused. All three phases, original windows/proportional widths,1-mm rows,component hues and0–.5 maps are preserved.

Sampling counts, literal baseline-response Venns and complete component×phase paired HGA displays share one notebook: `supplementary/fig3_supplementary.ipynb`. Exported main SVG subpanels are in `img/fig3/`; selection lists, matrices and provenance are in `results/fig3/`. No composite layout is imposed.
''',id='fig3-current-title'),setup(),nbf.v4.new_code_cell('from src.figure3 import render_main\nresults=render_main()\n',id='fig3-render')]
    groups={'picture':[('first_only','Sound > Image only · n=18'),('no_detected_difference','No detected difference · n=21'),('second_only','Image > Sound only · n=4'),('bidirectional','Bidirectional · D0080 LPI2–3 · n=1')],
        'lexical':[('first_only','Decision > Repeat · n=27'),('no_detected_difference','No detected difference · n=43'),('second_only','Repeat > Decision · n=1')]}
    for key,title in [('picture','Picture Naming · Sound–Image'),('lexical','Lexical Delay · Decision–Repeat')]:
        cells += [nbf.v4.new_markdown_cell('## '+title),nbf.v4.new_code_cell(f"display(SVG(filename=results['{key}']['diagram']))\ndisplay(pd.read_csv(MAIN_RESULTS/'regions'/'{key}'/'group_counts.csv'))\n")]
        for group,label in groups[key]:
            cells += [nbf.v4.new_markdown_cell('### '+label),nbf.v4.new_code_cell(f"display(SVG(filename=results['{key}']['brains']['{group}']))\n")]
            if (key,group) in [('picture','no_detected_difference'),('lexical','first_only')]:
                cells += [nbf.v4.new_markdown_cell('### Temporal waveforms and heatmaps · same selected electrodes'),nbf.v4.new_code_cell(f"display(SVG(filename=results['{key}']['time_portrait']))\n")]
    cells.extend(component_weight_cells())
    return nbf.v4.new_notebook(cells=cells)


def _sampling_section():
    cells=[nbf.v4.new_markdown_cell('''# Supplementary Figure 3 · Sampling and baseline responsiveness

This section contains sampling information and the **original kind of baseline-response Venn**, separate from the main condition-difference classification.

Full anatomically accepted task sampling: PN162/28 participants; LD214/38. These are coverage metadata, before native QC and paired-statistics requirements. Native-good paired complete cohorts: PN128/27; LD187/35. Functional statistical cohorts exclude Unassigned: PN81/24; LD135/35. Target-phase response unions: PN44/21, Stimulus0–1 s; LD71/27, Delay0–1 s. The full inventories, assigned lists, exclusions and subject×task count lists are saved separately. The four-task anatomical coverage and task-overlap figures remain in the independent `task_sampling.ipynb` notebook.

Venn sets are each condition's original native baseline mask AND pointwise p<.05 in the stated phase. PN: Sound35, Image21, Both12 (exclusive23/9). LD: Decision61, Repeat58, Both48 (exclusive13/10). The central overlap here means both conditions baseline-responsive. A condition-only response flag is not a direct condition-difference test. All functional counts/plots exclude Unassigned; anatomical Unknown endpoint retention is unchanged.

Sampling-versus-response and baseline-class maps preserve the previously reviewed renderer. No new test, anatomical labeling or NMF fit.
'''),setup(),nbf.v4.new_code_cell("from src.figure3 import render_sampling\nsampling=render_sampling()\ndisplay(pd.read_csv(SAMPLING_RESULTS/'cohort_counts.csv'))\ndisplay(pd.read_csv(SAMPLING_RESULTS/'subject_task_counts_and_lists.csv'))\n")]
    for key,title in [('picture','Picture Naming · Stimulus0–1 s'),('lexical','Lexical Delay · Delay0–1 s')]:
        cells += [nbf.v4.new_markdown_cell('## '+title)]
        for panel,label in [('sampling_response_brain','Assigned sampled sites and baseline responders'),('0_1_venn','Baseline-response Venn'),('baseline_classes_brain','Locations of baseline-response classes')]:
            cells += [nbf.v4.new_markdown_cell('### '+label),nbf.v4.new_code_cell(f"display(SVG(filename=sampling['{key}']['{panel}']))\n")]
        cells += [nbf.v4.new_code_cell(f"display(pd.read_csv(SAMPLING_RESULTS/'{key}'/'overview'/'set_counts.csv'))\n")]
    return nbf.v4.new_notebook(cells=cells)


def _hga_section():
    cells=[nbf.v4.new_markdown_cell('''# Supplementary Figure 3 · Complete condition HGA by phase and component

Preserve the original multiphase drawing: **Stimulus/Delay/Response columns × Sensory/Sustained/Motor blocks**. Each block contains both conditions' electrode mean±SEM and their separate native-baseline-masked HGA heatmaps. These are complete condition HGA values, not condition differences. Mask-false bins are white; original shared row order across phases, component/condition colors, P95 scaling and canvas conventions are retained. Exact original function bodies now live in `src/figure3_legacy_hga.py`, without reading/executing the old main notebook.

Sound–Image uses all81 assigned paired PN sites/24 participants; Decision–Repeat uses all135 assigned paired LD sites/35 participants. These complete-cohort displays are distinct from the selected21/27 temporal portraits in the main figure. Original retained-trial means and masks are read from the audited caches; no new averaging of trials, tests, QC, smoothing or NMF fitting.

Repeat–Passive remains supplementary using its original validated cached waveforms intersected with the current assigned PN cohort. Missing observations retain their original handling. Its Response alignment is the retained nominal alignment and **does not represent an actual passive vocal onset**. Each display exports its actual membership, phase/condition/component counts and heatmap matrices.
'''),setup(),nbf.v4.new_code_cell('from src.figure3 import render_full_hga\n')]
    for key,label in [('sound_image','Sound–Image · Picture Naming'),('decision_repeat','Decision–Repeat · Lexical Delay'),('repeat_passive','Repeat–Passive · Picture Naming')]:
        cells += [nbf.v4.new_markdown_cell('## '+label),nbf.v4.new_code_cell(f"display(SVG(filename=str(render_full_hga('{key}'))))\ndisplay(pd.read_csv(HGA_RESULTS/'{key}_counts.csv'))\n")]
    return nbf.v4.new_notebook(cells=cells)


def supplementary_notebook(sampling=None,hga=None):
    """Place sampling and complete HGA together for each comparison.

    Existing executed notebooks may be supplied for a lossless reorganization;
    the default builds fresh executable cells for the canonical Run All entry.
    """
    sampling=_sampling_section() if sampling is None else sampling
    hga=_hga_section() if hga is None else hga
    def sections(nb):
        result=[]
        for cell in nb.cells[3:]:
            if cell.cell_type=='markdown' and cell.source.startswith('## '):result.append([])
            result[-1].append(cell)
        return result
    sampling_parts=sections(sampling);hga_parts=sections(hga)
    intro='# Supplementary Figure 3 · Sampling, responsiveness and complete HGA\n\n'
    intro+='Each comparison keeps its sampling/response information beside the complete component-by-phase HGA displays.\n\n'
    intro+='## Sampling and baseline-response definitions\n\n'+sampling.cells[0].source.split('\n',1)[1].lstrip()
    intro+='\n\n## Complete HGA display conventions\n\n'+hga.cells[0].source.split('\n',1)[1].lstrip()
    cells=[nbf.v4.new_markdown_cell(intro,id='fig3-si-title'),sampling.cells[1],sampling.cells[2],hga.cells[2]]
    for index in range(2):
        cells+=sampling_parts[index]
        cells+=[nbf.v4.new_markdown_cell('### Complete component × phase HGA')]+hga_parts[index][1:]
    cells+=hga_parts[2]
    return nbf.v4.new_notebook(cells=cells,metadata=dict(sampling.metadata))


def main():
    assert os.environ.get('SLURM_JOB_ID'),'Use Slurm'
    from src.figure3 import sha,write
    checkpoint=json.loads((AUDIT/'checkpoint.json').read_text())
    old=subprocess.check_output(['git','show',checkpoint['checkpoint_commit']+':vizpub/fig3.ipynb'],cwd=ROOT)
    assert __import__('hashlib').sha256(old).hexdigest()==checkpoint['old_main_notebook_sha256']
    targets={ROOT/'vizpub/fig3.ipynb':main_notebook(),ROOT/'vizpub/supplementary/fig3_supplementary.ipynb':supplementary_notebook()}
    current={path:sha(path) if path.exists() else None for path in targets}
    for path in targets:
        if path.exists():
            backup=AUDIT/'before'/path.relative_to(ROOT);backup.parent.mkdir(parents=True,exist_ok=True)
            if not backup.exists():shutil.copy2(path,backup)
    protected=[]
    for folder in ('results/nmf','img/fig1','img/fig2'):
        protected += [p for p in (ROOT/folder).rglob('*') if p.is_file()]
    protected += [ROOT/f'vizpub/fig{i}.ipynb' for i in (1,2,4,5,6)]
    protected={str(p):sha(p) for p in protected};write(AUDIT/'protected.json',protected)
    kernel=AUDIT/'jupyter/kernels/figure3-consolidation';kernel.mkdir(parents=True,exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],display_name='Figure3 main and SI',language='python')))
    os.environ['JUPYTER_PATH']=str(AUDIT/'jupyter')+os.pathsep+os.environ.get('JUPYTER_PATH','')
    for path,nb in targets.items():
        print('EXECUTING',path,flush=True)
        try:NotebookClient(nb,kernel_name='figure3-consolidation',timeout=1800,resources={'metadata':{'path':str(ROOT)}}).execute()
        except Exception:nbf.write(nb,AUDIT/(path.stem+'_failed.ipynb'));raise
        assert not [o for c in nb.cells for o in c.get('outputs',[]) if o.get('output_type')=='error' or 'image/png' in o.get('data',{})]
        assert (sha(path) if path.exists() else None)==current[path],f'Concurrent author edit: {path}'
        nbf.write(nb,path)
    assert all(sha(p)==h for p,h in protected.items())
    write(AUDIT/'execution.json',dict(passed=True,job_id=os.environ['SLURM_JOB_ID'],
        completed_utc=datetime.now(timezone.utc).isoformat(),checkpoint_commit=checkpoint['checkpoint_commit'],
        old_main_verified_committed=True,notebooks={str(p):sha(p) for p in targets},
        protected_files_unchanged=len(protected),no_new_tests_or_NMF=True))
    print('COMPLETE main Figure3 plus combined sampling and full-HGA supplementary notebook',flush=True)


if __name__=='__main__':main()
