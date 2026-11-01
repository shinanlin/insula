"""Execute the PN baseline panels and their approved SI extensions."""
from pathlib import Path
from datetime import datetime,timezone
import json
import os
import sys
import xml.etree.ElementTree as ET

ROOT=Path('/cwork/ns458/insula-paper')
REVIEW=ROOT/'results/audit/picture_condition_sets_two_steps_2026-10-27'


def build():
    import nbformat as nbf
    md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
    cells=[md('''# Picture Naming: Stimulus 0–1 s baseline responses

**128 eligible paired insula bipolar sites / 27 participants.** The task-specific anatomical sample contains 162 sites / 28 participants before 34 existing native bad-channel/endpoint QC exclusions. All eligible paired sites form the denominator, without a prior responsiveness or NMF-membership gate.

Repeat sound and Repeat image trials retain their original native QC and the previously approved exact-Stimulus repairs. Response membership is any hit in the existing native baseline-significance mask within inclusive Stimulus [0,1] s (129 samples at 128 Hz). The masks and their original correction are reused unchanged.

The first figure shows sampling and the **Sound OR Image** response union. The second figure divides that same union into Sound-only, Image-only and Both, using matching colors in the Venn and bilateral brain map. Sound-only and Image-only describe detection relative to baseline, without establishing a direct preference between conditions. Both denotes detection in both conditions.

The accepted Figure 1 local-insula renderer is reused with the original cameras, 24-pixel points, camera-facing surface snap and 4-mm display lift. Analytic coordinates remain unchanged. Subject sets overlap and participant counts cannot be added.
'''),code('''from pathlib import Path
import sys,json
import pandas as pd
from IPython.display import SVG,display
ROOT=Path('/cwork/ns458/insula-paper')
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.supplementary import picture_condition_figures as plots
from scripts.refresh_picture_condition_sets import protected
table,counts=plots.load()
responsive=table.sound_responsive|table.image_responsive
print('Sampling:',len(table),'sites /',table.subject.nunique(),'participants')
print('Stimulus responsive in Sound or Image:',int(responsive.sum()),'sites /',table.loc[responsive].subject.nunique(),'participants')
print('Protected files verified:',protected())
'''),md('''## 1. Sampled sites and stimulus-responsive sites

All 128 eligible sampled sites are shown. Purple marks the 49 sites with a detected Stimulus response in Sound or Image; gray marks the 79 sites with neither condition detecting a response. Each site appears once. Hemisphere counts refer to all sampled sites in this first figure.
'''),code('''brain_paths=plots.brains(table)
display(SVG(filename=str(brain_paths['sampling_response'])))
columns=['subject','channel','electrode','sound_responsive','image_responsive','baseline_class','x','y','z']
site_list=table[columns].copy()
site_list['stimulus_responsive']=responsive
site_list.to_csv(plots.OUT/'sampling_response_electrodes.csv',index=False)
'''),md('''## 2. Sound-only, Image-only and Both: Venn and matching anatomy

Sound-only: **25 sites / 17 participants**. Image-only: **11 sites / 8 participants**. Both: **13 sites / 10 participants**. Thus Sound totals 38 sites, Image totals 24, and the union is 49 sites / 22 participants. The other 79 eligible sites are counted outside the circles.

The bilateral map displays exactly these same 49 responsive sites: Sound-only blue, Image-only orange and Both purple. Its hemisphere counts refer to the responsive union. The point size, camera and surface convention match the first figure.
'''),code('''venn_path=plots.venn(table)
condition_pair,sheet=plots.assemble(venn_path,brain_paths)
display(counts.drop(columns='participants'))
display(SVG(filename=str(condition_pair)))
site_list.loc[responsive].to_csv(plots.OUT/'condition_response_classes.csv',index=False)
specification=dict(notebook=str(ROOT/'vizpub/supplementary/fig3_picture_stimulus_condition_sets.ipynb'),
    task='PictureNaming',phase='Stimulus',window=[0,1],n_sites=128,n_participants=27,
    steps=[dict(step=1,meaning='Sampling and any Sound OR Image native baseline response',sampled=128,responsive=49,neither=79,image=str(brain_paths['sampling_response'])),
           dict(step=2,meaning='Baseline response sets and matching anatomy',sound_only=25,image_only=11,both=13,image=str(condition_pair))],
    sheet=str(sheet),condition_colors=plots.COLORS,panels={k:str(v) for k,v in brain_paths.items()},
    masks_and_cohort_unchanged=True,new_tests_run=False)
(plots.OUT/'specification.json').write_text(json.dumps(specification,indent=2)+'\\n')
print('Saved:',sheet)
print('Protected files verified:',protected())
''')]
    from src.supplementary.picture_functional_set_contours import notebook_cells
    cells.extend(notebook_cells())
    from src.supplementary.picture_fixed_template_expression import notebook_cells as expression_cells
    cells.extend(expression_cells())
    from src.supplementary.picture_expression_zoom import notebook_cells as zoom_cells
    cells.extend(zoom_cells())
    from src.supplementary.picture_region_nmf_weights import notebook_cells as composition_cells
    cells.extend(composition_cells())
    from src.supplementary.picture_region_point_panels import notebook_cells as point_cells
    cells.extend(point_cells())
    from src.supplementary.picture_combined_region_contours import notebook_cells as combined_cells
    cells.extend(combined_cells())
    from src.supplementary.picture_top_weight_gaussian import notebook_cells as top_weight_cells
    cells.extend(top_weight_cells())
    return nbf.v4.new_notebook(cells=cells)


def main():
    assert os.environ.get('SLURM_JOB_ID'),'Use Slurm'
    import nbformat
    from nbclient import NotebookClient
    from scripts.refresh_picture_condition_sets import OUT,IMG,NB,protected,sha,write
    protected()
    REVIEW.mkdir(parents=True,exist_ok=True)
    archived_nb=REVIEW/'before'/NB.relative_to(ROOT)
    before_sha=sha(NB)
    if before_sha!=sha(archived_nb):
        current=json.loads((OUT/'current_display.json').read_text())
        approved=json.loads(Path(current['completion']).read_text())
        assert approved['passed']
        assert before_sha==approved.get('notebook_after_sha256',approved.get('notebook_sha256')),'Notebook changed after its recorded author-approved update'
    mutable={'specification.json','brain_display_provenance.json','current_display.json',
             'sampling_response_electrodes.csv','condition_response_classes.csv'}
    protected_path=REVIEW/'protected_scientific_results.json'
    if not protected_path.exists():
        files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in mutable and not p.name.endswith('_display_coordinates.csv')]
        write(protected_path,{str(p):sha(p) for p in files})
    frozen=json.loads(protected_path.read_text())
    assert all(sha(p)==h for p,h in frozen.items())
    nb=build()
    kernel=REVIEW/'jupyter/kernels/picture-two-steps'
    kernel.mkdir(parents=True,exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],display_name='Picture baseline responses',language='python')))
    os.environ['JUPYTER_PATH']=str(REVIEW/'jupyter')+os.pathsep+os.environ.get('JUPYTER_PATH','')
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    def start(cell,cell_index,**kw):print(datetime.now(timezone.utc).isoformat(),'cell',cell_index,flush=True)
    try:
        NotebookClient(nb,kernel_name='picture-two-steps',timeout=1800,iopub_timeout=180,
            resources={'metadata':{'path':str(NB.parent)}},on_cell_start=start).execute()
    except Exception:
        nbformat.write(nb,REVIEW/'failed.ipynb');raise
    assert all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')
    assert not [o for c in nb.cells for o in c.get('outputs',[]) if o.get('output_type')=='error' or 'image/png' in o.get('data',{})]
    assert sha(NB)==before_sha,'Notebook changed during rendering'
    baseline_expected={'fig3_picture_stimulus_sampling_response_brain.svg','fig3_picture_stimulus_baseline_classes_brain.svg',
              'fig3_picture_stimulus_0_1_venn.svg','fig3_picture_stimulus_0_1_condition_sets.svg','fig3_picture_stimulus_0_1_summary.svg'}
    from src.supplementary.picture_functional_set_contours import EXPORT_NAMES,SHEET
    from src.supplementary.picture_fixed_template_expression import EXPORT_NAMES as EXPRESSION_EXPORTS
    from src.supplementary.picture_expression_zoom import EXPORT_NAME as ZOOM_EXPORT
    from src.supplementary.picture_region_nmf_weights import EXPORTS as COMPOSITION_EXPORTS
    from src.supplementary.picture_region_point_panels import EXPORTS as POINT_EXPORTS
    from src.supplementary.picture_combined_region_contours import EXPORTS as COMBINED_EXPORTS
    from src.supplementary.picture_top_weight_gaussian import EXPORTS as TOP_WEIGHT_EXPORTS
    expected=baseline_expected|set(EXPORT_NAMES.values())|{SHEET}|set(EXPRESSION_EXPORTS.values())|{ZOOM_EXPORT}|set(COMPOSITION_EXPORTS.values())|set(POINT_EXPORTS.values())|set(COMBINED_EXPORTS.values())|set(TOP_WEIGHT_EXPORTS.values())
    exports={}
    for name in sorted(expected):
        p=IMG/name
        xml=ET.fromstring(p.read_text())
        ids=[e.attrib['id'] for e in xml.iter() if 'id' in e.attrib]
        assert len(ids)==len(set(ids)),p
        text=' '.join(e.text or '' for e in xml.iter() if e.tag.endswith('}text'))
        assert ' > ' not in text,p
        if name in baseline_expected:
            assert not any(term in text for term in ['Sensory','Sustained','Motor','Unassigned']),p
        exports[str(p)]=sha(p)
    specification=json.loads((OUT/'specification.json').read_text())
    assert len(specification['steps'])==2
    assert all(sha(p)==h for p,h in frozen.items()),'Saved scientific results changed'
    protected_count=protected()
    # The superseded directional maps remain recoverable from the archive.
    retired=[]
    for p in IMG.glob('*.svg'):
        if p.name in expected:continue
        previous=REVIEW/'before'/p.relative_to(ROOT)
        assert previous.is_file() and sha(previous)==sha(p),'Unrecognized or changed prior export: '+str(p)
        p.unlink();retired.append(str(p))
    nbformat.write(nb,NB)
    steps=specification['steps']+[dict(step=3,meaning='Frozen functional colors and spatial-density contours for Picture-only, Sound-only and Both',image=str(IMG/SHEET))]
    steps.append(dict(step=4,meaning='Paired PN Sound/Image expression of frozen temporal NMF templates',
                      images={metric:str(IMG/name) for metric,name in EXPRESSION_EXPORTS.items()},
                      enlarged_display=str(IMG/ZOOM_EXPORT)))
    steps.append(dict(step=5,meaning='Three response regions using original frozen NMF weights, Unassigned sites explicit',
                      images={k:str(IMG/name) for k,name in COMPOSITION_EXPORTS.items()}))
    steps.append(dict(step=6,meaning='Independent neutral Venn, saved-weight bars and pure-color functional point maps',
                      images={k:str(IMG/name) for k,name in POINT_EXPORTS.items()},no_composite=True))
    steps.append(dict(step=7,meaning='Absolute counts and one bilateral map with functional colors, response-set symbols and unfilled contours',
                      images={k:str(IMG/name) for k,name in COMBINED_EXPORTS.items()},no_composite=True))
    steps.append(dict(step=8,meaning='Three independent bilateral Gaussian maps of the top original loading per electrode',
                      images={k:str(IMG/name) for k,name in TOP_WEIGHT_EXPORTS.items()},no_composite=True))
    report=dict(passed=True,notebook=str(NB),notebook_sha256=sha(NB),exports=exports,
        steps=steps,all_notebook_cells_executed=True,
        protected_files_unchanged=protected_count,scientific_results_unchanged=len(frozen),
        prior_outputs_archived=str(REVIEW/'before'),retired_active_exports=retired,
        job_id=os.environ['SLURM_JOB_ID'],completed_utc=datetime.now(timezone.utc).isoformat())
    write(REVIEW/'completion.json',report)
    write(OUT/'current_display.json',dict(completion=str(REVIEW/'completion.json'),notebook=str(NB),steps=8,scope='Three independent bilateral Gaussian maps using top original NMF loading only'))
    print('COMPLETE',NB,flush=True)


if __name__=='__main__':main()
