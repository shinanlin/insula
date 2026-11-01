"""Execute the independent Figure 3 denominator notebook on Slurm."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path('/cwork/ns458/insula-paper')
OUT = ROOT/'results/supplementary/fig3_response_characterization_2026-10-23'
AUDIT = ROOT/'results/audit/fig3_response_characterization_2026-10-23'
NB = ROOT/'vizpub/supplementary/fig3_response_characterization.ipynb'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import nbformat
    from nbclient import NotebookClient
    import cairosvg
    assert os.environ.get('SLURM_JOB_ID'), 'Slurm only'
    AUDIT.mkdir(parents=True, exist_ok=True)
    protected_paths = list((ROOT/'vizpub').glob('fig[1-6].ipynb'))
    protected_paths += [p for p in (ROOT/'results/nmf').glob('*') if p.is_file()]
    for folder in ('img/fig1','img/fig2','img/fig3','img/fig4','img/fig5','img/fig6',
                   'img/supplementary/sound_image_test','img/supplementary/decision_repeat_test'):
        protected_paths += list((ROOT/folder).glob('*.svg'))
    for topic in ('sound_image_test','decision_repeat_test'):
        protected_paths.append(ROOT/f'vizpub/supplementary/{topic}.ipynb')
        base = ROOT/f'results/supplementary/{topic}'
        protected_paths += list(base.glob('*.csv'))+list(base.glob('*.npz'))
    for folder in ('results/supplementary/sound_image_test/common_stimulus_response',
                   'results/supplementary/decision_repeat_test/delay_overlap_maps',
                   'results/supplementary/decision_repeat_test/delay_electrode_selection'):
        protected_paths += list((ROOT/folder).glob('*.csv'))+list((ROOT/folder).glob('*.json'))
    protected_paths += [ROOT/'results/supplementary/sound_image_test/event_repair_2026-10-21/waveforms.pkl',
                        ROOT/'results/supplementary/context_comparisons/waveforms.pkl']
    protected = {str(p): sha(p) for p in set(protected_paths)}
    (AUDIT/'protected.json').write_text(json.dumps(protected, indent=2)+'\n')
    kernels = AUDIT/'jupyter/kernels/insula-response-characterization'
    kernels.mkdir(parents=True, exist_ok=True)
    (kernels/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        display_name='Insula response characterization',language='python')))
    os.environ['PATH'] = str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    os.environ['JUPYTER_PATH'] = str(AUDIT/'jupyter')+os.pathsep+os.environ.get('JUPYTER_PATH','')
    nb = nbformat.read(NB,as_version=4)
    def started(cell,cell_index,**kwargs):
        print(datetime.now(timezone.utc).isoformat(),'cell',cell_index,cell.id,flush=True)
    try:
        NotebookClient(nb,kernel_name='insula-response-characterization',timeout=1800,
            iopub_timeout=180,raise_on_iopub_timeout=True,
            resources={'metadata':{'path':str(NB.parent)}},on_cell_start=started).execute()
    except Exception:
        nbformat.write(nb,AUDIT/'failed.ipynb')
        raise
    assert not [o for c in nb.cells for o in c.get('outputs',[]) if o.get('output_type')=='error' or 'image/png' in o.get('data',{})]
    assert all(sha(p)==h for p,h in protected.items()),'An existing numerical result or figure changed'
    nbformat.write(nb,NB)
    preview = Path('/hpc/home/ns458/.codex/visualizations/2026/10/17/01a14b02-953c-7931-bc4a-2605235435f8')
    previews=[]
    for key in ('sound_image','decision_repeat'):
        svg=ROOT/f'img/supplementary/fig3_response_characterization/fig3_{key}_full_response_characterization.svg'
        png=preview/f'fig3_{key}_full_response_characterization_20261023.png'
        cairosvg.svg2png(url=str(svg),write_to=str(png),output_width=1800)
        previews.append(str(png))
    report=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),job_id=os.environ['SLURM_JOB_ID'],
        notebook=str(NB),results=str(OUT),previews=previews,protected_files=len(protected),
        all_existing_cohorts_masks_models_waveforms_and_figures_unchanged=True,
        new_inference=False,native_masks_verified=True)
    (AUDIT/'execution_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
