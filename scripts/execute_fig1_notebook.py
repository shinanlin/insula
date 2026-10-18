#!/usr/bin/env python3
"""Execute the Figure 1 notebook in its worktree and verify saved outputs."""
from datetime import datetime, timezone
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import nbformat
from nbclient import NotebookClient

ROOT=Path('/cwork/ns458/insula-paper')
SOURCE=Path('/hpc/group/coganlab/nanlinshi/insula-functional')
OUT=ROOT/'results/fig1'
NOTEBOOK=ROOT/'vizpub/fig1.ipynb'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    original_legacy=sha(SOURCE/'vizpub/fig1.ipynb')
    nb=nbformat.read(NOTEBOOK,as_version=4)
    for i,cell in enumerate(nb.cells):
        if cell.cell_type=='code':ast.parse(cell.source,filename=f'fig1-cell-{i}')
    nbformat.validate(nb)
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    # A local kernelspec pins execution to the existing ieeg environment.
    jupyter=OUT/'jupyter'
    spec=jupyter/'kernels/insula-paper-fig1'
    spec.mkdir(parents=True,exist_ok=True)
    (spec/'kernel.json').write_text(json.dumps(dict(
        argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        display_name='Insula Figure 1 (ieeg)',language='python')))
    os.environ['JUPYTER_PATH']=str(jupyter)+os.pathsep+os.environ.get('JUPYTER_PATH','')
    def started(cell,cell_index,**kwargs):
        print(f'{datetime.now(timezone.utc).isoformat()} cell {cell_index}: '
              f'{cell.source.splitlines()[0][:100] if cell.source else "empty"}',flush=True)
    client=NotebookClient(nb,kernel_name='insula-paper-fig1',timeout=1800,iopub_timeout=180,
        raise_on_iopub_timeout=True,resources={'metadata':{'path':str(ROOT/'vizpub')}},
        on_cell_start=started,record_timing=True)
    try:
        client.execute()
    except Exception:
        nbformat.write(nb,OUT/'fig1_failed.ipynb')
        raise
    assert sha(SOURCE/'vizpub/fig1.ipynb')==original_legacy
    errors=[(i,o.get('ename'),o.get('evalue')) for i,c in enumerate(nb.cells) for o in c.get('outputs',[])
            if o.get('output_type')=='error']
    assert not errors, errors
    assert not [i for i,c in enumerate(nb.cells) for o in c.get('outputs',[]) if 'image/png' in o.get('data',{})]
    assert not list((ROOT/'img/fig1').glob('*.png'))
    manifest=json.loads((OUT/'figure_manifest.json').read_text())
    assert manifest['sampled_sites']==323 and manifest['confirmed_responsive_sites']==227
    assert manifest['nmf_files_unchanged'] and not manifest['nmf_used_for_selection']
    assert manifest['tfr_channel']=='D0064_LAI3-4'
    assert manifest['excluded_subjects']==['D0121']
    assert len(manifest['figures'])>=12
    nb.metadata['insula_figure1']=dict(
        cohort_version='2026-10-17',sampled_sites=323,confirmed_responsive_sites=227,
        figures_svg_only=True,significance_finalized=False,
        executed_utc=datetime.now(timezone.utc).isoformat())
    temporary=NOTEBOOK.with_suffix('.executed.tmp.ipynb')
    nbformat.write(nb,temporary)
    temporary.replace(NOTEBOOK)
    evidence=ROOT/'docs/evidence/figure1_refresh_2026-10-17'
    evidence.mkdir(parents=True,exist_ok=True)
    for name in ['figure_manifest.json','task_summary.csv','insula_laterality_counts.csv',
                 'phase_roi_counts.csv','roi_sampled_and_confirmed_counts.csv',
                 'average_hga_counts.csv','average_hga_missing_waveforms.csv',
                 'average_hga_site_membership.csv','average_hga_supplemented_epochs.json',
                 'average_hga_source_membership.csv',
                 'average_hga_time_coverage.csv',
                 'excluded_noise_subject_sites.csv',
                 'D0064_native_schematic_sites.csv','insula_plot_coordinates.csv']:
        shutil.copy2(OUT/name,evidence/name)
    figure_archive=evidence/'figures'
    figure_archive.mkdir(exist_ok=True)
    for name in manifest['figures']:
        shutil.copy2(ROOT/'img/fig1'/name,figure_archive/name)
    validation=dict(passed=True,code_cells=sum(c.cell_type=='code' for c in nb.cells),
        all_code_cells_executed=all(c.execution_count is not None for c in nb.cells if c.cell_type=='code'),
        output_errors=errors,legacy_notebook_unchanged=True,
        sampled_sites=323,confirmed_responsive_sites=227,svg_panels=len(manifest['figures']),
        tfr_example_in_reviewed_cohort=True,svg_inline_only=True,
        average_hga_counts=manifest['group_average_counts'],
        average_hga_missing_site_phase_pairs=manifest['group_average_missing_site_phase_pairs'],
        nmf_unchanged=True,notebook_sha256=sha(NOTEBOOK))
    (OUT/'execution_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    shutil.copy2(OUT/'execution_validation.json',evidence/'execution_validation.json')
    (OUT/'fig1_failed.ipynb').unlink(missing_ok=True)
    print(json.dumps(validation,indent=2),flush=True)


if __name__=='__main__':main()
