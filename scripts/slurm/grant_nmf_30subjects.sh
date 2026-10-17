#!/bin/bash
#SBATCH --job-name=grant_nmf30
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/grant_nmf30_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/grant_nmf30_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
export PYTHONPATH="$PWD${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_PANEL=false
export MNE_3D_BACKEND=notebook
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export LIBGL_ALWAYS_SOFTWARE=1
export VTK_DEFAULT_OPENGL_WINDOW=vtkXOpenGLRenderWindow

mkdir -p logs/slurm results/grant_nmf_30subjects img/grant_nmf_30subjects

xvfb-run -a -s '-screen 0 1600x1200x24' python -u - <<'PY'
from pathlib import Path
import hashlib
import os
import shutil
import nbformat
from nbclient import NotebookClient

source = Path('notebooks/grant_nmf_30subjects.ipynb')
executed = Path('results/grant_nmf_30subjects/grant_nmf_30subjects.executed.ipynb')
if os.getenv('GRANT_ARCHIVE_BEFORE_REFIT') == '1':
    archive = Path('results/grant_nmf_30subjects_before_response_refit')
    marker = archive/'archive_complete.txt'
    if not marker.exists():
        archive.mkdir(exist_ok=True)
        for item in Path('results/grant_nmf_30subjects').iterdir():
            target = archive/item.name
            if item.is_dir(): shutil.copytree(item, target, dirs_exist_ok=True)
            else: shutil.copy2(item, target)
        shutil.copytree('img/grant_nmf_30subjects',
            'img/grant_nmf_30subjects_before_response_refit', dirs_exist_ok=True)
        marker.write_text('Archived original response (0, 0.5) results before refit.\n')
    for name in ('temporal_redraw_validation.json','temporal_layouts.json'):
        Path('results/grant_nmf_30subjects', name).unlink(missing_ok=True)
    print('Previous fit archived:', archive.resolve(), flush=True)
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
notebook = nbformat.read(source, as_version=4)

def report_start(cell, cell_index, **kwargs):
    print(f"Cell {cell_index}: {cell.get('id', '')}", flush=True)

def checkpoint(**kwargs):
    nbformat.write(notebook, executed)

client = NotebookClient(
    notebook, timeout=3600, kernel_name='python3',
    resources={'metadata': {'path': str(Path.cwd())}},
    on_cell_start=report_start, on_cell_complete=checkpoint,
)
try:
    client.execute()
finally:
    nbformat.write(notebook, executed)
assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash, 'Notebook changed during run; executed copy retained'
nbformat.write(notebook, source)
print(f'Executed notebook: {executed.resolve()}', flush=True)
print('Validation: results/grant_nmf_30subjects/validation.json', flush=True)
PY
