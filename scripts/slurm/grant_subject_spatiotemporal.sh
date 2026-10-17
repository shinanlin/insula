#!/bin/bash
#SBATCH --job-name=grant_subject_space
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/grant_subject_space_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/grant_subject_space_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=2
#SBATCH --partition=common,scavenger
#SBATCH --account=coganlab
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
export PYVISTA_OFF_SCREEN=true LIBGL_ALWAYS_SOFTWARE=1
export VTK_DEFAULT_OPENGL_WINDOW=vtkXOpenGLRenderWindow
xvfb-run -a -s '-screen 0 1600x1200x24' python -u - <<'PY'
from pathlib import Path
import hashlib
import os
import nbformat
from nbclient import NotebookClient
source=Path('notebooks/grant_d0153_highgamma.ipynb')
executed=Path('results/grant_subject_spatiotemporal/grant_subject_spatiotemporal.executed.ipynb')
notebook=nbformat.read(source,as_version=4)
source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
indices=[i for i,c in enumerate(notebook.cells) if c.id.startswith('spatial-')] if os.getenv('GRANT_SPATIAL_ONLY')=='1' else list(range(len(notebook.cells)))
runbook=nbformat.v4.new_notebook(cells=[notebook.cells[i] for i in indices],metadata=notebook.metadata)
def report(cell,cell_index,**kwargs):
    print(f'Cell {cell_index}: {cell.id}',flush=True)
client=NotebookClient(runbook,timeout=900,kernel_name='python3',resources={'metadata':{'path':str(Path.cwd())}},on_cell_start=report)
try:
    client.execute()
finally:
    for i,cell in zip(indices,runbook.cells):
        notebook.cells[i]=cell
    nbformat.write(notebook,executed)
assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash,'Source edited during run; executed copy saved separately'
nbformat.write(notebook,source)
print('Completed:',source.resolve(),flush=True)
PY
