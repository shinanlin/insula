#!/bin/bash
#SBATCH --job-name=grant_d0153_ria56
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/grant_d0153_ria56_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/grant_d0153_ria56_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --partition=common,scavenger
#SBATCH --account=coganlab
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
export PYVISTA_OFF_SCREEN=true LIBGL_ALWAYS_SOFTWARE=1 VTK_DEFAULT_OPENGL_WINDOW=vtkXOpenGLRenderWindow
xvfb-run -a -s '-screen 0 1600x1200x24' python -u - <<'PY'
from pathlib import Path
import nbformat
from nbclient import NotebookClient
root=Path.cwd();source=root/'notebooks/grant_d0153_highgamma.ipynb'
out=root/'results/grant_d0153_ria56';out.mkdir(exist_ok=True)
nb=nbformat.read(source,as_version=4)
cells=[c for c in nb.cells if c.id.startswith('ria56-')]
originals={c.id:c.source for c in cells}
run=nbformat.v4.new_notebook(cells=cells,metadata=nb.metadata)
client=NotebookClient(run,timeout=900,kernel_name='python3',resources={'metadata':{'path':str(root)}},
    on_cell_start=lambda cell,cell_index,**kw:print(f'Cell {cell_index}: {cell.id}',flush=True))
try:client.execute()
finally:nbformat.write(run,out/'grant_d0153_ria56.executed.ipynb')
current=nbformat.read(source,as_version=4);updates={c.id:c for c in run.cells}
for i,c in enumerate(current.cells):
    if c.id in updates:
        assert c.source==originals[c.id],'RIA5-6 HGA source changed; executed copy retained'
        current.cells[i]=updates[c.id]
nbformat.write(current,source)
print('RIA5-6 HGA figures complete; existing HGA cells preserved.',flush=True)
PY
