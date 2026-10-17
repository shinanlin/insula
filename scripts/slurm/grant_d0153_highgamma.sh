#!/bin/bash
#SBATCH --job-name=grant_d0153_hg
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/grant_d0153_hg_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/grant_d0153_hg_%j.err
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
python -u - <<'PY'
from pathlib import Path
import nbformat
from nbclient import NotebookClient
source = Path('notebooks/grant_d0153_highgamma.ipynb')
executed = Path('results/grant_d0153_highgamma/grant_d0153_highgamma.executed.ipynb')
executed.parent.mkdir(parents=True, exist_ok=True)
notebook = nbformat.read(source, as_version=4)
def report(cell, cell_index, **kwargs):
    print(f'Cell {cell_index}: {cell.id}', flush=True)
client = NotebookClient(notebook, timeout=600, kernel_name='python3',
                        resources={'metadata': {'path': str(Path.cwd())}}, on_cell_start=report)
try:
    client.execute()
finally:
    nbformat.write(notebook, executed)
# Keep the requested notebook immediately viewable with its tables and SVG output.
nbformat.write(notebook, source)
print(f'Executed and saved: {source.resolve()}', flush=True)
PY
