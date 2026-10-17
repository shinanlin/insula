#!/bin/bash
# Execute the standalone anatomy notebook; outputs remain SVG only.
#SBATCH --job-name=shi100_insula
#SBATCH --partition=common,scavenger
#SBATCH --account=coganlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/shi100_insula_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/shi100_insula_%j.err
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
export PYVISTA_OFF_SCREEN=true
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_CACHE_DIR=/work/ns458/numba_cache

python -u - <<'PY'
from pathlib import Path
import nbformat
from nbclient import NotebookClient

path = Path('notebooks/shi100_insula_contacts.ipynb')
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(notebook, timeout=1200, kernel_name='python3',
                        resources={'metadata': {'path': str(Path.cwd())}})
print(f'Executing {path}', flush=True)
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
    for cell in notebook.cells:
        for output in cell.get('outputs', []):
            if output.output_type == 'stream':
                print(output.text, end='', flush=True)
            elif output.output_type == 'error':
                print('\n'.join(output.traceback), flush=True)
print('Notebook completed.', flush=True)
PY
