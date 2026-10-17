#!/bin/bash
#SBATCH --job-name=grant_d0149_hg
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/grant_d0149_hg_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/grant_d0149_hg_%j.err
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
import hashlib
import nbformat
from nbclient import NotebookClient

source = Path('notebooks/grant_d0153_highgamma.ipynb')
executed = Path('results/grant_d0149_highgamma/grant_subject_highgamma.executed.ipynb')
notebook = nbformat.read(source,as_version=4)
indices = [i for i,c in enumerate(notebook.cells) if c.id.startswith('d0149-')]
assert len(indices)==8
subset = nbformat.v4.new_notebook(cells=[notebook.cells[i] for i in indices],metadata=notebook.metadata)
protected = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('img/grant_d0153_highgamma').glob('*.svg')}
assert len(protected)==4
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
def report(cell,cell_index,**kwargs):
    print(f'D0149 cell {cell_index}: {cell.id}',flush=True)
client = NotebookClient(subset,timeout=600,kernel_name='python3',
    resources={'metadata':{'path':str(Path.cwd())}},on_cell_start=report)
try:
    client.execute()
finally:
    for i,cell in zip(indices,subset.cells):
        notebook.cells[i]=cell
    nbformat.write(notebook,executed)
assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in protected.items()), 'D0153 figures changed'
assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash, 'Notebook edited during execution; results saved separately'
nbformat.write(notebook,source)
print(f'Saved {source.resolve()}; D0153 figures unchanged.',flush=True)
PY
