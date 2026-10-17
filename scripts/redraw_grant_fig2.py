"""Execute the complete Figure 2 supplement from the frozen grant NMF cache."""
from pathlib import Path
import hashlib
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'notebooks/grant_nmf_30subjects.ipynb'
OUT = ROOT/'results/grant_nmf_30subjects'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    notebook = nbformat.read(SOURCE, as_version=4)
    cells = [c for c in notebook.cells if c.id.startswith('grant30-fig2-')]
    sources = {c.id: c.source for c in cells}
    protected = [ROOT/'vizpub/fig2.ipynb']
    protected += list((ROOT/'results/nmf').glob('*.csv'))
    protected += list((ROOT/'img/fig2').glob('*.svg'))
    protected += list(OUT.glob('*.npz')) + list(OUT.glob('*.csv'))
    protected += [OUT/'cohort.json', OUT/'nmf_manifest.json']
    before = {str(p):digest(p) for p in protected}
    runbook = nbformat.v4.new_notebook(cells=cells, metadata=notebook.metadata)
    executed = OUT/'grant_fig2.executed.ipynb'
    def report(cell, cell_index, **kwargs):
        print(f'Cell {cell_index}: {cell.id}',flush=True)
    def checkpoint(**kwargs):
        nbformat.write(runbook, executed)
    client = NotebookClient(runbook, timeout=7200, kernel_name='python3',
        resources={'metadata':{'path':str(ROOT)}},on_cell_start=report,
        on_cell_complete=checkpoint)
    try:
        client.execute()
    finally:
        nbformat.write(runbook, executed)
    assert all(digest(Path(p))==sha for p,sha in before.items()), 'Protected source changed'
    current = nbformat.read(SOURCE,as_version=4)
    updates = {c.id:c for c in runbook.cells}
    for i,c in enumerate(current.cells):
        if c.id in updates:
            assert c.source==sources[c.id], 'Supplement source changed; executed copy retained'
            current.cells[i]=updates[c.id]
    nbformat.write(current,SOURCE)
    nbformat.write(current,OUT/'grant_nmf_30subjects.executed.ipynb')
    print('Complete: all Figure 2 panels; protected source and frozen fit unchanged.',flush=True)

if __name__=='__main__':main()
