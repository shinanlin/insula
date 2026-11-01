"""Build and execute the separate Figure 3 draft notebook on Slurm."""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/fig3/draft_context_2026-10-20"
IMG = ROOT / "img/fig3/drafts/fig3_context_three_comparisons.svg"
NOTEBOOK = ROOT / "vizpub/drafts/fig3_context_draft.ipynb"
OUT.mkdir(parents=True, exist_ok=True)
NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)

nb = nbformat.v4.new_notebook(cells=[
    nbformat.v4.new_markdown_cell("""# Figure 3 — actual-data layout draft

Author-approved draft: three within-task comparisons (input modality,
speech requirement, response rule), each with sensory/sustained/motor curves
and paired heatmaps. Current three-phase sound-only NMF labels are frozen.

Native validated HGA and saved baseline masks are read directly. Curves are
electrode mean ± SEM, never significance-filtered within a condition. The
original task sig-union gate remains on the two context comparisons. Heatmaps
blank mask-false bins; the two conditions share electrode order. There are
no new between-condition tests or significance stars. This separate notebook
does not replace the main Figure 3 notebook or its published images.
"""),
    nbformat.v4.new_code_cell("""from pathlib import Path
import sys, json
ROOT = Path('/cwork/ns458/insula-paper')
sys.path.insert(0, str(ROOT))
from src.figure3_context_draft import load_data, plot_draft, protected_snapshot, write_json
from IPython.display import display
%matplotlib inline
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('svg')
OUT = ROOT / 'results/fig3/draft_context_2026-10-20'
IMG = ROOT / 'img/fig3/drafts/fig3_context_three_comparisons.svg'
before = protected_snapshot()
write_json(OUT / 'protected_before.json', before)
"""),
    nbformat.v4.new_code_cell("data = load_data(OUT)\nprint('Waveform rows:', len(data))\ndisplay(data.groupby(['row', 'functional_cluster', 'level']).electrode.nunique().unstack())"),
    nbformat.v4.new_markdown_cell("## Three task comparisons × three populations\n\nHeatmap color scales retain the source per-population P95 convention. Blank samples indicate a nonsignificant saved baseline mask, not missing HGA or a between-condition test."),
    nbformat.v4.new_code_cell("fig, counts = plot_draft(data, IMG, OUT)\ndisplay(fig)\ndisplay(counts)"),
    nbformat.v4.new_code_cell("after = protected_snapshot()\nassert before == after, 'Protected Figure 1/2 or NMF input changed during draft'\nwrite_json(OUT / 'protected_after.json', after)\nprint('Protected inputs unchanged:', len(after))"),
])
nb.metadata["kernelspec"] = dict(display_name="Insula Figure 3 draft", language="python", name="insula-paper-fig3-draft")
jupyter = OUT / "jupyter"
kernel = jupyter / "kernels/insula-paper-fig3-draft"
kernel.mkdir(parents=True, exist_ok=True)
(kernel / "kernel.json").write_text(json.dumps(dict(argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"], display_name="Insula Figure 3 draft", language="python")))
os.environ["JUPYTER_PATH"] = str(jupyter) + os.pathsep + os.environ.get("JUPYTER_PATH", "")
nbformat.write(nb, NOTEBOOK)

def started(cell, cell_index, **kwargs):
    print(datetime.now(timezone.utc).isoformat(), "cell", cell_index, cell.source.splitlines()[0], flush=True)

client = NotebookClient(nb, kernel_name="insula-paper-fig3-draft", timeout=7200,
                        iopub_timeout=180, record_timing=True,
                        resources={"metadata": {"path": str(ROOT)}}, on_cell_start=started)
try:
    client.execute()
except Exception:
    nbformat.write(nb, OUT / "failed.ipynb")
    raise
assert not [o for c in nb.cells for o in c.get("outputs", []) if o.get("output_type") == "error"]
assert IMG.is_file()
nbformat.write(nb, NOTEBOOK)
receipt = dict(completed_utc=datetime.now(timezone.utc).isoformat(),
               notebook=str(NOTEBOOK), image=str(IMG), protected_unchanged=True,
               passed=True, job_id=os.environ.get("SLURM_JOB_ID"))
(OUT / "execution_validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2), flush=True)
