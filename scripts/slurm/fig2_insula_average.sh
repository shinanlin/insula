#!/bin/bash
#SBATCH --job-name=fig2_traces
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/fig2_traces_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/fig2_traces_%j.err
#SBATCH --time=00:40:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional/vizpub

set -eo pipefail

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

export PYTHONPATH="/hpc/group/coganlab/nanlinshi/insula-functional${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1

python - <<'PY'
import json
from pathlib import Path

nb = json.loads(Path("fig2.ipynb").read_text())
cells = {c.get("id"): c for c in nb["cells"]}
ns = {"__name__": "__main__"}
for cid in (
    "fig2-imports",
    "fig2-style",
    "fig2-load",
    "fig2-avg-plot",
    "fig2-hga-plot",
    "6c3ec19c",  # hard-label HGA heatmap
):
    print(f"=== exec {cid} ===", flush=True)
    exec("".join(cells[cid]["source"]), ns)
img = ns["IMG_DIR"]
for name in (
    "fig2_insula_average_traces.svg",
    "fig2_hga_traces.svg",
    "fig2_hga_heatmap.svg",
):
    print("wrote", img / name, flush=True)
PY
