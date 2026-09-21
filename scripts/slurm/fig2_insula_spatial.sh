#!/bin/bash
#SBATCH --job-name=fig2_spatial
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/fig2_spatial_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/fig2_spatial_%j.err
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
export PYVISTA_OFF_SCREEN=true
export MESA_GL_VERSION_OVERRIDE=3.3
export PYVISTA_USE_PANEL=false
export MNE_3D_BACKEND=notebook
export OMP_NUM_THREADS=1

python - <<'PY'
import json
from pathlib import Path

import pandas as pd

nb = json.loads(Path("fig2.ipynb").read_text())
cells = {c.get("id"): c for c in nb["cells"]}
ns = {"__name__": "__main__"}
for cid in ("fig2-imports", "fig2-style"):
    print(f"=== exec {cid} ===", flush=True)
    exec("".join(cells[cid]["source"]), ns)
print("=== load assignments ===", flush=True)
ns["assignments"] = pd.read_csv(ns["NMF_ASSIGNMENTS"])
print("=== exec fig2-spatial-plot ===", flush=True)
exec("".join(cells["fig2-spatial-plot"]["source"]), ns)
print("done", flush=True)
PY
