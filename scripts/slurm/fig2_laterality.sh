#!/bin/bash
#SBATCH --job-name=fig2_laterality
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/fig2_laterality_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/fig2_laterality_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
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

import pandas as pd

nb = json.loads(Path("fig2.ipynb").read_text())
cells = {c.get("id"): c for c in nb["cells"]}
ns = {"__name__": "__main__"}
for cid in ("fig2-imports", "fig2-style"):
    print(f"=== exec {cid} ===", flush=True)
    exec("".join(cells[cid]["source"]), ns)
print("=== load assignments ===", flush=True)
ns["assignments"] = pd.read_csv(ns["NMF_ASSIGNMENTS"])
print("=== exec fig2-laterality-plot ===", flush=True)
exec("".join(cells["fig2-laterality-plot"]["source"]), ns)
print("done", flush=True)
PY
