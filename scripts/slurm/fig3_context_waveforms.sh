#!/bin/bash
#SBATCH --job-name=fig3_delta
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/fig3_delta_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/fig3_delta_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional/vizpub

set -eo pipefail

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

export PYTHONPATH="/hpc/group/coganlab/nanlinshi/insula-functional${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MNE_DONTWRITE_HOME=true

python - <<'PY'
import json
from pathlib import Path

nb = json.loads(Path("fig3.ipynb").read_text())
cells = {c.get("id"): c for c in nb["cells"]}
ns = {"__name__": "__main__"}
for cid in (
    "fig3-imports",
    "fig3-style",
    "fig3-helpers",
    "fig3-load",
    "fig3-wave-plot",
    "fig3-wave-delta-plot",
    "fig3-wave-delta-by-contrast-plot",
):
    print(f"=== exec {cid} ===", flush=True)
    src = "".join(cells[cid]["source"])
    src = "\n".join(
        line for line in src.splitlines() if not line.startswith("%")
    )
    exec(src, ns)
img = ns["IMG_DIR"]
for name in (
    "fig3_context_pooled_stim_resp.svg",
    "fig3_context_paired_delta.svg",
    "fig3_context_paired_delta_by_contrast.svg",
):
    print("wrote", img / name, flush=True)
PY
