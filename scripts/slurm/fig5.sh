#!/bin/bash
#SBATCH --job-name=fig5
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/fig5_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/fig5_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional/vizpub

set -eo pipefail

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

export PYTHONPATH="/hpc/group/coganlab/nanlinshi/insula-functional${PYTHONPATH:+:${PYTHONPATH}}"
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_PANEL=false
export MNE_3D_BACKEND=notebook
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1

mkdir -p /hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm \
  /hpc/group/coganlab/nanlinshi/insula-functional/img/fig5

python - <<'PY'
import json
from pathlib import Path

nb = json.loads(Path("fig5.ipynb").read_text())
cells = {c.get("id"): c for c in nb["cells"]}
needed = (
    "fig5-imports",
    "fig5-style",
    "fig5-load",
    "fig5-raster",
    "fig5-encoding",
    "fig5-spatial",
)
missing = [cid for cid in needed if cid not in cells]
if missing:
    raise SystemExit(f"missing cells: {missing}")
ns = {"__name__": "__main__"}
for cid in needed:
    print(f"=== exec {cid} ===", flush=True)
    src = "".join(cells[cid]["source"])
    src = "\n".join(
        line for line in src.splitlines() if not line.startswith("%")
    )
    exec(src, ns)
print("done", flush=True)
PY

echo "===== svg ====="
find /hpc/group/coganlab/nanlinshi/insula-functional/img/fig5 -type f | sort
echo "done"
