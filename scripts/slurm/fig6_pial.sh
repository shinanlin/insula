#!/bin/bash
#SBATCH --job-name=fig6_pial
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/fig6_pial_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/fig6_pial_%j.err
#SBATCH --time=00:40:00
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
  /hpc/group/coganlab/nanlinshi/insula-functional/img/fig6

python - <<'PY'
import json
from pathlib import Path

nb = json.loads(Path("fig6.ipynb").read_text())
cells = {c.get("id"): c for c in nb["cells"]}
needed = (
    "fig6-imports",
    "fig6-style",
    "fig6-helpers",
    "fig6-pial",
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
ls -l /hpc/group/coganlab/nanlinshi/insula-functional/img/fig6/wholebrain_surface_motifs_focused_combined_pial_top50.svg
echo "done"
