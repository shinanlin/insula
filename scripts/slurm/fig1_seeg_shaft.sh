#!/bin/bash
#SBATCH --job-name=fig1_seeg
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/fig1_seeg_shaft_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/fig1_seeg_shaft_%j.err
#SBATCH --time=00:40:00
#SBATCH --mem=32G
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
export MESA_GL_VERSION_OVERRIDE=3.3
export OMP_NUM_THREADS=1
export NUMBA_CACHE_DIR=/work/ns458/numba_cache
export MNE_CONFIG_PATH=/work/ns458/mne_config

mkdir -p /hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm \
  /hpc/group/coganlab/nanlinshi/insula-functional/img/fig1 \
  /work/ns458/numba_cache /work/ns458/mne_config

python - <<'PY'
import json
from pathlib import Path

nb = json.loads(Path("fig1.ipynb").read_text())
cells = {c.get("id"): c for c in nb["cells"]}
needed = (
    "fig1-imports",
    "fig1-style",
    "fig1-shaft-load",
    "fig1-seeg-plot",
)
missing = [cid for cid in needed if cid not in cells]
if missing:
    raise SystemExit(f"missing cells: {missing}")

ns = {"__name__": "__main__"}
for cid in needed:
    print(f"=== exec {cid} ===", flush=True)
    src = "".join(cells[cid]["source"])
    src = "\n".join(line for line in src.splitlines() if not line.startswith("%"))
    exec(src, ns)
print("done", flush=True)
PY

echo "===== svg ====="
ls -l /hpc/group/coganlab/nanlinshi/insula-functional/img/fig1/fig1_insula_electrode_schematic_seeg.svg \
  /hpc/group/coganlab/nanlinshi/insula-functional/img/fig1/fig1_insula_electrode_schematic.svg
echo "done"
