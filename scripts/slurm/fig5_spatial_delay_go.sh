#!/bin/bash
#SBATCH --job-name=fig5_spat_dg
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/fig5_spatial_delay_go_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/fig5_spatial_delay_go_%j.err
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
export NUMBA_CACHE_DIR=/work/ns458/numba_cache
export MNE_CONFIG_PATH=/work/ns458/mne_config

mkdir -p /hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm \
  /hpc/group/coganlab/nanlinshi/insula-functional/img/fig5 \
  /work/ns458/numba_cache /work/ns458/mne_config

python - <<'PY'
import json
from pathlib import Path

nb = json.loads(Path("fig5.ipynb").read_text())
cells = {c.get("id"): c for c in nb["cells"]}
needed = (
    "fig5-imports",
    "fig5-style",
    "fig5-load",
    "fig5-spatial",
    "fig5-spatial-delay-go",
)
missing = [cid for cid in needed if cid not in cells]
if missing:
    raise SystemExit(f"missing cells: {missing}")
ns = {"__name__": "__main__"}
for cid in ("fig5-imports", "fig5-style", "fig5-load"):
    print(f"=== exec {cid} ===", flush=True)
    src = "".join(cells[cid]["source"])
    src = "\n".join(line for line in src.splitlines() if not line.startswith("%"))
    exec(src, ns)

print("=== exec fig5-spatial (setup render_hemi) ===", flush=True)
src = "".join(cells["fig5-spatial"]["source"])
src = "\n".join(line for line in src.splitlines() if not line.startswith("%"))
exec(src, ns)

print("=== exec fig5-spatial-delay-go ===", flush=True)
src = "".join(cells["fig5-spatial-delay-go"]["source"])
src = "\n".join(line for line in src.splitlines() if not line.startswith("%"))
exec(src, ns)
print("done", flush=True)
PY

echo "===== svg ====="
ls -l /hpc/group/coganlab/nanlinshi/insula-functional/img/fig5/fig5_rt_hit_electrodes_delay_go.svg
echo "done"
