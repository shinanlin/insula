#!/bin/bash
#SBATCH --job-name=fig5_enc
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/fig5_encoding_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/fig5_encoding_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional

set -eo pipefail

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

export PYTHONPATH="/hpc/group/coganlab/nanlinshi/insula-functional${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1

mkdir -p logs/slurm img/fig5

DELAY_CSV=results/rt/summaries/hga_rt_delay_traces.csv
if [[ ! -f "$DELAY_CSV" ]]; then
  echo "===== HGA-RT encoding CSVs (includes Delay traces) ====="
  python -m src.reaction_time.summarize_insula_rt_encoding
else
  echo "===== using existing $DELAY_CSV ====="
fi

echo "===== fig5 encoding panels ====="
cd vizpub
python - <<'PY'
import json
from pathlib import Path

nb = json.loads(Path("fig5.ipynb").read_text())
cells = {c.get("id"): c for c in nb["cells"]}
needed = (
    "fig5-imports",
    "fig5-style",
    "fig5-load",
    "fig5-encoding",
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
find /hpc/group/coganlab/nanlinshi/insula-functional/img/fig5 -name 'fig5_hga_rt_*.svg' | sort
echo "done"
