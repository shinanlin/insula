#!/bin/bash
#SBATCH --job-name=viz_cross_task
#SBATCH --output=logs/slurm/viz_insula_patterns_cross_task_%j.out
#SBATCH --error=logs/slurm/viz_insula_patterns_cross_task_%j.err
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -eo pipefail

PROJECT_ROOT="/hpc/group/coganlab/nanlinshi/insula-functional"
cd "$PROJECT_ROOT"
mkdir -p logs/slurm img/insula_patterns

source ~/.bashrc
conda activate ieeg

export PYVISTA_OFF_SCREEN=true
export MNE_3D_BACKEND=notebook

python - <<'PY'
from pathlib import Path
from src.decoding.viz_insula_patterns import (
    load_assignments,
    plot_cross_task_repeat_brain,
)
from src.paths import PROJECT_ROOT, RESULTS_ROOT
from src.univariate.viz_mean import BrainSurfaceContext

out_dir = RESULTS_ROOT / "fig" / "insula_patterns"
out_dir.mkdir(parents=True, exist_ok=True)
assignments = load_assignments(PROJECT_ROOT)
ctx = BrainSurfaceContext()
for feature in ("phoneme", "articulator"):
    path = plot_cross_task_repeat_brain(
        PROJECT_ROOT, assignments, out_dir, feature=feature, ctx=ctx
    )
    print("SAVED", feature, path)
PY
