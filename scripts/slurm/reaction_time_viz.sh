#!/bin/bash
#SBATCH --job-name=rt_viz
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/reaction_time_viz_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/reaction_time_viz_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional

set -eo pipefail

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

export PYTHONPATH="/hpc/group/coganlab/nanlinshi/insula-functional${PYTHONPATH:+:${PYTHONPATH}}"
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_PANEL=false
export MNE_3D_BACKEND=notebook
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1

mkdir -p logs/slurm img/reaction_time

echo "===== summarize ====="
python -m src.reaction_time.summarize_insula_rt_ridge
python - <<'PY'
import pandas as pd
from pathlib import Path
d = Path("results/rt/summaries")
for name in ["coverage.csv", "electrodes.csv", "significant_clusters.csv"]:
    df = pd.read_csv(d / name)
    phases = df.phase.value_counts().to_dict() if "phase" in df.columns else {}
    print(f"{name}: rows={len(df)} phases={phases}")
PY

echo "===== HGA-RT encoding ====="
python -m src.reaction_time.summarize_insula_rt_encoding
python - <<'PY'
import pandas as pd
from pathlib import Path
d = Path("results/rt/summaries")
for name in [
    "hga_rt_go_traces.csv",
    "hga_rt_delay_traces.csv",
    "hga_rt_response_traces.csv",
    "hga_rt_encoding_electrodes.csv",
    "hga_rt_encoding_subjects.csv",
    "hga_rt_trials.csv",
    "hga_rt_inference.csv",
]:
    path = d / name
    df = pd.read_csv(path)
    print(f"{name}: rows={len(df)}")
PY

echo "===== execute notebook ====="
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 \
  --ExecutePreprocessor.kernel_name=python3 \
  notebooks/reaction_time.ipynb

echo "===== svg ====="
find img/reaction_time -type f | sort
echo "done"
