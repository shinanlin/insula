#!/bin/bash
#SBATCH --job-name=export_analysis_svg
#SBATCH --output=logs/slurm/export_analysis_svgs_%j.out
#SBATCH --error=logs/slurm/export_analysis_svgs_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

set -eo pipefail

PROJECT_ROOT="/hpc/group/coganlab/nanlinshi/insula-functional"
cd "$PROJECT_ROOT"
mkdir -p logs/slurm img/{decode,decode_functional,modulation,connectivity}

source ~/.bashrc
conda activate ieeg

export PYVISTA_OFF_SCREEN=true
export MNE_3D_BACKEND=notebook
export MPLBACKEND=Agg

# Optional args: subset of exporters, e.g. connectivity decode_functional
python scripts/export_analysis_svgs.py "$@"
