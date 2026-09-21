#!/bin/bash
#SBATCH --job-name=viz_dvrep_cluster
#SBATCH --output=logs/slurm/viz_decision_repeat_cluster_%j.out
#SBATCH --error=logs/slurm/viz_decision_repeat_cluster_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -eo pipefail

PROJECT_ROOT="/hpc/group/coganlab/nanlinshi/insula-functional"
cd "$PROJECT_ROOT"
mkdir -p logs/slurm img/univariate

source ~/.bashrc
conda activate ieeg

export PYVISTA_OFF_SCREEN=true
export MNE_3D_BACKEND=notebook

python -m src.univariate.viz_cluster_direction
