#!/bin/bash
#SBATCH --job-name=viz_ins_patterns
#SBATCH --output=logs/slurm/viz_insula_patterns_%j.out
#SBATCH --error=logs/slurm/viz_insula_patterns_%j.err
#SBATCH --time=02:00:00
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

python -m src.decoding.viz_insula_patterns
