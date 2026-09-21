#!/bin/bash
#SBATCH --job-name=viz_dvrep_k3
#SBATCH --output=logs/slurm/viz_decision_repeat_cluster_k3_%j.out
#SBATCH --error=logs/slurm/viz_decision_repeat_cluster_k3_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=common
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional

set -eo pipefail

mkdir -p logs/slurm img/univariate

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

export PYTHONPATH="/hpc/group/coganlab/nanlinshi/insula-functional${PYTHONPATH:+:${PYTHONPATH}}"
export PYVISTA_OFF_SCREEN=true
export MNE_3D_BACKEND=notebook

python -m src.univariate.viz_cluster_direction --k 3

echo "Wrote k=3 DecisionVsRepeat cluster brains under img/univariate/*_k3.svg"
