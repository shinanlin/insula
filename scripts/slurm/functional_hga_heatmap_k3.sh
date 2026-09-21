#!/bin/bash
#SBATCH --job-name=fhga_hm_k3
#SBATCH --output=logs/slurm/functional_hga_heatmap_k3_%j.out
#SBATCH --error=logs/slurm/functional_hga_heatmap_k3_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=common
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional

set -eo pipefail
mkdir -p logs/slurm img/functional_hga

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

export PYTHONPATH="/hpc/group/coganlab/nanlinshi/insula-functional${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg

python scripts/plot_functional_hga_heatmap.py --k 3

echo "Done: img/functional_hga/functional_hga_heatmap_k3.svg"
