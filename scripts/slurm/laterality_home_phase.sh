#!/bin/bash
#SBATCH --job-name=lat_home_hga
#SBATCH --output=logs/slurm/laterality_home_phase_%j.out
#SBATCH --error=logs/slurm/laterality_home_phase_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=common
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional

set -eo pipefail
mkdir -p logs/slurm img/nmf results/nmf

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

export PYTHONPATH="/hpc/group/coganlab/nanlinshi/insula-functional${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg

python scripts/plot_laterality_home_phase.py
