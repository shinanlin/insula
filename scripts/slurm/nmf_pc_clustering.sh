#!/bin/bash
#SBATCH --job-name=nmf_pc_clust
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/nmf_pc_clustering_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/nmf_pc_clustering_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=common
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

export PYTHONPATH="/hpc/group/coganlab/nanlinshi/insula-functional${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg

mkdir -p logs/slurm results/nmf

echo "=== PC scree + PC-space clustering tables (no figures) ==="
python scripts/run_nmf_pc_clustering.py \
  --exclude-channels-file results/nmf/exclude_channels.txt \
  --n-scree 20 \
  --k-min 2 \
  --k-max 10 \
  --n-iter 500 \
  --random-state 42

echo "Done:"
echo "  results/nmf/pc_scree.csv"
echo "  results/nmf/pc_scores.csv"
echo "  results/nmf/pc_clustering_iterations.csv"
echo "  results/nmf/pc_clustering_metrics.csv"
echo "  results/nmf/pc_clustering_meta.json"
echo "Figures: notebooks/nmf_pc_clustering.ipynb"
