#!/bin/bash
#SBATCH --job-name=nmf_rank
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/nmf_rank_bootstrap_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/nmf_rank_bootstrap_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=common
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

export PYTHONPATH="/hpc/group/coganlab/nanlinshi/insula-functional${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg
mkdir -p logs/slurm results/nmf img/nmf

echo "=== Electrode bootstrap consensus rank selection (flat results/nmf + img/nmf) ==="
python scripts/run_nmf_rank_selection.py \
  --k-min 2 \
  --k-max 6 \
  --n-boot 200 \
  --row-frac 0.8 \
  --max-iter 5000 \
  --random-state 42 \
  --exclude-channels-file results/nmf/exclude_channels.txt

echo "Done:"
echo "  results/nmf/rank_selection_metrics.csv"
echo "  results/nmf/chosen_k.json"
echo "  img/nmf/rank_metrics.svg"
echo "  img/nmf/consensus_k*.svg"
