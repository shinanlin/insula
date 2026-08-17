#!/bin/bash
#SBATCH --job-name=nmf_concat
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/nmf_concat_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/nmf_concat_%j.err
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

echo "=== Concat-NMF (postonset, k=3, exclude_channels.txt) ==="
python scripts/plot_nmf_concat_phases.py \
  --windows postonset \
  --k 3 \
  --k-max 3 \
  --exclude-channels-file results/nmf/exclude_channels.txt

echo "Done:"
echo "  results/nmf/"
echo "  img/nmf/"
