#!/bin/bash
#SBATCH --job-name=nmf_wave_pca
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/nmf_waveform_pca_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/nmf_waveform_pca_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=common
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

export PYTHONPATH="/hpc/group/coganlab/nanlinshi/insula-functional${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg

mkdir -p logs/slurm results/nmf img/nmf

echo "=== Waveform PCA of concat-NMF X (frozen channel_assignments.csv) ==="
python scripts/plot_nmf_waveform_pca.py \
  --exclude-channels-file results/nmf/exclude_channels.txt

echo "Done:"
echo "  results/nmf/waveform_pca_scores.csv"
echo "  results/nmf/waveform_pca_meta.json"
echo "  img/nmf/waveform_pca.svg"
