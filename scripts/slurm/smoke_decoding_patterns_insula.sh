#!/bin/bash
#SBATCH --job-name=smoke_pat_ins
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/smoke_decoding_patterns_insula_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/smoke_decoding_patterns_insula_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p logs/slurm results/decoding/LexicalDelay

python -u src/decoding/run_decoding_patterns.py \
  --bids_root /cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/ \
  --subject INSl \
  --ref bipolar \
  --phase Delay \
  --description Repeat \
  --datatype lexicality \
  --n_perm 5 \
  --n_folds 5 \
  --n_jobs 4
