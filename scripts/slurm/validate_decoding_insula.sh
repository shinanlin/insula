#!/bin/bash
#SBATCH --job-name=val_decode_ins
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/validate_decoding_insula_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/validate_decoding_insula_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p logs/slurm results/decoding_insula

python -u -m src.decoding.validate_insula_decoding_dataset \
  --lexical-delay-root "/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/derivatives/decoding(bipolar)" \
  --phoneme-root "/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/derivatives/decoding(bipolar)" \
  --bids-task LexicalDelay \
  --report "$(pwd)/results/decoding_insula/LexicalDelay_input_census.json"
