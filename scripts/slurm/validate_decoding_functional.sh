#!/bin/bash
#SBATCH --job-name=validate_decode_func
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/validate_decoding_functional_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/validate_decoding_functional_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p logs/slurm results/decoding_functional

python -u -m src.decoding.validate_functional_decoding_dataset \
  --lexical-root "/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/derivatives/decoding(bipolar)" \
  --phoneme-root "/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/derivatives/decoding(bipolar)" \
  --intersection-root "/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/derivatives/decoding(intersection)(bipolar)"
