#!/bin/bash
#SBATCH --job-name=test_lda_cross
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/test_lda_cross_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/test_lda_cross_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p logs/slurm

python -m pytest -q tests/test_decoding_lda_cross.py
