#!/bin/bash
#SBATCH --job-name=census_decode_func
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/census_decoding_functional_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/census_decoding_functional_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

python -u -m src.decoding.validate_functional_decoding_results
