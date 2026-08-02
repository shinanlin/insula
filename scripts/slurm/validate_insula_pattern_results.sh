#!/bin/bash
#SBATCH --job-name=val_ins_pat
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/validate_insula_pattern_results_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/validate_insula_pattern_results_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

python -u -m src.decoding.validate_insula_pattern_results \
  --report "$(pwd)/results/decoding/pattern_census.json"
