#!/bin/bash
# Prepare only IFGl PhonemeSequence Repeat articulator inputs needed below.
# The project-local wrapper injects the missing Hammers IFG mapping and limits
# the external preparation to Stimulus, Delay, and Response.
#SBATCH --job-name=prep_ifgl_art_ps
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/prep_ifgl_art_ps_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/prep_ifgl_art_ps_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional

set -eo pipefail

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
set -u

mkdir -p logs/slurm

python -u scripts/prepare_decoding_ifgl_articulator.py \
  --bids-root /cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS
