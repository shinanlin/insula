#!/bin/bash
# Time-resolved decoding: PhonemeSequence, hammers ROIs.
# 10 subjects × 1 description × 2 datatypes × 4 phases = 80 array tasks.
#SBATCH --job-name=decode_res_phon
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/decoding_resolved_phoneme_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/decoding_resolved_phoneme_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula
#SBATCH --array=0-79%20

export TASK=PhonemeSequence
bash scripts/decoding_resolved.sh
