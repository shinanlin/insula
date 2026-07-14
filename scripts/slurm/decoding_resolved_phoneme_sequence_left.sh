#!/bin/bash
# Time-resolved decoding: PhonemeSequence, left-hemisphere ROIs only.
# 5 subjects × 1 description × 2 datatypes × 4 phases = 40 array tasks.
#SBATCH --job-name=decode_res_phon_l
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/decoding_resolved_phoneme_left_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/decoding_resolved_phoneme_left_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula
#SBATCH --array=0-39%20

export TASK=PhonemeSequence
bash scripts/decoding_resolved_left.sh
