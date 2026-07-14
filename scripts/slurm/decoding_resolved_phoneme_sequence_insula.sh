#!/bin/bash
# Time-resolved decoding: PhonemeSequence, insula ROIs only (AIC + PIC).
# 4 subjects × 1 description × 2 datatypes × 4 phases = 32 array tasks.
#SBATCH --job-name=decode_res_phon_insula
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/decoding_resolved_phoneme_insula_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/decoding_resolved_phoneme_insula_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula
#SBATCH --array=0-31%20

export TASK=PhonemeSequence
bash scripts/decoding_resolved_insula.sh
