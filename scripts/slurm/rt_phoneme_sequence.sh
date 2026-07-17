#!/bin/bash
# Reaction-time prediction: PhonemeSequence (hammers), 46 subjects.
#SBATCH --job-name=rt_phoneme_seq
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/rt_phoneme_sequence_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/rt_phoneme_sequence_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=30
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula
#SBATCH --array=0-45%8

export TASK=PhonemeSequence
bash scripts/slurm/run_reaction_time.sh
