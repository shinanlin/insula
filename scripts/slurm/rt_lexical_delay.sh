#!/bin/bash
# Reaction-time prediction: LexicalDelay (hammers), 49 subjects.
#SBATCH --job-name=rt_lex_delay
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/rt_lexical_delay_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/rt_lexical_delay_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=30
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula
#SBATCH --array=0-48%8

export TASK=LexicalDelay
bash scripts/slurm/run_reaction_time.sh
