#!/bin/bash
# Reaction-time prediction: LexicalNoDelay (hammers), 22 subjects.
#SBATCH --job-name=rt_lex_nodelay
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/rt_lexical_nodelay_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/rt_lexical_nodelay_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=10
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula
#SBATCH --array=0-21%8

export TASK=LexicalNoDelay
bash scripts/slurm/run_reaction_time.sh
