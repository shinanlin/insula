#!/bin/bash
# Time-resolved decoding: LexicalDelay, left-hemisphere ROIs only.
# 5 subjects × 2 descriptions × 3 datatypes × 4 phases = 120 array tasks.
#SBATCH --job-name=decode_res_lex_l
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/decoding_resolved_lexical_left_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/decoding_resolved_lexical_left_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula
#SBATCH --array=0-119%20

export TASK=LexicalDelay
bash scripts/decoding_resolved_left.sh
