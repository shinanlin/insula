#!/bin/bash
#SBATCH --job-name=ins_win_lex
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/ins_win_lex_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/ins_win_lex_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
#SBATCH --array=0-47%20

export TASK=LexicalDelay MODE=window
bash scripts/insula_decoding_worker.sh
