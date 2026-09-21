#!/bin/bash
#SBATCH --job-name=func_res_lex
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/func_res_lex_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/func_res_lex_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
#SBATCH --array=0-71%20

export TASK=LexicalDelay MODE=resolved
bash scripts/functional_decoding_worker.sh

