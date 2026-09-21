#!/bin/bash
#SBATCH --job-name=ld_lex_sweep
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/ld_lex_sweep_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/ld_lex_sweep_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
#SBATCH --array=0-287%24

bash scripts/window_decode_param_sweep_ld_lexicality_worker.sh
