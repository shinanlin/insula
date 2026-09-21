#!/bin/bash
#SBATCH --job-name=ld_lex_rerun
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/ld_lex_rerun_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/ld_lex_rerun_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
#SBATCH --array=0-23%24

bash scripts/window_decode_ld_lexicality_rerun_worker.sh
