#!/bin/bash
#SBATCH --job-name=lda_cross_smoke
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/lda_cross_smoke_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/lda_cross_smoke_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
#SBATCH --array=0-1

# STGl Delay both directions, 50 permutations. Gate the 8-job production
# array on this: STGl within-condition lexicality is already strong.

export TASK=LexicalDelay
export SUBJECTS=STGl
export PHASES=Delay
export DATATYPES=lexicality
export N_PERM=50
export EXPECTED_JOBS=2

bash scripts/decoding_cross_condition_lda_worker.sh
