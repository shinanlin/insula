#!/bin/bash
#SBATCH --job-name=lda_cross_ifg_smoke
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/lda_cross_ifg_smoke_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/lda_cross_ifg_smoke_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
#SBATCH --array=0-1

# IFGl Delay both directions, 50 permutations. Gate production on this
# after Delay within-condition lexicality looks non-empty.

export TASK=LexicalDelay
export SUBJECTS=IFGl
export PHASES=Delay
export DATATYPES=lexicality
export N_PERM=50
export EXPECTED_JOBS=2

bash scripts/decoding_cross_condition_lda_worker.sh
