#!/bin/bash
#SBATCH --job-name=lda_cross_lex
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/lda_cross_lex_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/lda_cross_lex_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
#SBATCH --array=0-7

# Production 2D cross-condition lexicality (SHI-59). Submit only after the
# STGl Delay smoke (decoding_cross_condition_lda_smoke.sh) looks sane.
# 4 ROI × Delay × 2 directions = 8.

export TASK=LexicalDelay
export SUBJECTS="Sensory Sustain Motor STGl"
export PHASES=Delay
export DATATYPES=lexicality
export N_PERM=5000
export EXPECTED_JOBS=8

bash scripts/decoding_cross_condition_lda_worker.sh
