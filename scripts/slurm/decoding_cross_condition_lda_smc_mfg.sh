#!/bin/bash
#SBATCH --job-name=lda_cross_smc_mfg
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/lda_cross_smc_mfg_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/lda_cross_smc_mfg_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
#SBATCH --array=0-3

# SHI-59 Week 80: SMCl / MFGl Delay Repeat↔Decision 2D LDA. 2 ROI × 2 directions = 4.

export TASK=LexicalDelay
export SUBJECTS="SMCl MFGl"
export PHASES=Delay
export DATATYPES=lexicality
export N_PERM=5000
export EXPECTED_JOBS=4

bash scripts/decoding_cross_condition_lda_worker.sh
