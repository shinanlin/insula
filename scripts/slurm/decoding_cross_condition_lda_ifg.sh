#!/bin/bash
#SBATCH --job-name=lda_cross_ifg
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/lda_cross_ifg_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/lda_cross_ifg_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
#SBATCH --array=0-1

# SHI-59: IFGl Delay Repeat↔Decision 2D LDA. Submit after the N_PERM=50 smoke.

export TASK=LexicalDelay
export SUBJECTS=IFGl
export PHASES=Delay
export DATATYPES=lexicality
export N_PERM=5000
export EXPECTED_JOBS=2

bash scripts/decoding_cross_condition_lda_worker.sh
