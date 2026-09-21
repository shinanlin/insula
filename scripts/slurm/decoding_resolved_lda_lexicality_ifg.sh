#!/bin/bash
#SBATCH --job-name=lda_res_lex_ifg
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/lda_res_lex_ifg_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/lda_res_lex_ifg_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
#SBATCH --array=0-1

# SHI-59: IFGl Delay within-condition lexicality LDA. Gate the 2D cross jobs
# on a non-empty curve. 1 ROI × Delay × Repeat/Decision = 2.

export TASK=LexicalDelay
export DATATYPES=lexicality
export SUBJECTS=IFGl
export PHASES=Delay
export N_PERM=5000
export EXPECTED_JOBS=2

bash scripts/decoding_resolved_lda_worker.sh
