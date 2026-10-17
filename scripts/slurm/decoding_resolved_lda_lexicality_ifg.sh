#!/bin/bash
#SBATCH --job-name=lda_res_lex_ifg
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/lda_res_lex_ifg_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/lda_res_lex_ifg_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
#SBATCH --array=0-5

# IFGl within-condition lexicality LDA. The worker skips complete outputs, so
# the existing Delay results are preserved while Stimulus/Response are filled.
# 1 ROI × 3 phases × Repeat/Decision = 6 tasks (4 compute, 2 skip).

export TASK=LexicalDelay
export DATATYPES=lexicality
export SUBJECTS=IFGl
export PHASES="Stimulus Delay Response"
export N_PERM=5000
export EXPECTED_JOBS=6

bash scripts/decoding_resolved_lda_worker.sh
