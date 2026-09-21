#!/bin/bash
#SBATCH --job-name=lda_res_lex_exp
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/lda_res_lex_exp_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/lda_res_lex_exp_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
#SBATCH --array=0-35

# Expand the first lexicality LDA pass (SHI-39 / SHI-59): add Response and
# left anatomical controls SMCl / MFGl. 6 ROI × 3 phases × 2 descriptions = 36.
# Existing Stimulus/Delay files for Sensory/Sustain/Motor/STGl are skipped.

export TASK=LexicalDelay
export DATATYPES=lexicality
export SUBJECTS="Sensory Sustain Motor STGl SMCl MFGl"
export PHASES="Stimulus Delay Response"
export N_PERM=5000
export EXPECTED_JOBS=36

bash scripts/decoding_resolved_lda_worker.sh
