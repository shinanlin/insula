#!/bin/bash
#SBATCH --job-name=lda_res_art_ld
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/lda_res_art_ld_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/lda_res_art_ld_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
#SBATCH --array=0-15

# 4 ROI x 2 phases x 2 descriptions. Ungrouped folds match the production
# articulator runs this is compared against; the null stays word-level.
export TASK=LexicalDelay
export DATATYPES=articulator
export CV_SCHEME=stratified
export N_PERM=5000
export EXPECTED_JOBS=16

bash scripts/decoding_resolved_lda_worker.sh
