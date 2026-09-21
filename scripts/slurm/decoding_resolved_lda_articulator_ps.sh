#!/bin/bash
#SBATCH --job-name=lda_res_art_ps
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/lda_res_art_ps_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/lda_res_art_ps_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
#SBATCH --array=0-7

# 4 ROI x 2 phases, Repeat only. sequence_articulator assigns a catch-all
# "other" to phonemes outside the four buckets; it is dropped before modelling.
export TASK=PhonemeSequence
export DATATYPES=articulator
export CV_SCHEME=stratified
export DROP_LABELS=other
export N_PERM=5000
export EXPECTED_JOBS=8

bash scripts/decoding_resolved_lda_worker.sh
