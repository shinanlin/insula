#!/bin/bash
#SBATCH --job-name=ins_pat_phon
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/ins_pat_phon_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/ins_pat_phon_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
#SBATCH --array=0-15%4

export TASK=PhonemeSequence
bash scripts/insula_pattern_worker.sh
