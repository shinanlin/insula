#!/bin/bash
#SBATCH --job-name=func_res_phon
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/func_res_phon_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/func_res_phon_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
#SBATCH --array=0-23%12

export TASK=PhonemeSequence MODE=resolved
bash scripts/functional_decoding_worker.sh

