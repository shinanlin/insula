#!/bin/bash
# Reaction-time prediction: PictureNaming (hammers), 33 subjects.
#SBATCH --job-name=rt_pict_naming
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/rt_picture_naming_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/rt_picture_naming_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=10
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula
#SBATCH --array=0-32%8

export TASK=PictureNaming
bash scripts/slurm/run_reaction_time.sh
