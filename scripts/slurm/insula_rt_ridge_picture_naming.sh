#!/bin/bash
#SBATCH --job-name=insula_rt_pn
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/insula_rt_pn_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/insula_rt_pn_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=10
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
#SBATCH --array=0-32%8

export TASK=PictureNaming
bash scripts/slurm/run_insula_rt_ridge.sh
