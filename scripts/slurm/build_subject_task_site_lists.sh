#!/bin/bash
#SBATCH --job-name=insula_subject_task_lists
#SBATCH --partition=common
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/subject_task_lists_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/subject_task_lists_%j.err
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python scripts/build_subject_task_site_lists.py
