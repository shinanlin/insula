#!/bin/bash
#SBATCH --job-name=insula_sound_cohort
#SBATCH --partition=common
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/sound_cohort_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/sound_cohort_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python scripts/build_sound_only_cohort.py
