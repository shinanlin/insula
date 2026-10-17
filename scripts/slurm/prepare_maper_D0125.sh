#!/bin/bash
#SBATCH --job-name=D0125_maper_prepare
#SBATCH --partition=common
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/maper_D0125_prepare_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/maper_D0125_prepare_%j.err
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python scripts/prepare_maper_D0125.py
