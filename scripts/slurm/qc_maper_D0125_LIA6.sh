#!/bin/bash
#SBATCH --job-name=D0125_LIA6_qc
#SBATCH --partition=common
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/maper_D0125_LIA6_qc_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/maper_D0125_LIA6_qc_%j.err
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python scripts/qc_maper_D0125_LIA6.py
