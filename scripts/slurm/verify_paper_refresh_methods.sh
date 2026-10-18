#!/bin/bash
#SBATCH --job-name=paper_method_checks
#SBATCH --partition=common
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/paper_method_checks_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/paper_method_checks_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python -u scripts/verify_paper_refresh_methods.py
