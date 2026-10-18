#!/bin/bash
#SBATCH --job-name=insula_unknown_report
#SBATCH --partition=common
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/unknown_report_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/unknown_report_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python scripts/report_unknown_endpoint_update.py
