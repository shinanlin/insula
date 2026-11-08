#!/bin/bash
#SBATCH --job-name=fig3_manual14_audit
#SBATCH --account=coganlab
#SBATCH --partition=common
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/fig3_manual14_audit_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/fig3_manual14_audit_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python scripts/refresh_fig3_manual14.py audit
