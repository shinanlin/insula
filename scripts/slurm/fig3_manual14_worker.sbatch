#!/bin/bash
#SBATCH --job-name=fig3_manual14_worker
#SBATCH --account=coganlab
#SBATCH --partition=common
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/fig3_manual14_worker_%A_%a.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/fig3_manual14_worker_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python scripts/refresh_fig3_manual14.py worker --index "$SLURM_ARRAY_TASK_ID"
