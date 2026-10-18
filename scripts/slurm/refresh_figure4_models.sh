#!/bin/bash
#SBATCH --job-name=paper_fig4_model
#SBATCH --partition=common
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-32%8
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/fig4_model_%A_%a.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/fig4_model_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=/cwork/ns458/insula-paper
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python -u scripts/refresh_figure4.py fit "$SLURM_ARRAY_TASK_ID"
