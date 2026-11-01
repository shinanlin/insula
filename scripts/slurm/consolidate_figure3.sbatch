#!/bin/bash
#SBATCH --account=coganlab
#SBATCH --partition=common,scavenger
#SBATCH --job-name=fig3_consolidate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:30:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/fig3_consolidate_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/fig3_consolidate_%j.err
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/cwork/ns458/insula-paper PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python -u scripts/consolidate_figure3.py
