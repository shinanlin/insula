#!/bin/bash
#SBATCH --job-name=insula_approved_methods
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/approved_methods_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/approved_methods_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python scripts/apply_approved_summary_methods.py
bash scripts/slurm/execute_fig1_notebook.sh
bash scripts/slurm/redraw_fig2_notebook.sh
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python scripts/apply_approved_summary_methods.py --verify
