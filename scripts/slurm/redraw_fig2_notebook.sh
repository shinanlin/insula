#!/bin/bash
#SBATCH --job-name=insula_fig2_redraw
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/fig2_redraw_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/fig2_redraw_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python scripts/verify_fig2_redraw_inputs.py
bash scripts/slurm/execute_fig2_notebook.sh
