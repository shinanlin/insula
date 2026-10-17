#!/bin/bash
#SBATCH --job-name=insula_four_task_inventory
#SBATCH --partition=common
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/four_task_inventory_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/four_task_inventory_%j.err
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python scripts/inventory_four_task_selection.py
