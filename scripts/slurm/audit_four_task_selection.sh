#!/bin/bash
#SBATCH --job-name=insula_four_task_audit
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/four_task_audit_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/four_task_audit_%j.err
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python scripts/audit_four_task_selection.py
