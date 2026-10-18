#!/bin/bash
#SBATCH --job-name=paper_fig4_prepare
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/fig4_prepare_%A_%a.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/fig4_prepare_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/cwork/ns458/insula-paper
TASKS=(PhonemeSequence LexicalDelay)
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python -u scripts/refresh_figure4.py prepare "${TASKS[$SLURM_ARRAY_TASK_ID]}"
