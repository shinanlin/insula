#!/bin/bash
#SBATCH --job-name=paper_fig5_model
#SBATCH --partition=common
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/fig5_model_%A_%a.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/fig5_model_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=/cwork/ns458/insula-paper
export INSULA_RT_COHORT=/cwork/ns458/insula-paper/docs/evidence/sound_only_cohort_2026-10-17/sampled_sites_by_subject_task.csv
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python -u scripts/refresh_figure5.py fit "$SLURM_ARRAY_TASK_ID"
