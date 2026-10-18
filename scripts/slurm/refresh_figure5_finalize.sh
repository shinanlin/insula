#!/bin/bash
#SBATCH --job-name=paper_fig5_finalize
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/fig5_finalize_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/fig5_finalize_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/cwork/ns458/insula-paper
export INSULA_RT_COHORT=/cwork/ns458/insula-paper/docs/evidence/sound_only_cohort_2026-10-17/sampled_sites_by_subject_task.csv
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python -u scripts/refresh_figure5.py summarize
bash scripts/slurm/refresh_paper_notebook.sh 5
