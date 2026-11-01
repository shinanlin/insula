#!/bin/bash
#SBATCH --job-name=fig3_context_draft
#SBATCH --account=coganlab
#SBATCH --partition=common,scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/fig3_context_draft_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/fig3_context_draft_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=/cwork/ns458/insula-paper MPLBACKEND=Agg
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python -u scripts/run_figure3_context_draft.py
