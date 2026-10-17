#!/bin/bash
#SBATCH --job-name=insula_selection_audit
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/selection_audit_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/selection_audit_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=common
#SBATCH --chdir=/cwork/ns458/insula-paper

set -euo pipefail
export PYTHONPATH=/cwork/ns458/insula-paper
export MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python scripts/audit_fig1_nmf_selection.py \
  --source /hpc/group/coganlab/nanlinshi/insula-functional/results \
  --out /cwork/ns458/insula-paper/results/audit/fig1_nmf_selection_2026-10-17 \
  --workers 4
