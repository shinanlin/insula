#!/bin/bash
#SBATCH --job-name=fig4_formal_setup
#SBATCH --account=coganlab
#SBATCH --partition=common,scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/fig4_formal_setup_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/fig4_formal_setup_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=/cwork/ns458/insula-paper MPLBACKEND=Agg
TASK_PY=/hpc/home/ns458/miniconda3/envs/ieeg/bin/python
"$TASK_PY" -m pytest -q tests/test_formal_decoding.py tests/test_repeated_cv_inference.py
"$TASK_PY" -u -m src.decoding.figure4_formal prepare
