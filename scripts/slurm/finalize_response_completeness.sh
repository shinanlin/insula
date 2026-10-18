#!/bin/bash
#SBATCH --job-name=insula_response_finalize
#SBATCH --partition=common
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/response_finalize_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/response_finalize_%j.err
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python scripts/finalize_response_completeness.py
