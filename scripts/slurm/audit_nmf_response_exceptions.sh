#!/bin/bash
#SBATCH --job-name=insula_exception_audit
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/exception_audit_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/exception_audit_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=common
#SBATCH --chdir=/cwork/ns458/insula-paper
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python scripts/audit_nmf_response_exceptions.py
