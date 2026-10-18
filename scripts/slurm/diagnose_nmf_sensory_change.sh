#!/bin/bash
#SBATCH --job-name=insula_sensory_audit
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/nmf_sensory_audit_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/nmf_sensory_audit_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python scripts/diagnose_nmf_sensory_change.py
