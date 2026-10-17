#!/bin/bash
#SBATCH --job-name=insula_restore_verify
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/recon_restore_verify_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/recon_restore_verify_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=common
#SBATCH --chdir=/cwork/ns458/insula-paper
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python scripts/verify_restored_native_anatomy.py
