#!/bin/bash
#SBATCH --job-name=insula_globus_inventory
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/globus_recon_inventory_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/globus_recon_inventory_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --partition=common
#SBATCH --chdir=/cwork/ns458/insula-paper
set -euo pipefail
export PYTHONPATH=/cwork/ns458/.globus-cli
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python scripts/inventory_globus_recon_restore.py "$@"
