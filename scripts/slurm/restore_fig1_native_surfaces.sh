#!/bin/bash
#SBATCH --job-name=fig1_native_restore
#SBATCH --partition=common
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/fig1_native_restore_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/fig1_native_restore_%j.err
set -euo pipefail
export PYTHONPATH=/cwork/ns458/.globus-cli
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python scripts/restore_fig1_native_surfaces.py
