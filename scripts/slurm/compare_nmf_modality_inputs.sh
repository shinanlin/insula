#!/bin/bash
#SBATCH --job-name=insula_modality_control
#SBATCH --partition=common
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/nmf_modality_control_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/nmf_modality_control_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python scripts/compare_nmf_modality_inputs.py
