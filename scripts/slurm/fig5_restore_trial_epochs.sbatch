#!/bin/bash
#SBATCH --account=coganlab
#SBATCH --partition=common,scavenger
#SBATCH --job-name=fig5_restore_HGA
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/fig5_restore_HGA_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/fig5_restore_HGA_%j.err
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/cwork/ns458/insula-paper
if [[ "$1" == fetch ]]; then
    export PYTHONPATH=/cwork/ns458/.globus-cli:/cwork/ns458/insula-paper
fi
exec /hpc/home/ns458/miniconda3/envs/ieeg/bin/python -u scripts/restore_fig5_trial_epochs.py "$@"
