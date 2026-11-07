#!/bin/bash
#SBATCH --job-name=fig4_formal
#SBATCH --account=coganlab
#SBATCH --partition=common,scavenger
#SBATCH --cpus-per-task=32
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/fig4_formal_%A_%a.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/fig4_formal_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=/cwork/ns458/insula-paper MPLBACKEND=Agg
TASK_MODE=${1:?stage or resolved}
shift
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python -u -m src.decoding.figure4_formal fit --mode "$TASK_MODE" --index "${SLURM_ARRAY_TASK_ID:-0}" --n-jobs "$SLURM_CPUS_PER_TASK" "$@"
