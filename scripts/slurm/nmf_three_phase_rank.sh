#!/bin/bash
#SBATCH --job-name=insula_reviewed_rank
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/nmf_rank_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/nmf_rank_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=common
#SBATCH --chdir=/cwork/ns458/insula-paper
set -euo pipefail
for output in results results/nmf img img/nmf; do
  [[ ! -L "$output" ]] || { echo "Refusing symlinked output: $output" >&2; exit 2; }
done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python scripts/fit_reviewed_nmf.py --rank-only
