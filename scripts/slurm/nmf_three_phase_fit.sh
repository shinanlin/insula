#!/bin/bash
#SBATCH --job-name=insula_reviewed_nmf
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/nmf_reviewed_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/nmf_reviewed_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --partition=common
#SBATCH --chdir=/cwork/ns458/insula-paper
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
for output in results results/nmf img img/nmf; do
  [[ ! -L "$output" ]] || { echo "Refusing symlinked output: $output" >&2; exit 2; }
done
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python scripts/prepare_reviewed_nmf.py
/hpc/home/ns458/miniconda3/envs/ieeg/bin/python scripts/fit_reviewed_nmf.py \
  --reference-run docs/evidence/figure2_refresh_2026-10-17
