#!/bin/bash
#SBATCH --job-name=insula_fig1_cohort67
#SBATCH --account=coganlab
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/fig1_cohort67_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/fig1_cohort67_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYVISTA_OFF_SCREEN=true PYVISTA_USE_PANEL=false MNE_3D_BACKEND=notebook
export MPLBACKEND=Agg MESA_GL_VERSION_OVERRIDE=3.3
export NUMBA_CACHE_DIR=/cwork/ns458/insula-paper/results/fig1/numba_cache
export MNE_CONFIG_PATH=/cwork/ns458/insula-paper/results/fig1/mne_config
PAPER_PYTHON=/hpc/home/ns458/miniconda3/envs/ieeg/bin/python
"$PAPER_PYTHON" scripts/refresh_fig1_cohort67.py prepare
"$PAPER_PYTHON" scripts/refresh_fig1_cohort67.py display-inputs
xvfb-run -a -s '-screen 0 1600x1600x24' \
  "$PAPER_PYTHON" scripts/refresh_fig1_cohort67.py execute
