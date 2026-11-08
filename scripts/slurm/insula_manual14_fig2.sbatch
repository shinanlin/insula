#!/bin/bash
#SBATCH --job-name=insula_manual14_fig2
#SBATCH --account=coganlab
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/insula_manual14_fig2_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/insula_manual14_fig2_%j.err
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYVISTA_OFF_SCREEN=true PYVISTA_USE_PANEL=false MNE_3D_BACKEND=notebook
export MPLBACKEND=Agg MESA_GL_VERSION_OVERRIDE=3.3
export NUMBA_CACHE_DIR=/cwork/ns458/insula-paper/results/insula_manual14_2026-11-07/numba_cache
export MNE_CONFIG_PATH=/cwork/ns458/insula-paper/results/insula_manual14_2026-11-07/mne_config
mkdir -p "$NUMBA_CACHE_DIR" "$MNE_CONFIG_PATH"
if [ "$1" = execute ]; then
    xvfb-run -a -s '-screen 0 1600x1600x24' /hpc/home/ns458/miniconda3/envs/ieeg/bin/python scripts/execute_fig2_notebook.py
else
    /hpc/home/ns458/miniconda3/envs/ieeg/bin/python -u scripts/adopt_insula_manual14_fig2.py "$1"
fi
