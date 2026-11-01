#!/bin/bash
#SBATCH --account=coganlab
#SBATCH --partition=common,scavenger
#SBATCH --job-name=pn_region_points
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/pn_region_points_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/pn_region_points_%j.err
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/cwork/ns458/insula-paper PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export PYVISTA_OFF_SCREEN=true PYVISTA_USE_PANEL=false MNE_3D_BACKEND=notebook MESA_GL_VERSION_OVERRIDE=3.3
export NUMBA_CACHE_DIR=/cwork/ns458/insula-paper/results/audit/picture_region_point_panels_2026-10-27/numba_cache
export MNE_CONFIG_PATH=/cwork/ns458/insula-paper/results/audit/picture_region_point_panels_2026-10-27/mne_config
mkdir -p "$NUMBA_CACHE_DIR" "$MNE_CONFIG_PATH"
xvfb-run -a -s '-screen 0 1600x1600x24' /hpc/home/ns458/miniconda3/envs/ieeg/bin/python -u scripts/render_picture_region_point_panels.py
