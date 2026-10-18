#!/bin/bash
#SBATCH --job-name=paper_figure
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/paper_figure_%j.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/paper_figure_%j.err
set -euo pipefail
PAPER_ROOT=/cwork/ns458/insula-paper
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PAPER_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYVISTA_OFF_SCREEN=true PYVISTA_USE_PANEL=false MNE_3D_BACKEND=notebook
export MPLBACKEND=Agg MESA_GL_VERSION_OVERRIDE=3.3
export INSULA_HGA_INPUT_ROOT=/hpc/group/coganlab/nanlinshi/insula-functional/results/hga
export INSULA_RT_COHORT="$PAPER_ROOT/docs/evidence/sound_only_cohort_2026-10-17/sampled_sites_by_subject_task.csv"
export NUMBA_CACHE_DIR="$PAPER_ROOT/results/fig${1}/numba_cache"
export MNE_CONFIG_PATH="$PAPER_ROOT/results/fig${1}/mne_config"
mkdir -p "$NUMBA_CACHE_DIR" "$MNE_CONFIG_PATH"
xvfb-run -a -s '-screen 0 1600x1600x24' \
 /hpc/home/ns458/miniconda3/envs/ieeg/bin/python scripts/execute_paper_notebook.py "$@"
