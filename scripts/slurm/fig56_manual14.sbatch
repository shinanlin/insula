#!/bin/bash
#SBATCH --account=coganlab
#SBATCH --partition=common,scavenger
#SBATCH --job-name=fig56_refresh
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --chdir=/cwork/ns458/insula-paper
#SBATCH --output=/cwork/ns458/insula-paper/logs/slurm/fig56_refresh_%A_%a.out
#SBATCH --error=/cwork/ns458/insula-paper/logs/slurm/fig56_refresh_%A_%a.err
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=/cwork/ns458/insula-paper MPLBACKEND=Agg
export INSULA_HGA_INPUT_ROOT=/hpc/group/coganlab/nanlinshi/insula-functional/results/hga
export PYVISTA_OFF_SCREEN=true PYVISTA_USE_PANEL=false MNE_3D_BACKEND=notebook MESA_GL_VERSION_OVERRIDE=3.3
if [[ "$1" == surface || "$1" == execute ]]; then
    exec xvfb-run -a -n "$((SLURM_JOB_ID % 20000 + 100))" -s '-screen 0 1600x1600x24' /hpc/home/ns458/miniconda3/envs/ieeg/bin/python -u scripts/refresh_fig56_manual14.py "$@"
fi
exec /hpc/home/ns458/miniconda3/envs/ieeg/bin/python -u scripts/refresh_fig56_manual14.py "$@"
