#!/bin/bash
#SBATCH --job-name=grant_nmf30_fig2
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/grant_nmf30_fig2_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/grant_nmf30_fig2_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --partition=common,scavenger
#SBATCH --account=coganlab
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
export PYTHONPATH="$PWD${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true PYVISTA_USE_PANEL=false
export MNE_3D_BACKEND=notebook
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LIBGL_ALWAYS_SOFTWARE=1 VTK_DEFAULT_OPENGL_WINDOW=vtkXOpenGLRenderWindow
xvfb-run -a -s '-screen 0 1600x1200x24' python -u scripts/redraw_grant_fig2.py
