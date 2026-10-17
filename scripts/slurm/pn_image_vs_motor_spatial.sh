#!/bin/bash
#SBATCH --job-name=pn_spatial
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/pn_image_vs_motor_spatial_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/pn_image_vs_motor_spatial_%j.err
#SBATCH --time=00:40:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional

set -eo pipefail

mkdir -p logs/slurm

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

export PYTHONPATH="/hpc/group/coganlab/nanlinshi/insula-functional${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg
export PYVISTA_OFF_SCREEN=true
export MESA_GL_VERSION_OVERRIDE=3.3
export PYVISTA_USE_PANEL=false
export MNE_3D_BACKEND=notebook
export OMP_NUM_THREADS=1

python scripts/plot_pn_image_vs_motor_spatial.py
