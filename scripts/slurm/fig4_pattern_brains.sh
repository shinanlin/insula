#!/bin/bash
# Off-screen Fig 4 supplement: PS articulator vs LD lexicality Haufe brains.
#SBATCH --job-name=fig4_pat_brains
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/fig4_pattern_brains_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/fig4_pattern_brains_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
export PYTHONPATH="/hpc/group/coganlab/nanlinshi/insula-functional${PYTHONPATH:+:${PYTHONPATH}}"
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_PANEL=false
export MNE_3D_BACKEND=notebook
export MPLBACKEND=Agg
export MESA_GL_VERSION_OVERRIDE=3.3
export OMP_NUM_THREADS=1
mkdir -p logs/slurm img/fig4

python -u scripts/plot_fig4_pattern_brains.py
