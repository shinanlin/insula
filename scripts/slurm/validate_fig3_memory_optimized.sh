#!/bin/bash
#SBATCH --job-name=fig3_mem_check
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/fig3_mem_check_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/fig3_mem_check_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
export MPLBACKEND=Agg
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_PANEL=false
export MNE_3D_BACKEND=notebook
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
if [ "$#" -eq 0 ]; then
    set -- --full
fi
python -u scripts/validate_fig3_memory_optimized.py "$@"
