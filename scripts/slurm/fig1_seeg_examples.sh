#!/bin/bash
#SBATCH --job-name=fig1_seeg_ex
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/fig1_seeg_examples_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/fig1_seeg_examples_%j.err
#SBATCH --time=00:40:00
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
export NUMBA_CACHE_DIR=/work/ns458/numba_cache
export MNE_CONFIG_PATH=/work/ns458/mne_config

mkdir -p /hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm \
  /hpc/group/coganlab/nanlinshi/insula-functional/img/fig1/seeg_examples \
  /work/ns458/numba_cache /work/ns458/mne_config

python scripts/plot_fig1_seeg_shaft_examples.py

echo "===== svg ====="
ls -l /hpc/group/coganlab/nanlinshi/insula-functional/img/fig1/seeg_examples
echo "done"
