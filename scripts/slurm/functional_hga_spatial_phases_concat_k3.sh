#!/bin/bash
#SBATCH --job-name=fhga_spat_k3
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/functional_hga_spatial_phases_k3_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/functional_hga_spatial_phases_k3_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=common
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

export PYTHONPATH="/hpc/group/coganlab/nanlinshi/insula-functional${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg
export PYVISTA_OFF_SCREEN=true
export MESA_GL_VERSION_OVERRIDE=3.3
mkdir -p logs/slurm img/nmf_concat_phases/crop_postonset/k3

python scripts/run_functional_hga_spatial_phases_nb.py

echo "Expected: img/nmf_concat_phases/crop_postonset/k3/functional_hga_spatial_phases.svg"
