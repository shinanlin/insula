#!/bin/bash
#SBATCH --job-name=nmf_wb_plot
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/nmf_whole_brain_projection_plot_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/nmf_whole_brain_projection_plot_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

export PYTHONPATH="/hpc/group/coganlab/nanlinshi/insula-functional${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg
export PYVISTA_OFF_SCREEN=true
export MESA_GL_VERSION_OVERRIDE=3.3
export PYVISTA_USE_PANEL=false
export MNE_3D_BACKEND=notebook
export OMP_NUM_THREADS=1

mkdir -p logs/slurm img/nmf/whole_brain_projection

echo "=== Combined focused territories on one brain (from existing npz cache) ==="
python scripts/plot_nmf_whole_brain_projection.py --combined-from-cache

echo "Expected:"
echo "  img/nmf/whole_brain_projection/wholebrain_surface_motifs_focused_combined.svg"
echo "  img/nmf/whole_brain_projection/wholebrain_surface_motifs_focused_combined_pial.svg"
