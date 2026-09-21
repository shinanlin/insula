#!/bin/bash
#SBATCH --job-name=oaec_focused
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/oaec_on_focused_motifs_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/oaec_on_focused_motifs_%j.err
#SBATCH --time=00:40:00
#SBATCH --mem=64G
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

mkdir -p logs/slurm img/nmf/whole_brain_projection \
    results/nmf/whole_brain_projection/visualization

echo "=== Tests: OAEC focused overlay selection ==="
python -m pytest tests/test_oaec_focused_overlay.py -q

echo "=== Overlay OAEC electrodes on focused combined pial ==="
python scripts/plot_oaec_on_focused_motifs.py

echo "Expected:"
echo "  img/nmf/whole_brain_projection/wholebrain_surface_motifs_focused_combined_pial_oaec_electrodes.svg"
echo "  results/nmf/whole_brain_projection/visualization/oaec_overlay_electrodes.csv"
