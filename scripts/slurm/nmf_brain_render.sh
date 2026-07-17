#!/usr/bin/env bash
#SBATCH --job-name=nmf_brain
#SBATCH --partition=common,scavenger
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/nmf_brain_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/slurm/nmf_brain_%j.err

set -eo pipefail
source ~/.bashrc || true
conda activate ieeg
export NUMBA_CACHE_DIR=/tmp/numba_cache_${SLURM_JOB_ID}
export PYVISTA_OFF_SCREEN=true
export MESA_GL_VERSION_OVERRIDE=3.3

cd /hpc/group/coganlab/nanlinshi/insula/notebooks
ATLAS="${1:-hammers}"
python /hpc/group/coganlab/nanlinshi/insula/scripts/nmf_brain_render_legacy.py "${ATLAS}"
