#!/bin/bash
#SBATCH --job-name=hga_motif_report
#SBATCH --partition=common,scavenger
#SBATCH --account=coganlab
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -eo pipefail

: "${HGA_MOTIF_ROOT:?HGA_MOTIF_ROOT is required}"
: "${HGA_MOTIF_LAGGED_ROOT:?HGA_MOTIF_LAGGED_ROOT is required}"

REPOSITORY=/hpc/group/coganlab/nanlinshi/insula-functional
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
cd "$REPOSITORY"

export PYTHONDONTWRITEBYTECODE=1
export MNE_DONTWRITE_HOME=true
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/xdg-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$XDG_CACHE_HOME"

python scripts/run_hga_motif_decision.py finalize \
  --output-root "$HGA_MOTIF_ROOT" \
  --lagged-root "$HGA_MOTIF_LAGGED_ROOT"
