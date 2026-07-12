#!/bin/bash
#SBATCH --job-name=atlas_conflict_slices
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula/logs/atlas_conflict_slices_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula/logs/atlas_conflict_slices_%j.err

set -euo pipefail

SCRIPT_DIR=/hpc/group/coganlab/nanlinshi/insula/pipeline/05_visual_qc
PYTHON=/hpc/home/ns458/miniconda3/envs/ieeg/bin/python

mkdir -p /hpc/group/coganlab/nanlinshi/insula/logs
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-atlas_conflict_slices}"
mkdir -p "$MPLCONFIGDIR"
"$PYTHON" "$SCRIPT_DIR/plot_atlas_conflict_slices.py" "$@"
