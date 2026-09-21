#!/bin/bash
# Reproduce residual xcorr, motif inference, lagged-OAEC, figures, and report.

set -eo pipefail

REPOSITORY=/hpc/group/coganlab/nanlinshi/insula-functional
ROOT="$REPOSITORY/results/connectivity/hga_amplitude_motif_decision"
MANIFEST="$ROOT/manifests/four_task_repeat.tsv"
RESIDUAL_ROOT="$ROOT/residualized_connectivity"
LOG_ROOT="$REPOSITORY/logs/hga_amplitude_motif_decision"
MAX_CONCURRENT=${HGA_MOTIF_MAX_CONCURRENT:-20}
N_PERM=${HGA_MOTIF_N_PERM:-1000}

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
cd "$REPOSITORY"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba-hga-motif-master"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-hga-motif-master"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg-hga-motif-master"
mkdir -p "$ROOT/manifests" "$RESIDUAL_ROOT" "$LOG_ROOT"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

python scripts/run_hga_motif_decision.py prepare-manifest --output "$MANIFEST"
N_ROWS=$(($(wc -l < "$MANIFEST") - 1))

RESIDUAL_JOB=$(sbatch \
  --parsable \
  --job-name="hga_xcorr_resid" \
  --array="1-${N_ROWS}%${MAX_CONCURRENT}" \
  --output="$LOG_ROOT/%x_%A_%a.out" \
  --error="$LOG_ROOT/%x_%A_%a.err" \
  --export="ALL,CONNECTIVITY_MANIFEST=${MANIFEST},CONNECTIVITY_OUTPUT_ROOT=${RESIDUAL_ROOT},CONNECTIVITY_METRICS=xcorr_resid,CONNECTIVITY_N_PERM=${N_PERM}" \
  "$REPOSITORY/scripts/slurm/run_pairwise_connectivity.sbatch")

TABLE_JOB=$(sbatch \
  --parsable \
  --dependency="afterok:${RESIDUAL_JOB}" \
  --output="$LOG_ROOT/%x_%j.out" \
  --error="$LOG_ROOT/%x_%j.err" \
  --export="ALL,HGA_MOTIF_MANIFEST=${MANIFEST},HGA_MOTIF_ROOT=${ROOT},HGA_MOTIF_MAX_CONCURRENT=${MAX_CONCURRENT}" \
  "$REPOSITORY/scripts/slurm/run_hga_motif_postprocess.sbatch")

echo "residual_job=${RESIDUAL_JOB} entities=${N_ROWS} table_orchestrator_job=${TABLE_JOB}"
