#!/bin/bash
# Submit the fixed four-task Repeat evoked-mean-residualized HGA xcorr analysis.

set -eo pipefail

REPOSITORY=/hpc/group/coganlab/nanlinshi/insula-functional
DECISION_ROOT="$REPOSITORY/results/connectivity/hga_amplitude_motif_decision"
MANIFEST="$DECISION_ROOT/manifests/four_task_repeat.tsv"
OUTPUT_ROOT="$DECISION_ROOT/residualized_connectivity"
LOG_ROOT="$REPOSITORY/logs/hga_amplitude_motif_decision"
SBATCH_SCRIPT="$REPOSITORY/scripts/slurm/run_pairwise_connectivity.sbatch"
MAX_CONCURRENT=${CONNECTIVITY_MAX_CONCURRENT:-20}
N_PERM=${CONNECTIVITY_N_PERM:-1000}

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
cd "$REPOSITORY"

export PYTHONDONTWRITEBYTECODE=1
export MNE_DONTWRITE_HOME=true
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-hga-motif-submit"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba-hga-motif-submit"
mkdir -p "$DECISION_ROOT/manifests" "$OUTPUT_ROOT" "$LOG_ROOT"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

python scripts/run_hga_motif_decision.py prepare-manifest --output "$MANIFEST"
N_ROWS=$(($(wc -l < "$MANIFEST") - 1))

JOB_ID=$(sbatch \
  --parsable \
  --job-name="hga_xcorr_resid" \
  --array="1-${N_ROWS}%${MAX_CONCURRENT}" \
  --output="$LOG_ROOT/%x_%A_%a.out" \
  --error="$LOG_ROOT/%x_%A_%a.err" \
  --export="ALL,CONNECTIVITY_MANIFEST=${MANIFEST},CONNECTIVITY_OUTPUT_ROOT=${OUTPUT_ROOT},CONNECTIVITY_METRICS=xcorr_resid,CONNECTIVITY_N_PERM=${N_PERM}" \
  "$SBATCH_SCRIPT")

echo "job_id=${JOB_ID} rows=${N_ROWS} metric=xcorr_resid manifest=${MANIFEST} output=${OUTPUT_ROOT}"
