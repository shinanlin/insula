#!/bin/bash
#SBATCH --job-name=hga_motif_tables
#SBATCH --partition=common,scavenger
#SBATCH --account=coganlab
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -eo pipefail

: "${HGA_MOTIF_MANIFEST:?HGA_MOTIF_MANIFEST is required}"
: "${HGA_MOTIF_ROOT:?HGA_MOTIF_ROOT is required}"

REPOSITORY=/hpc/group/coganlab/nanlinshi/insula-functional
BASELINE_ROOT="$REPOSITORY/results/connectivity"
RESIDUAL_ROOT="$HGA_MOTIF_ROOT/residualized_connectivity"
CANDIDATE_MANIFEST="$HGA_MOTIF_ROOT/manifests/lagged_oaec_candidates.tsv"
LAGGED_ROOT="$HGA_MOTIF_ROOT/lagged_oaec_connectivity"
LOG_ROOT="$REPOSITORY/logs/hga_amplitude_motif_decision"
MAX_CONCURRENT=${HGA_MOTIF_MAX_CONCURRENT:-20}

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
cd "$REPOSITORY"

export PYTHONDONTWRITEBYTECODE=1
export MNE_DONTWRITE_HOME=true
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/xdg-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$XDG_CACHE_HOME" "$LOG_ROOT"

python -m src.connectivity.pairwise.cli audit \
  --manifest "$HGA_MOTIF_MANIFEST" \
  --metrics xcorr_resid \
  --output-root "$RESIDUAL_ROOT" \
  --output "$HGA_MOTIF_ROOT/qc/residual_xcorr_audit.tsv"

if [[ $(($(wc -l < "$HGA_MOTIF_ROOT/qc/residual_xcorr_audit_retry.tsv") - 1)) -ne 0 ]]; then
  echo "Residual xcorr audit contains incomplete entities" >&2
  exit 2
fi

python scripts/run_hga_motif_decision.py build-tables \
  --manifest "$HGA_MOTIF_MANIFEST" \
  --baseline-root "$BASELINE_ROOT" \
  --residual-root "$RESIDUAL_ROOT" \
  --output-root "$HGA_MOTIF_ROOT"

N_ROWS=$(($(wc -l < "$CANDIDATE_MANIFEST") - 1))
if [[ "$N_ROWS" -lt 1 ]]; then
  echo "No lagged-OAEC candidate entities" >&2
  exit 3
fi

LAGGED_JOB=$(sbatch \
  --parsable \
  --array="1-${N_ROWS}%${MAX_CONCURRENT}" \
  --output="$LOG_ROOT/%x_%A_%a.out" \
  --error="$LOG_ROOT/%x_%A_%a.err" \
  --export="ALL,HGA_MOTIF_CANDIDATE_MANIFEST=${CANDIDATE_MANIFEST},HGA_MOTIF_LAGGED_ROOT=${LAGGED_ROOT}" \
  "$REPOSITORY/scripts/slurm/run_hga_motif_lagged_oaec.sbatch")

FINAL_JOB=$(sbatch \
  --parsable \
  --dependency="afterok:${LAGGED_JOB}" \
  --output="$LOG_ROOT/%x_%j.out" \
  --error="$LOG_ROOT/%x_%j.err" \
  --export="ALL,HGA_MOTIF_ROOT=${HGA_MOTIF_ROOT},HGA_MOTIF_LAGGED_ROOT=${LAGGED_ROOT}" \
  "$REPOSITORY/scripts/slurm/run_hga_motif_finalize.sbatch")

echo "lagged_job=${LAGGED_JOB} candidate_entities=${N_ROWS} final_job=${FINAL_JOB}"
