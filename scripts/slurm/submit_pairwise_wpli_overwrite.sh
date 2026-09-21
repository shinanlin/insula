#!/bin/bash
#
# Overwrite TF-dwPLI only (after null/band-assignment fix).
# Skips xcorr/OAEC. Requires existing ready manifests from the main submit.
#
# Usage:
#   bash scripts/slurm/submit_pairwise_wpli_overwrite.sh
#   CONNECTIVITY_MAX_CONCURRENT_PER_DATASET=5 bash scripts/slurm/submit_pairwise_wpli_overwrite.sh

set -eo pipefail

REPOSITORY=/hpc/group/coganlab/nanlinshi/insula-functional
MANIFEST_ROOT="$REPOSITORY/results/connectivity/manifests"
OUTPUT_ROOT="$REPOSITORY/results/connectivity"
LOG_ROOT="$REPOSITORY/logs/connectivity"
SBATCH_SCRIPT="$REPOSITORY/scripts/slurm/run_pairwise_connectivity.sbatch"
MAX_CONCURRENT=${CONNECTIVITY_MAX_CONCURRENT_PER_DATASET:-10}
N_PERM=${CONNECTIVITY_N_PERM:-1000}

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
cd "$REPOSITORY"

mkdir -p "$LOG_ROOT"

DATASETS=(
  LexicalDelay
  LexicalNoDelay
  PhonemeSequence
  PictureNaming
  SentenceRep
)

echo "Submitting wPLI-only overwrite arrays (n_perm=${N_PERM}, max_concurrent=${MAX_CONCURRENT})"

for DATASET in "${DATASETS[@]}"; do
  MANIFEST="$MANIFEST_ROOT/${DATASET}_ready.tsv"
  if [[ ! -f "$MANIFEST" ]]; then
    echo "dataset=${DATASET} missing manifest ${MANIFEST}; skip"
    continue
  fi
  N_ROWS=$(($(wc -l < "$MANIFEST") - 1))
  if [[ "$N_ROWS" -lt 1 ]]; then
    echo "dataset=${DATASET} has no ready entities; skip"
    continue
  fi
  JOB_ID=$(sbatch \
    --parsable \
    --job-name="conn_wpli_${DATASET}" \
    --mem=64G \
    --array="1-${N_ROWS}%${MAX_CONCURRENT}" \
    --export="ALL,CONNECTIVITY_MANIFEST=${MANIFEST},CONNECTIVITY_OUTPUT_ROOT=${OUTPUT_ROOT},CONNECTIVITY_N_PERM=${N_PERM},CONNECTIVITY_METRICS=wpli,CONNECTIVITY_OVERWRITE=1" \
    "$SBATCH_SCRIPT")
  echo "dataset=${DATASET} job_id=${JOB_ID} rows=${N_ROWS} metrics=wpli overwrite=1 mem=64G"
done
