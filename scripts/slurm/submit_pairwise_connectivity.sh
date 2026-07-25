#!/bin/bash
#
# Build five ready-only manifests and submit one array per dataset.
# Run the D0092 prototype first; this script intentionally performs no
# prototype or automatic resource extrapolation.

set -eo pipefail

REPOSITORY=/hpc/group/coganlab/nanlinshi/insula-functional
MANIFEST_ROOT="$REPOSITORY/results/connectivity/manifests"
OUTPUT_ROOT="$REPOSITORY/results/connectivity"
LOG_ROOT="$REPOSITORY/logs/connectivity"
SBATCH_SCRIPT="$REPOSITORY/scripts/slurm/run_pairwise_connectivity.sbatch"
MAX_CONCURRENT=${CONNECTIVITY_MAX_CONCURRENT_PER_DATASET:-4}
N_PERM=${CONNECTIVITY_N_PERM:-10000}

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
cd "$REPOSITORY"

export PYTHONDONTWRITEBYTECODE=1
export MNE_DONTWRITE_HOME=true
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-connectivity-submit"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba-connectivity-submit"
mkdir -p "$MANIFEST_ROOT" "$OUTPUT_ROOT" "$LOG_ROOT"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

python -c "import pyarrow" >/dev/null 2>&1 || {
  echo "pyarrow is required for formal Parquet output but is not installed in the ieeg environment." >&2
  exit 2
}

DATASETS=(
  LexicalDelay
  LexicalNoDelay
  PhonemeSequence
  PictureNaming
  SentenceRep
)

for DATASET in "${DATASETS[@]}"; do
  MANIFEST="$MANIFEST_ROOT/${DATASET}_ready.tsv"
  EXCLUDED="$MANIFEST_ROOT/${DATASET}_excluded.tsv"
  python -m src.connectivity.pairwise.cli build-manifest \
    --dataset "$DATASET" \
    --ready-only \
    --excluded-output "$EXCLUDED" \
    --output "$MANIFEST"
  N_ROWS=$(($(wc -l < "$MANIFEST") - 1))
  if [[ "$N_ROWS" -lt 1 ]]; then
    echo "dataset=${DATASET} has no ready entities; not submitting"
    continue
  fi
  JOB_ID=$(sbatch \
    --parsable \
    --job-name="conn_${DATASET}" \
    --array="1-${N_ROWS}%${MAX_CONCURRENT}" \
    --export="ALL,CONNECTIVITY_MANIFEST=${MANIFEST},CONNECTIVITY_OUTPUT_ROOT=${OUTPUT_ROOT},CONNECTIVITY_N_PERM=${N_PERM}" \
    "$SBATCH_SCRIPT")
  echo "dataset=${DATASET} job_id=${JOB_ID} rows=${N_ROWS} manifest=${MANIFEST} excluded=${EXCLUDED}"
done
