#!/bin/bash
#
# Interactive prototype: D0092 / LexicalDelay / Response / Repeat.
# Set CONNECTIVITY_PAIR_LIMIT for a smaller smoke test.

set -eo pipefail

REPOSITORY=/hpc/group/coganlab/nanlinshi/insula-functional
MANIFEST="${TMPDIR:-/tmp}/LexicalDelay_connectivity_prototype.tsv"
OUTPUT_ROOT=${CONNECTIVITY_OUTPUT_ROOT:-"$REPOSITORY/results/connectivity"}
N_PERM=${CONNECTIVITY_N_PERM:-1000}

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
cd "$REPOSITORY"

export PYTHONDONTWRITEBYTECODE=1
export MNE_DONTWRITE_HOME=true
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-connectivity-prototype"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba-connectivity-prototype"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$OUTPUT_ROOT"

python -m src.connectivity.pairwise.cli build-manifest \
  --dataset LexicalDelay \
  --ready-only \
  --output "$MANIFEST"

ROW_INDEX=$(awk -F '\t' \
  'NR > 1 && $3 == "D0092" && $5 == "Response" && $6 == "Repeat" {print NR - 2; exit}' \
  "$MANIFEST")
if [[ -z "$ROW_INDEX" ]]; then
  echo "D0092/LexicalDelay/Response/Repeat was not found in $MANIFEST" >&2
  exit 3
fi

PAIR_LIMIT_ARGS=()
if [[ -n "${CONNECTIVITY_PAIR_LIMIT:-}" ]]; then
  PAIR_LIMIT_ARGS=(--pair-limit "$CONNECTIVITY_PAIR_LIMIT")
fi

python -m src.connectivity.pairwise.cli run-row \
  --manifest "$MANIFEST" \
  --row-index "$ROW_INDEX" \
  --metrics xcorr oaec wpli \
  --output-root "$OUTPUT_ROOT" \
  --scratch-dir "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}" \
  --n-perm "$N_PERM" \
  --n-jobs "${SLURM_CPUS_PER_TASK:-8}" \
  --pair-block-size 16 \
  --permutation-chunk-size 50 \
  --save-full-null \
  "${PAIR_LIMIT_ARGS[@]}"

ENTITY_DIR="$OUTPUT_ROOT/LexicalDelay/sub-D0092"
python -m src.connectivity.pairwise.cli diagnostics \
  --entity-dir "$ENTITY_DIR" \
  --output "$ENTITY_DIR/diag_prototype.png"
