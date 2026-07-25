#!/bin/bash
#SBATCH --job-name=insula_conn
#SBATCH --partition=common,scavenger
#SBATCH --account=coganlab
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/connectivity/%x_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/connectivity/%x_%A_%a.err

set -eo pipefail

: "${CONNECTIVITY_MANIFEST:?CONNECTIVITY_MANIFEST is required}"
: "${CONNECTIVITY_OUTPUT_ROOT:?CONNECTIVITY_OUTPUT_ROOT is required}"

REPOSITORY=/hpc/group/coganlab/nanlinshi/insula-functional
ROW_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
N_PERM=${CONNECTIVITY_N_PERM:-10000}

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
cd "$REPOSITORY"

export PYTHONDONTWRITEBYTECODE=1
export MNE_DONTWRITE_HOME=true
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

echo "start=$(date -u +%FT%TZ) host=$(hostname) job=${SLURM_JOB_ID} array=${SLURM_ARRAY_TASK_ID} row=${ROW_INDEX} manifest=${CONNECTIVITY_MANIFEST}"

python -m src.connectivity.pairwise.cli run-row \
  --manifest "$CONNECTIVITY_MANIFEST" \
  --row-index "$ROW_INDEX" \
  --metrics xcorr oaec wpli \
  --output-root "$CONNECTIVITY_OUTPUT_ROOT" \
  --scratch-dir "${SLURM_TMPDIR:-/tmp}" \
  --n-perm "$N_PERM" \
  --n-jobs "${SLURM_CPUS_PER_TASK:-8}" \
  --pair-block-size 32 \
  --permutation-chunk-size 100 \
  --require-parquet

echo "finish=$(date -u +%FT%TZ) job=${SLURM_JOB_ID} array=${SLURM_ARRAY_TASK_ID}"
