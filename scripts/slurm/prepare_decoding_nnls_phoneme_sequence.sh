#!/bin/bash
#SBATCH --job-name=prep_decode_nnls
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/prepare_decoding_nnls_%j.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/prepare_decoding_nnls_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache_${USER}"
mkdir -p logs/slurm results/decoding_nnls "${NUMBA_CACHE_DIR}"

BIDS_ROOT="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS"
EXTRA_ARGS=(--overwrite)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

python -u -m src.decoding.prepare_nnls_decoding_dataset \
  --bids-root "${BIDS_ROOT}" \
  "${EXTRA_ARGS[@]}"
