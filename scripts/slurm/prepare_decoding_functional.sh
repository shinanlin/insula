#!/bin/bash
#SBATCH --job-name=prep_decode_func
#SBATCH --output=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/prepare_decoding_functional_%A_%a.out
#SBATCH --error=/hpc/group/coganlab/nanlinshi/insula-functional/logs/slurm/prepare_decoding_functional_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/group/coganlab/nanlinshi/insula-functional
#SBATCH --array=0-1%2

set -eo pipefail
source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p logs/slurm results/decoding_functional

ASSIGNMENTS="$(pwd)/results/nmf/channel_assignments.csv"
EXTRA_ARGS=(--overwrite)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    BIDS_ROOT="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS"
    BIDS_TASK=LexicalDelay
    ;;
  1)
    BIDS_ROOT="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS"
    BIDS_TASK=PhonemeSequence
    ;;
  *)
    echo "Unexpected array index ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

python -u -m src.decoding.prepare_functional_decoding_dataset \
  --bids-root "${BIDS_ROOT}" \
  --bids-task "${BIDS_TASK}" \
  --assignments "${ASSIGNMENTS}" \
  --expected-assignment-rows 255 \
  "${EXTRA_ARGS[@]}"
