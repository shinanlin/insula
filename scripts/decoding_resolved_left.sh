#!/bin/bash
# Time-resolved decoding for left-hemisphere pseudo-subjects only.
# Array index maps to: subject × description × datatype × phase (band fixed).
#
# Invoked by scripts/slurm/decoding_resolved_*_left.sh (preferred) or locally:
#   TASK=LexicalDelay bash scripts/decoding_resolved_left.sh

set -eo pipefail

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

TASK="${TASK:-LexicalDelay}"

SUBJECTS=(
  AICl
  PICl
  STGl
  SMCl
  MFGl
)

BANDS=(highgamma)
REF='bipolar'
WINDOW="${WINDOW:-0.3}"
STEP="${STEP:-0.03}"
VARIANCE=0.9
N_PERMUTATIONS=200
N_FOLDS=5

CPUS="${SLURM_CPUS_PER_TASK:-16}"
N_JOBS="${N_JOBS:-${CPUS}}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

case "${TASK}" in
  LexicalDelay)
    BIDS_ROOT="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/"
    DESCRIPTIONS=(Repeat Decision)
    DATATYPES=(phoneme articulator lexicality)
    PHASES=(Stimulus Delay Go Response)
    EXPECTED_JOBS=120
    ;;
  PhonemeSequence)
    BIDS_ROOT="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/"
    DESCRIPTIONS=(Repeat)
    DATATYPES=(phoneme articulator)
    PHASES=(Stimulus Delay Go Response)
    EXPECTED_JOBS=40
    ;;
  *)
    echo "Unknown TASK=${TASK}. Use LexicalDelay or PhonemeSequence."
    exit 1
    ;;
esac

declare -a ALL_SUBJ ALL_BAND ALL_DESC ALL_DATA ALL_PHASE

for subj in "${SUBJECTS[@]}"; do
  for band in "${BANDS[@]}"; do
    for desc in "${DESCRIPTIONS[@]}"; do
      for data in "${DATATYPES[@]}"; do
        for phase in "${PHASES[@]}"; do
          ALL_SUBJ+=("$subj")
          ALL_BAND+=("$band")
          ALL_DESC+=("$desc")
          ALL_DATA+=("$data")
          ALL_PHASE+=("$phase")
        done
      done
    done
  done
done

TOTAL_JOBS=${#ALL_SUBJ[@]}

if [ "${TOTAL_JOBS}" -ne "${EXPECTED_JOBS}" ]; then
  echo "Config mismatch: TASK=${TASK} flattened ${TOTAL_JOBS} jobs, expected ${EXPECTED_JOBS}."
  exit 1
fi

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL_JOBS}" ]; then
  echo "SLURM_ARRAY_TASK_ID (${SLURM_ARRAY_TASK_ID}) out of bounds (total=${TOTAL_JOBS})."
  exit 0
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

SUBJECT=${ALL_SUBJ[$TASK_ID]}
BAND=${ALL_BAND[$TASK_ID]}
TYPE=${ALL_DESC[$TASK_ID]}
DATATYPE=${ALL_DATA[$TASK_ID]}
PHASE=${ALL_PHASE[$TASK_ID]}

echo "TASK=${TASK} array=${TASK_ID}/$((TOTAL_JOBS - 1))"
echo "Combination: subject=${SUBJECT} band=${BAND} desc=${TYPE} datatype=${DATATYPE} phase=${PHASE}"
echo "bids_root=${BIDS_ROOT}"
echo "n_jobs=${N_JOBS} cpus=${CPUS}"
echo "Current working directory: $(pwd)"
echo "Python: $(which python) ($(python --version 2>&1))"
echo "Conda env: ${CONDA_DEFAULT_ENV}"

python -u src/decoding/run_decoding_resolved.py \
  --bids_root "${BIDS_ROOT}" \
  --subject "${SUBJECT}" \
  --ref "${REF}" \
  --description "${TYPE}" \
  --phase "${PHASE}" \
  --band "${BAND}" \
  --datatype "${DATATYPE}" \
  --variance "${VARIANCE}" \
  --window "${WINDOW}" \
  --step "${STEP}" \
  --n_perm "${N_PERMUTATIONS}" \
  --n_folds "${N_FOLDS}" \
  --n_jobs "${N_JOBS}"

echo "Exit code: $?"
