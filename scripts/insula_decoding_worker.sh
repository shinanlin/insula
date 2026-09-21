#!/bin/bash
# Window decoding for merged INS pseudo-subjects (INSl/INSr).

set -eo pipefail

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

TASK="${TASK:?Set TASK=LexicalDelay}"
MODE="${MODE:?Set MODE=window}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

SUBJECTS=(INSl INSr)
PHASES=(Stimulus Delay Go Response)
BAND=highgamma
REF=bipolar

case "${TASK}" in
  LexicalDelay)
    BIDS_ROOT="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/"
    DESCRIPTIONS=(Repeat Decision)
    DATATYPES=(phoneme articulator lexicality)
    EXPECTED_JOBS=48
    ;;
  *)
    echo "Unknown TASK=${TASK}" >&2
    exit 2
    ;;
esac

case "${MODE}" in
  window) ;;
  *)
    echo "Unknown MODE=${MODE}" >&2
    exit 2
    ;;
esac

declare -a ALL_SUBJ ALL_DESC ALL_DATA ALL_PHASE
for subject in "${SUBJECTS[@]}"; do
  for description in "${DESCRIPTIONS[@]}"; do
    for datatype in "${DATATYPES[@]}"; do
      for phase in "${PHASES[@]}"; do
        ALL_SUBJ+=("${subject}")
        ALL_DESC+=("${description}")
        ALL_DATA+=("${datatype}")
        ALL_PHASE+=("${phase}")
      done
    done
  done
done

TOTAL_JOBS=${#ALL_SUBJ[@]}
if [[ "${TOTAL_JOBS}" -ne "${EXPECTED_JOBS}" ]]; then
  echo "Job matrix mismatch: ${TOTAL_JOBS} != ${EXPECTED_JOBS}" >&2
  exit 2
fi
if [[ "${TASK_ID}" -ge "${TOTAL_JOBS}" ]]; then
  echo "Array index ${TASK_ID} is outside 0-$((TOTAL_JOBS - 1))" >&2
  exit 2
fi

SUBJECT=${ALL_SUBJ[$TASK_ID]}
DESCRIPTION=${ALL_DESC[$TASK_ID]}
DATATYPE=${ALL_DATA[$TASK_ID]}
PHASE=${ALL_PHASE[$TASK_ID]}
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-16}}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

echo "TASK=${TASK} MODE=${MODE} array=${TASK_ID}/$((TOTAL_JOBS - 1))"
echo "subject=${SUBJECT} description=${DESCRIPTION} datatype=${DATATYPE} phase=${PHASE}"

python -u src/decoding/run_decoding.py \
  --bids_root "${BIDS_ROOT}" \
  --subject "${SUBJECT}" \
  --ref "${REF}" \
  --description "${DESCRIPTION}" \
  --phase "${PHASE}" \
  --band "${BAND}" \
  --datatype "${DATATYPE}" \
  --variance 0.95 \
  --n_perm 200 \
  --n_folds 5 \
  --n_repeats 30 \
  --n_jobs "${N_JOBS}"
