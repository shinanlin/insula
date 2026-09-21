#!/bin/bash
# LexicalDelay lexicality-only window re-run after feature-specific param lock.
# Matrix: 3 subjects × 2 descriptions × 4 phases = 24.

set -eo pipefail

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
BIDS_ROOT="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/"
REF=bipolar
BAND=highgamma
DATATYPE=lexicality
VARIANCE=0.95
C=1.0
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-16}}"

SUBJECTS=(Sensory Sustain Motor)
DESCRIPTIONS=(Repeat Decision)
PHASES=(Stimulus Delay Go Response)

declare -a ALL_SUBJ ALL_DESC ALL_PHASE
for subject in "${SUBJECTS[@]}"; do
  for description in "${DESCRIPTIONS[@]}"; do
    for phase in "${PHASES[@]}"; do
      ALL_SUBJ+=("${subject}")
      ALL_DESC+=("${description}")
      ALL_PHASE+=("${phase}")
    done
  done
done

TOTAL=${#ALL_SUBJ[@]}
EXPECTED=24
if [[ "${TOTAL}" -ne "${EXPECTED}" ]]; then
  echo "Job matrix mismatch: ${TOTAL} != ${EXPECTED}" >&2
  exit 2
fi
if [[ "${TASK_ID}" -ge "${TOTAL}" ]]; then
  echo "Array index ${TASK_ID} outside 0-$((TOTAL - 1))" >&2
  exit 2
fi

SUBJECT=${ALL_SUBJ[$TASK_ID]}
DESCRIPTION=${ALL_DESC[$TASK_ID]}
PHASE=${ALL_PHASE[$TASK_ID]}

INPUT_H5="${BIDS_ROOT%/}/derivatives/decoding(${REF})/sub-${SUBJECT}/${DATATYPE}/sub-${SUBJECT}_task-LexicalDelay_proc-${PHASE}_desc-${DESCRIPTION}_${BAND}.h5"
if [[ ! -f "${INPUT_H5}" ]]; then
  echo "SKIP: missing prepared input ${INPUT_H5}"
  exit 0
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

echo "ld_lex_window ${TASK_ID}/$((TOTAL - 1)): subject=${SUBJECT} desc=${DESCRIPTION} phase=${PHASE} variance=${VARIANCE}"

python -u src/decoding/run_decoding.py \
  --bids_root "${BIDS_ROOT}" \
  --subject "${SUBJECT}" \
  --ref "${REF}" \
  --description "${DESCRIPTION}" \
  --phase "${PHASE}" \
  --band "${BAND}" \
  --datatype "${DATATYPE}" \
  --variance "${VARIANCE}" \
  --C "${C}" \
  --n_perm 200 \
  --n_folds 5 \
  --n_repeats 30 \
  --n_jobs "${N_JOBS}"
