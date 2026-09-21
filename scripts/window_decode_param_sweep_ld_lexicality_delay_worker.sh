#!/bin/bash
# LexicalDelay Delay window-decode sweep for lexicality.
# Axes: Delay crop × PCA variance × LinearSVC C × {Sensory,Sustain,Motor} × {Repeat,Decision}.

set -eo pipefail

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
BIDS_ROOT="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/"
RESULTS_ROOT="/hpc/group/coganlab/nanlinshi/insula-functional/results/decoding_sweep/LexicalDelay_lexicality_window_delay"
REF=bipolar
BAND=highgamma
DATATYPE=lexicality
PHASE=Delay
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-16}}"

SUBJECTS=(Sensory Sustain Motor)
DESCRIPTIONS=(Repeat Decision)
# Current default [0, 0.7] plus shorter/longer post-onset crops.
WINDOWS=(0.0:0.5 0.0:0.7 0.0:1.0)
VARIANCES=(0.80 0.90 0.95 0.99)
CS=(0.01 0.1 1.0 10.0)

declare -a ALL_SUBJ ALL_DESC ALL_TMIN ALL_TMAX ALL_VAR ALL_C
for subject in "${SUBJECTS[@]}"; do
  for description in "${DESCRIPTIONS[@]}"; do
    for win in "${WINDOWS[@]}"; do
      tmin="${win%%:*}"
      tmax="${win##*:}"
      for variance in "${VARIANCES[@]}"; do
        for C in "${CS[@]}"; do
          ALL_SUBJ+=("${subject}")
          ALL_DESC+=("${description}")
          ALL_TMIN+=("${tmin}")
          ALL_TMAX+=("${tmax}")
          ALL_VAR+=("${variance}")
          ALL_C+=("${C}")
        done
      done
    done
  done
done

TOTAL=${#ALL_SUBJ[@]}
EXPECTED=288
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
TMIN=${ALL_TMIN[$TASK_ID]}
TMAX=${ALL_TMAX[$TASK_ID]}
VARIANCE=${ALL_VAR[$TASK_ID]}
C=${ALL_C[$TASK_ID]}

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

mkdir -p "${RESULTS_ROOT}" logs/slurm

echo "sweep ${TASK_ID}/$((TOTAL - 1)): subject=${SUBJECT} desc=${DESCRIPTION} tmin=${TMIN} tmax=${TMAX} variance=${VARIANCE} C=${C}"

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
  --tmin "${TMIN}" \
  --tmax "${TMAX}" \
  --results_root "${RESULTS_ROOT}" \
  --n_perm 200 \
  --n_folds 5 \
  --n_repeats 30 \
  --n_jobs "${N_JOBS}"
