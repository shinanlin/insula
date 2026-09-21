#!/bin/bash
# Worker for shrinkage-LDA / pooled-AUC time-resolved decoding.
#
# Default matrix: Sensory / Sustain / Motor / STGl × Stimulus / Delay.
# Override SUBJECTS and PHASES to expand (e.g. Response, SMCl, MFGl).
# Existing result H5s are skipped unless OVERWRITE=1, so a larger array can
# be submitted without recomputing the first-pass cells.
#
# Environment:
#   TASK           LexicalDelay | PhonemeSequence
#   SUBJECTS       space-separated pools, default "Sensory Sustain Motor STGl"
#   PHASES         space-separated phases, default "Stimulus Delay"
#   DATATYPES      space-separated targets, default "lexicality"
#   CV_SCHEME      group (default, binary lexicality) | stratified (articulator)
#   DROP_LABELS    space-separated string labels to exclude, e.g. "other"
#   OVERWRITE=1    recompute even if the result H5 already exists
#   EXPECTED_JOBS  asserted against the matrix size, to catch an --array range
#                  that has drifted from the matrix
#   PREFLIGHT=1    report which inputs exist for the whole matrix, then exit.
#                  A missing input otherwise exits 0 and Slurm records the
#                  no-op as a success, so check before submitting.

set -eo pipefail

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

TASK="${TASK:-LexicalDelay}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

read -r -a SUBJECTS <<< "${SUBJECTS:-Sensory Sustain Motor STGl}" || true
read -r -a PHASES <<< "${PHASES:-Stimulus Delay}" || true
read -r -a DATATYPES <<< "${DATATYPES:-lexicality}" || true
read -r -a DROP_LABEL_LIST <<< "${DROP_LABELS:-}" || true
CV_SCHEME="${CV_SCHEME:-group}"
BAND=highgamma
REF=bipolar

case "${TASK}" in
  LexicalDelay)
    BIDS_ROOT="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/"
    DESCRIPTIONS=(Repeat Decision)
    ;;
  PhonemeSequence)
    BIDS_ROOT="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/"
    DESCRIPTIONS=(Repeat)
    ;;
  *)
    echo "Unknown TASK=${TASK}" >&2
    exit 2
    ;;
esac

# PhonemeSequence epochs carry a recording entity; LexicalDelay ones do not.
input_h5_for() {
  local subject="$1" description="$2" datatype="$3" phase="$4"
  local stem="sub-${subject}_task-${TASK}_proc-${phase}"
  if [[ "${TASK}" == "PhonemeSequence" ]]; then
    stem="${stem}_recording-1"
  fi
  echo "${BIDS_ROOT%/}/derivatives/decoding(${REF})/sub-${subject}/${datatype}/${stem}_desc-${description}_${BAND}.h5"
}

output_h5_for() {
  local subject="$1" description="$2" datatype="$3" phase="$4"
  local stem="sub-${subject}_proc-${phase}"
  if [[ "${TASK}" == "PhonemeSequence" ]]; then
    stem="${stem}_recording-1"
  fi
  echo "results/decoding/${TASK}/sub-${subject}/(decode)(resolved)(lda)${datatype}/${stem}_desc-${description}_${BAND}.h5"
}

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
EXPECTED_JOBS="${EXPECTED_JOBS:-${TOTAL_JOBS}}"
if [[ "${TOTAL_JOBS}" -ne "${EXPECTED_JOBS}" ]]; then
  echo "Job matrix mismatch: ${TOTAL_JOBS} != ${EXPECTED_JOBS}" >&2
  exit 2
fi

if [[ "${PREFLIGHT:-0}" == "1" ]]; then
  echo "TASK=${TASK} subjects=${SUBJECTS[*]} phases=${PHASES[*]}"
  echo "datatypes=${DATATYPES[*]} matrix=${TOTAL_JOBS}"
  preflight_status=0
  for index in "${!ALL_SUBJ[@]}"; do
    candidate=$(input_h5_for "${ALL_SUBJ[$index]}" "${ALL_DESC[$index]}" \
                             "${ALL_DATA[$index]}" "${ALL_PHASE[$index]}")
    output=$(output_h5_for "${ALL_SUBJ[$index]}" "${ALL_DESC[$index]}" \
                           "${ALL_DATA[$index]}" "${ALL_PHASE[$index]}")
    if [[ -f "${candidate}" ]]; then
      if [[ -f "${output}" ]]; then
        echo "OK   [${index}] input+output exist"
      else
        echo "OK   [${index}] input exists, output missing"
      fi
    else
      echo "MISS [${index}] ${candidate}"
      preflight_status=1
    fi
  done
  exit "${preflight_status}"
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
N_PERM="${N_PERM:-5000}"
N_FOLDS="${N_FOLDS:-5}"
N_BINS="${N_BINS:-5}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

echo "TASK=${TASK} array=${TASK_ID}/$((TOTAL_JOBS - 1))"
echo "subject=${SUBJECT} description=${DESCRIPTION} datatype=${DATATYPE} phase=${PHASE}"
echo "cv_scheme=${CV_SCHEME} drop_labels=${DROP_LABELS:-none}"
echo "n_perm=${N_PERM} n_folds=${N_FOLDS} n_bins=${N_BINS} n_jobs=${N_JOBS}"

INPUT_H5=$(input_h5_for "${SUBJECT}" "${DESCRIPTION}" "${DATATYPE}" "${PHASE}")
if [[ ! -f "${INPUT_H5}" ]]; then
  echo "SKIP: missing prepared input ${INPUT_H5}"
  exit 0
fi

OUTPUT_H5=$(output_h5_for "${SUBJECT}" "${DESCRIPTION}" "${DATATYPE}" "${PHASE}")
if [[ "${OVERWRITE:-0}" != "1" && -f "${OUTPUT_H5}" ]]; then
  echo "SKIP: existing output ${OUTPUT_H5}"
  exit 0
fi

DROP_ARGS=()
if [[ ${#DROP_LABEL_LIST[@]} -gt 0 ]]; then
  DROP_ARGS=(--drop_labels "${DROP_LABEL_LIST[@]}")
fi

python -u src/decoding/run_decoding_resolved_lda.py \
  --bids_root "${BIDS_ROOT}" \
  --subject "${SUBJECT}" \
  --ref "${REF}" \
  --description "${DESCRIPTION}" \
  --phase "${PHASE}" \
  --band "${BAND}" \
  --datatype "${DATATYPE}" \
  --n_bins "${N_BINS}" \
  --window 0.30 \
  --step 0.03 \
  --n_perm "${N_PERM}" \
  --n_folds "${N_FOLDS}" \
  --cv_scheme "${CV_SCHEME}" \
  "${DROP_ARGS[@]}" \
  --n_jobs "${N_JOBS}"
