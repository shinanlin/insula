#!/bin/bash
# Worker for shrinkage-LDA 2D cross-condition decoding.
#
# Default matrix: Sensory / Sustain / Motor / STGl × Delay ×
# Repeat→Decision and Decision→Repeat (8 jobs).
#
# Environment:
#   TASK           LexicalDelay (only; Repeat↔Decision pairing)
#   SUBJECTS       space-separated pools, default "Sensory Sustain Motor STGl"
#   PHASES         space-separated phases, default "Delay"
#   DATATYPES      space-separated targets, default "lexicality"
#   DIRECTIONS     space-separated "Train:Test" pairs, default
#                  "Repeat:Decision Decision:Repeat"
#   CV_SCHEME      group (default)
#   DROP_LABELS    space-separated string labels to exclude
#   OVERWRITE=1    recompute even if the result H5 already exists
#   EXPECTED_JOBS  asserted against the matrix size
#   PREFLIGHT=1    report inputs, then exit

set -eo pipefail

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg

TASK="${TASK:-LexicalDelay}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

read -r -a SUBJECTS <<< "${SUBJECTS:-Sensory Sustain Motor STGl}" || true
read -r -a PHASES <<< "${PHASES:-Delay}" || true
read -r -a DATATYPES <<< "${DATATYPES:-lexicality}" || true
read -r -a DIRECTION_LIST <<< "${DIRECTIONS:-Repeat:Decision Decision:Repeat}" || true
read -r -a DROP_LABEL_LIST <<< "${DROP_LABELS:-}" || true
CV_SCHEME="${CV_SCHEME:-group}"
BAND=highgamma
REF=bipolar

if [[ "${TASK}" != "LexicalDelay" ]]; then
  echo "TASK=${TASK} is not supported; cross-condition pairing is LexicalDelay only" >&2
  exit 2
fi

BIDS_ROOT="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/"

input_h5_for() {
  local subject="$1" description="$2" datatype="$3" phase="$4"
  echo "${BIDS_ROOT%/}/derivatives/decoding(${REF})/sub-${subject}/${datatype}/sub-${subject}_task-${TASK}_proc-${phase}_desc-${description}_${BAND}.h5"
}

output_h5_for() {
  local subject="$1" train_on="$2" test_on="$3" datatype="$4" phase="$5"
  echo "results/decoding/${TASK}/sub-${subject}/(cross)(resolved)(lda)${datatype}/sub-${subject}_proc-${phase}_desc-${train_on}2${test_on}_${BAND}.h5"
}

declare -a ALL_SUBJ ALL_TRAIN ALL_TEST ALL_DATA ALL_PHASE
for subject in "${SUBJECTS[@]}"; do
  for direction in "${DIRECTION_LIST[@]}"; do
    train_on="${direction%%:*}"
    test_on="${direction##*:}"
    for datatype in "${DATATYPES[@]}"; do
      for phase in "${PHASES[@]}"; do
        ALL_SUBJ+=("${subject}")
        ALL_TRAIN+=("${train_on}")
        ALL_TEST+=("${test_on}")
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
  echo "directions=${DIRECTION_LIST[*]} datatypes=${DATATYPES[*]} matrix=${TOTAL_JOBS}"
  preflight_status=0
  for index in "${!ALL_SUBJ[@]}"; do
    src=$(input_h5_for "${ALL_SUBJ[$index]}" "${ALL_TRAIN[$index]}" \
                       "${ALL_DATA[$index]}" "${ALL_PHASE[$index]}")
    tgt=$(input_h5_for "${ALL_SUBJ[$index]}" "${ALL_TEST[$index]}" \
                       "${ALL_DATA[$index]}" "${ALL_PHASE[$index]}")
    output=$(output_h5_for "${ALL_SUBJ[$index]}" "${ALL_TRAIN[$index]}" \
                           "${ALL_TEST[$index]}" "${ALL_DATA[$index]}" \
                           "${ALL_PHASE[$index]}")
    if [[ -f "${src}" && -f "${tgt}" ]]; then
      if [[ -f "${output}" ]]; then
        echo "OK   [${index}] input+output exist"
      else
        echo "OK   [${index}] inputs exist, output missing"
      fi
    else
      echo "MISS [${index}] src=${src} tgt=${tgt}"
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
TRAIN_ON=${ALL_TRAIN[$TASK_ID]}
TEST_ON=${ALL_TEST[$TASK_ID]}
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
echo "subject=${SUBJECT} phase=${PHASE} ${TRAIN_ON}→${TEST_ON} datatype=${DATATYPE}"
echo "cv_scheme=${CV_SCHEME} n_perm=${N_PERM} n_folds=${N_FOLDS} n_jobs=${N_JOBS}"

SRC_H5=$(input_h5_for "${SUBJECT}" "${TRAIN_ON}" "${DATATYPE}" "${PHASE}")
TGT_H5=$(input_h5_for "${SUBJECT}" "${TEST_ON}" "${DATATYPE}" "${PHASE}")
if [[ ! -f "${SRC_H5}" || ! -f "${TGT_H5}" ]]; then
  echo "SKIP: missing prepared input ${SRC_H5} or ${TGT_H5}"
  exit 0
fi

OUTPUT_H5=$(output_h5_for "${SUBJECT}" "${TRAIN_ON}" "${TEST_ON}" "${DATATYPE}" "${PHASE}")
if [[ "${OVERWRITE:-0}" != "1" && -f "${OUTPUT_H5}" ]]; then
  echo "SKIP: existing output ${OUTPUT_H5}"
  exit 0
fi

DROP_ARGS=()
if [[ ${#DROP_LABEL_LIST[@]} -gt 0 ]]; then
  DROP_ARGS=(--drop_labels "${DROP_LABEL_LIST[@]}")
fi

python -u src/decoding/run_cross_condition_resolved_lda.py \
  --bids_root "${BIDS_ROOT}" \
  --subject "${SUBJECT}" \
  --ref "${REF}" \
  --phase "${PHASE}" \
  --train_on "${TRAIN_ON}" \
  --test_on "${TEST_ON}" \
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
