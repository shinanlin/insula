#!/bin/bash
# One-subject worker for strict-insula, item-grouped time-resolved RT ridge.

set -eo pipefail

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
module purge

PROJECT_ROOT=/hpc/group/coganlab/nanlinshi/insula-functional
OUTPUT_ROOT="${OUTPUT_ROOT:-/hpc/group/coganlab/nanlinshi/insula-functional/results/rt}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${PROJECT_ROOT}/logs/slurm"

TASK="${TASK:?Set TASK to LexicalDelay, PhonemeSequence, or PictureNaming}"
WINDOW_S="${WINDOW_S:-0.2}"
STEP_S="${STEP_S:-0.02}"
N_PERM="${N_PERM:-1000}"
N_FOLDS="${N_FOLDS:-10}"
INNER_FOLDS="${INNER_FOLDS:-5}"
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-10}}"
RANDOM_STATE="${RANDOM_STATE:-42}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

case "${TASK}" in
  LexicalDelay)
    BIDS_ROOT=/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS
    SUBJECTS=(
      D0023 D0024 D0026 D0027 D0028 D0029 D0032 D0035 D0038 D0042
      D0044 D0047 D0053 D0054 D0055 D0057 D0059 D0063 D0065 D0066
      D0068 D0069 D0070 D0071 D0077 D0079 D0080 D0081 D0084 D0086
      D0090 D0092 D0094 D0096 D0100 D0101 D0102 D0103 D0115 D0117
      D0127 D0128 D0129 D0132 D0135 D0137 D0138 D0140 D0143
    )
    ;;
  PhonemeSequence)
    BIDS_ROOT=/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS
    SUBJECTS=(
      D0019 D0022 D0023 D0024 D0025 D0028 D0029 D0035 D0041 D0042
      D0045 D0049 D0052 D0053 D0054 D0055 D0056 D0057 D0058 D0059
      D0060 D0061 D0063 D0064 D0066 D0067 D0068 D0069 D0070 D0071
      D0073 D0075 D0077 D0079 D0084 D0085 D0086 D0088 D0092 D0093
      D0094 D0095 D0096 D0100 D0102 D0103
    )
    ;;
  PictureNaming)
    BIDS_ROOT=/cwork/ns458/BIDS-1.3_PictureNaming/BIDS
    SUBJECTS=(
      D0076 D0077 D0079 D0080 D0081 D0084 D0085 D0086 D0088 D0090
      D0092 D0093 D0094 D0096 D0097 D0100 D0101 D0102 D0105 D0108
      D0118 D0119 D0122 D0123 D0125 D0126 D0129 D0130 D0131 D0134
      D0135 D0137 D0138
    )
    ;;
  *)
    echo "Unsupported TASK=${TASK}" >&2
    exit 2
    ;;
esac

if [[ -n "${SUBJECT_OVERRIDE:-}" ]]; then
  SUBJECT="${SUBJECT_OVERRIDE}"
else
  TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
  if (( TASK_ID >= ${#SUBJECTS[@]} )); then
    echo "SLURM_ARRAY_TASK_ID=${TASK_ID} is out of bounds" >&2
    exit 2
  fi
  SUBJECT="${SUBJECTS[$TASK_ID]}"
fi

ARGS=(
  --bids-root "${BIDS_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --task "${TASK}"
  --subject "${SUBJECT}"
  --phases Delay Go
  --description Repeat
  --band highgamma
  --ref bipolar
  --atlas hammers
  --window-s "${WINDOW_S}"
  --step-s "${STEP_S}"
  --n-folds "${N_FOLDS}"
  --inner-folds "${INNER_FOLDS}"
  --n-perm "${N_PERM}"
  --random-state "${RANDOM_STATE}"
  --n-jobs "${N_JOBS}"
)
if [[ -n "${MAX_WINDOWS:-}" ]]; then
  ARGS+=(--max-windows "${MAX_WINDOWS}")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

echo "task=${TASK} subject=${SUBJECT} output=${OUTPUT_ROOT}"
echo "window=${WINDOW_S} step=${STEP_S} outer=${N_FOLDS} inner=${INNER_FOLDS} permutations=${N_PERM}"
python -u src/reaction_time/run_insula_rt_ridge.py "${ARGS[@]}"
