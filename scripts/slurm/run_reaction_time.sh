#!/bin/bash
# Reaction-time prediction worker (one subject per array task).
#
# Invoked by scripts/slurm/rt_*.sh wrappers, or locally:
#   TASK=LexicalDelay SLURM_ARRAY_TASK_ID=0 bash scripts/slurm/run_reaction_time.sh
#
# Cohort: hammers_parcellation_manifest.tsv subjects for each task
# (same subjects used by current hammers packaging / parcellation pipelines).
# Hyperparams match the legacy scripts/run_reaction_time.sh values.

set -eo pipefail

source /hpc/home/ns458/miniconda3/etc/profile.d/conda.sh
conda activate ieeg
module purge

PROJECT_ROOT="/hpc/group/coganlab/nanlinshi/insula"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${PROJECT_ROOT}/logs/slurm"

TASK="${TASK:?Set TASK (LexicalDelay|LexicalNoDelay|PhonemeSequence|PictureNaming)}"

WINDOW=0.2
STEP=0.02
N_PERM=500
N_FOLDS=10
CPUS="${SLURM_CPUS_PER_TASK:-10}"
N_JOBS="${N_JOBS:-${CPUS}}"
BAND=highgamma
REF=bipolar
ATLAS=hammers

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

case "${TASK}" in
  LexicalDelay)
    BIDS_ROOT="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/"
    SUBJECTS=(
      D0023 D0024 D0026 D0027 D0028 D0029 D0032 D0035 D0038 D0042
      D0044 D0047 D0053 D0054 D0055 D0057 D0059 D0063 D0065 D0066
      D0068 D0069 D0070 D0071 D0077 D0079 D0080 D0081 D0084 D0086
      D0090 D0092 D0094 D0096 D0100 D0101 D0102 D0103 D0115 D0117
      D0127 D0128 D0129 D0132 D0135 D0137 D0138 D0140 D0143
    )
    ;;
  LexicalNoDelay)
    BIDS_ROOT="/cwork/ns458/BIDS-1.0_LexicalDecRepNoDelay/BIDS/"
    # hammers_parcellation_manifest (current hammers cohort).
    # Older RT script also listed D0121/D0128 (no hammers) instead of D0133/D0138/D0140.
    SUBJECTS=(
      D0024 D0026 D0027 D0028 D0029 D0053 D0054 D0057 D0063 D0065
      D0069 D0071 D0077 D0086 D0090 D0092 D0094 D0100 D0133 D0137
      D0138 D0140
    )
    ;;
  PhonemeSequence)
    BIDS_ROOT="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/"
    # hammers_parcellation_manifest (drops legacy RT-only D0031/D0040/D0091).
    SUBJECTS=(
      D0019 D0022 D0023 D0024 D0025 D0028 D0029 D0035 D0041 D0042
      D0045 D0049 D0052 D0053 D0054 D0055 D0056 D0057 D0058 D0059
      D0060 D0061 D0063 D0064 D0066 D0067 D0068 D0069 D0070 D0071
      D0073 D0075 D0077 D0079 D0084 D0085 D0086 D0088 D0092 D0093
      D0094 D0095 D0096 D0100 D0102 D0103
    )
    ;;
  PictureNaming)
    BIDS_ROOT="/cwork/ns458/BIDS-1.3_PictureNaming/BIDS/"
    SUBJECTS=(
      D0076 D0077 D0079 D0080 D0081 D0084 D0085 D0086 D0088 D0090
      D0092 D0093 D0094 D0096 D0097 D0100 D0101 D0102 D0105 D0108
      D0118 D0119 D0122 D0123 D0125 D0126 D0129 D0130 D0131 D0134
      D0135 D0137 D0138
    )
    ;;
  *)
    echo "Unknown TASK=${TASK}. Use LexicalDelay|LexicalNoDelay|PhonemeSequence|PictureNaming."
    exit 1
    ;;
esac

N_SUBJ=${#SUBJECTS[@]}
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if [ "${TASK_ID}" -ge "${N_SUBJ}" ]; then
  echo "SLURM_ARRAY_TASK_ID=${TASK_ID} out of bounds (n_subj=${N_SUBJ})."
  exit 0
fi

SUBJECT=${SUBJECTS[$TASK_ID]}

echo "=========================================="
echo "Start time: $(date)"
echo "TASK=${TASK} subject=${SUBJECT} array=${TASK_ID}/$((N_SUBJ - 1))"
echo "bids_root=${BIDS_ROOT}"
echo "atlas=${ATLAS} ref=${REF} band=${BAND}"
echo "window=${WINDOW} step=${STEP} n_perm=${N_PERM} n_folds=${N_FOLDS} n_jobs=${N_JOBS}"
echo "cwd=$(pwd) python=$(which python) conda=${CONDA_DEFAULT_ENV}"
echo "=========================================="

python -u src/reaction_time/run_reaction_time.py \
  --subject "${SUBJECT}" \
  --bids_root "${BIDS_ROOT}" \
  --band "${BAND}" \
  --ref "${REF}" \
  --atlas "${ATLAS}" \
  --window "${WINDOW}" \
  --step "${STEP}" \
  --n_perm "${N_PERM}" \
  --n_folds "${N_FOLDS}" \
  --n_jobs "${N_JOBS}"

echo "Exit code: $?"
echo "End time: $(date)"
