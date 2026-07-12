#!/bin/bash
#SBATCH --job-name=xcorr_perm
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --partition=common,scavenger
#SBATCH --chdir=/hpc/home/ns458/coganlab/nanlinshi/insula
#SBATCH --output=/hpc/home/ns458/coganlab/nanlinshi/insula/logs/xcorr_perm_%A_%a.out
#SBATCH --error=/hpc/home/ns458/coganlab/nanlinshi/insula/logs/xcorr_perm_%A_%a.err

source ~/.bashrc
conda activate ieeg
mkdir -p /hpc/home/ns458/coganlab/nanlinshi/insula/logs

# --------------------------------------------------------------------
# Parameterized xcorr permutation runner.
#
# Caller (e.g. submit_all_xcorr_perm.sh) sets:
#   DATASET_IDX      : 0..4 (index into DATASETS array below)
#   SLURM_ARRAY_TASK_ID : index into auto-enumerated subject list
# --------------------------------------------------------------------

# Format: "task_name|BIDS_ROOT|phase1,phase2,..."
DATASETS=(
  "LexicalDelay|/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS/|Stimulus,Delay,Go,Response"
  "LexicalNoDelay|/cwork/ns458/BIDS-1.0_LexicalDecRepNoDelay/BIDS/|Stimulus,Response"
  "PhonemeSequence|/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS/|Stimulus,Delay,Go,Response"
  "PictureNaming|/cwork/ns458/BIDS-1.3_PictureNaming/BIDS/|Stimulus,Delay,Go,Response"
  "SentenceRep|/cwork/ns458/BIDS-1.4_SentenceRep/BIDS/|Stimulus,Delay,Go,Response"
)

: "${DATASET_IDX:?Must set DATASET_IDX (0-4)}"
ENTRY="${DATASETS[$DATASET_IDX]}"
TASK_NAME="$(echo "$ENTRY" | cut -d'|' -f1)"
BIDS_ROOT="$(echo "$ENTRY" | cut -d'|' -f2)"
PHASES_CSV="$(echo "$ENTRY" | cut -d'|' -f3)"
IFS=',' read -r -a PHASES <<< "$PHASES_CSV"

DESCRIPTIONS=('Repeat')
BAND=${BAND:-highgamma}
REFERENCE=${REFERENCE:-bipolar}
N_PERM=${N_PERM:-1000}
ALPHA=${ALPHA:-0.05}
MAX_LAG_S=${MAX_LAG_S:-1.0}

# Auto-enumerate subjects from sig(effective) directory for this dataset
SIG_DIR_PATTERN="${BIDS_ROOT}derivatives/epoch(${REFERENCE})"
mapfile -t SUBJECTS < <(
  find "$SIG_DIR_PATTERN" \
    -path "*/epoch(band)(sig)(effective)/*${BAND}*.h5" 2>/dev/null \
    | sed -E 's|.*/sub-([^/]+)/.*|\1|' \
    | sort -u
)

if [ "${#SUBJECTS[@]}" -eq 0 ]; then
  echo "No subjects found under $SIG_DIR_PATTERN"
  exit 1
fi

SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
if [ -z "$SUBJECT" ]; then
  echo "No subject for array index ${SLURM_ARRAY_TASK_ID}. Total subjects: ${#SUBJECTS[@]}"
  exit 0
fi

echo "Job ID:          ${SLURM_JOB_ID}"
echo "Array Task ID:   ${SLURM_ARRAY_TASK_ID}"
echo "Dataset:         ${TASK_NAME} (idx=${DATASET_IDX})"
echo "BIDS_ROOT:       ${BIDS_ROOT}"
echo "Subject:         ${SUBJECT}"
echo "Phases:          ${PHASES[*]}"
echo "Descriptions:    ${DESCRIPTIONS[*]}"

for DESCRIPTION in "${DESCRIPTIONS[@]}"; do
  for PHASE in "${PHASES[@]}"; do
    echo "Processing: ${SUBJECT} (${TASK_NAME}, ${BAND}, ${PHASE}, ${DESCRIPTION})"
    python -u src/xcorr/run_xcorr_pair_permutation.py \
      --bids_root "$BIDS_ROOT" \
      --subject "$SUBJECT" \
      --phase "$PHASE" \
      --description "$DESCRIPTION" \
      --band "$BAND" \
      --reference "$REFERENCE" \
      --max_lag_s "$MAX_LAG_S" \
      --n_perm "$N_PERM" \
      --alpha "$ALPHA" \
      > /hpc/home/ns458/coganlab/nanlinshi/insula/logs/xcorr_perm_${TASK_NAME}_${SUBJECT}_${BAND}_${PHASE}_${DESCRIPTION}.out \
      2> /hpc/home/ns458/coganlab/nanlinshi/insula/logs/xcorr_perm_${TASK_NAME}_${SUBJECT}_${BAND}_${PHASE}_${DESCRIPTION}.err
    echo "Exit code: $?"
  done
done

echo "Completed: ${TASK_NAME} ${SUBJECT}"
